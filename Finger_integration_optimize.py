import pickle
import random
import string

import numpy as np
from numpy import array, uint64
import logging
import math
import multiprocessing
from utils import *
import pathlib

import difflib
from multiprocessing import Pool, Manager
import hashlib
from nltk import ngrams
from datasketch import MinHash, MinHashLSH
import time


##### 该文件为修改过的integration文件，为指纹提取的多线程特征提取部分#####


def callback(call):
    """
        某进程成功运行结束后，调用该函数，输出进度信息
    :param call: 进程返回值
    :return: 无
    """
    global now_num
    print('Done for {} / {}'.format(now_num, total_num))
    now_num += 1


def error_callback(err):
    """
        多进程错误捕捉
    :param err: 进程返回值
    :return: 无
    """
    print('进程运行出错：' + str(err))


#########
def main(file_path, log_base_dir, embeddings, domain_keywords, intersection, ocr, table_engine, ans,
         table_extract=False, table_dir='', print_info=True):
    """
        执行一个文件的信息抽取
    :param file_path: 文件绝对路径
    :param log_base_dir: # 结果输出路径
    :param embeddings: 词嵌入表
    :param domain_keywords: 领域关键词
    :param intersection: 交集关键词
    :param ocr: OCR模型
    :param table_engine: 版面识别模型
    :param uie_dict: 存储了预加载UIE模型的字典
    :param table_extract: 是否抽取表格
    :param table_dir: 表格抽取结果保存路径
    :param print_info: 是否输出结果信息
    :return: 若提取到了文件内容，则返回True，否则返回False
    """
    file_hash = FileHash()
    # 提取文本内容
    ori_filename, file_name = read_file(file_path, ocr, table_engine)
    time_start = time.time()
    # 领域分类
    domain_pred, simi, words, domain_matched = get_domain_type(file_name, embeddings, domain_keywords, intersection)
    # domain_pred_res: 领域预测结果，str
    file_hash.domain_pred = domain_pred
    # simi_sum / match_num: 预测相似度 / 置信度，float
    file_hash.domain_pred_match_num = simi
    # words: 文件名 / 标题关键词提取结果，set 其中不包括跨领域词
    file_hash.words = words
    # domain_matched: 各领域下匹配到的文本关键词，dict {领域: [文本关键词, 匹配的最佳领域词, 相似度]} 其中包括了非预测领域的结果
    file_hash.domain_matched = domain_matched
    time_end = time.time()
    time_c = time_end - time_start
    print("抽取文本内容与保存用时", time_c, 's')
    # domain_pred = 'sechool'
    schema_key = domain_pred
    # if domain_pred not in schemas_dict:
    #     schema_key = 'others'  # 预测领域若没有schema，则使用'others'对应的schema抽取信息
    #     print('There is no corresponding schema in this field!')
    #     file_hash.use_other = True
    #
    # # 信息抽取
    # file_MG_level, have_sensitive_words, res = extract_information(file_name, file_txt, log_base_dir, field=domain_pred,
    #                                                                schemas=schemas_dict[schema_key], uie_dict=uie_dict,
    #                                                                ori_filename=ori_filename)
    # extraction_result_list_length = len(res)
    # loop = 0
    # while loop < extraction_result_list_length:
    #     if loop % 2 == 0:
    #         single_schema_extraction_key = res[loop]['schema']
    #     if loop % 2 != 0:
    #         single_schema_extraction_results = res[loop]
    #         file_hash.file_key_word[single_schema_extraction_key] = single_schema_extraction_results
    #     loop += 1
    #
    ans.append(file_hash)
    # print(len(ans))
    if print_info:
        print('文件名：{}'.format(ori_filename))
        print('提取文件名：{}'.format(file_name))
        print('预测领域：{}'.format(domain_pred))
        # print('敏感等级：{}'.format(file_MG_level))
        # print('是否包含机密词：{}'.format(have_sensitive_words))
        print('文本关键词：{}'.format(words))
        print('领域匹配信息：{}'.format(domain_matched))

    return True


def main_for_list(file_path_list, log_base_dir, embeddings, domain_keywords, intersection, ans, table_extract=False,
                  table_dir='', print_info=True):
    """
        执行一组文件的信息抽取
    :param ans:
    :param file_path_list: 文件绝对路径列表
    :param log_base_dir: # 结果输出路径
    :param embeddings: 词嵌入表
    :param domain_keywords: 领域关键词
    :param intersection: 交集关键词
    :param table_extract: 是否抽取表格
    :param table_dir: 表格抽取结果保存路径
    :param print_info: 是否输出结果信息
    :return: 无
    """
    time_start = time.time()
    ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="ch")  # 初始化ocr模型
    table_engine = PPStructure(table=False, ocr=False, show_log=True)  # 初始化版面识别模型
    time_end = time.time()
    time_c = time_end - time_start  # 运行所花时间
    print('初始化ocr与版面识别模型用时', time_c, 's')
    for file_path in file_path_list:
        try:
            have_words = main(file_path, log_base_dir, embeddings, domain_keywords, intersection, ocr, table_engine,
                              ans, print_info=print_info)

            if not have_words:
                print('[未提取到内容]: ' + file_path.split('/')[-1])
        except Exception as e:
            print(file_path.split('/')[-1] + '[Main Error] : ' + str(e))
            continue


def main_for_multiprocess(file_dir, log_base_dir, embeddings, domain_keywords, intersection, ocr, table_engine,
                          uie_dict, table_extract=False, table_dir='', print_info=True):
    """
        多进程执行对应目录下所有文件的信息抽取
    :param file_dir: 文件夹路径
    :param log_base_dir: # 结果输出路径
    :param embeddings: 词嵌入表
    :param domain_keywords: 领域关键词
    :param intersection: 交集关键词
    :param ocr: OCR模型
    :param table_engine: 版面识别模型
    :param uie_dict: 存储了预加载UIE模型的字典
    :param table_extract: 是否抽取表格
    :param table_dir: 表格抽取结果保存路径
    :param print_info: 是否输出结果信息
    :return: 无
    """
    global total_num
    filenames = os.listdir(file_dir)
    pool = multiprocessing.Pool(processes=min(10, len(filenames)))  # 最多初始化10个进程

    if not os.path.isdir(os.path.join(file_dir, filenames[0])):
        # 若file_dir下为文件，则将文件分组，每组使用一个进程处理
        file_paths = list(map(lambda x: os.path.join(file_dir, x), filenames))
        split_file_paths = []
        num_in_group = max(1, math.ceil(len(file_paths) / 10))  # 最大分为10组时，每组内文件个数
        for i in range(0, len(file_paths), num_in_group):
            split_file_paths.append(file_paths[i: i + num_in_group])

        total_num = len(split_file_paths)  # 总共要执行的线程个数

        # 多进程处理
        for i in range(len(split_file_paths)):
            pool.apply_async(main_for_list, (
                split_file_paths[i], log_base_dir, embeddings, domain_keywords, intersection, ans, table_extract,
                table_dir,
                print_info), callback=callback, error_callback=error_callback)
        pool.close()
        pool.join()
    else:
        # 若file_dir下为文件夹，则按文件夹分组，每组使用一个进程处理
        total_num = len(filenames)

        for foldername in filenames:
            real_file_dir = os.path.join(file_dir, foldername)
            real_log_base_dir = os.path.join(log_base_dir, foldername)
            real_table_dir = os.path.join(table_dir, foldername)

            # 读入文件路径
            real_filenames = os.listdir(real_file_dir)
            real_file_paths = list(map(lambda x: os.path.join(real_file_dir, x), real_filenames))

            # 多进程处理
            pool.apply_async(main_for_list, (
                real_file_paths, real_log_base_dir, embeddings, domain_keywords, intersection, ans, table_extract,
                real_table_dir, print_info), callback=callback, error_callback=error_callback)
        pool.close()
        pool.join()

