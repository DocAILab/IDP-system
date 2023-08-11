"""
文件敏感识别全流程处理。
流程：
    1. 读取标题及文本内容
    2. 基于标题做领域分类
    3. 信息抽取 (基于标题的敏感判断、基于内容的机密词匹配、基于schema的抽取、基于schema抽取结果的敏感分级等)
    4. 结果保存
输入：
    file_dir: 文件路径/文件夹路径
    log_base_dir: 结果输出路径
    table_dir: 表格抽取结果输出路径（若无需抽取表格，则不用填）
    embedding_path: 词嵌入文件路径，对应100000-small-modi.txt文件
    keywords_path: 领域关键词文件路径，domain_keywords.txt
    intersection_path: 领域间关键词交集文件路径，intersection.txt
    table_extract: 是否抽取表格
    print_info: 是否输出每个文件的结果信息
输出：
    每个文件输出一个log文件，包含上述信息抽取内容。
    每个文件输出一个excel文件，包含抽取出的表格内容。（当table_extract为True时）
"""

import logging
import math
import multiprocessing
from utils import *

logging.getLogger('ppocr').setLevel(logging.ERROR)

now_num = 1
total_num = 0

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
def main(file_path, log_base_dir, embeddings, domain_keywords, intersection, ocr, table_engine, uie_dict, table_extract=False, table_dir='', print_info=True):
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
    # 提取文本内容
    ori_filename, file_name, file_txt = read_file(file_path, ocr, table_engine, table_extract=table_extract, table_dir=table_dir)

    # 领域分类
    domain_pred, simi, words, domain_matched = get_domain_type(file_name, embeddings, domain_keywords, intersection)
    schema_key = domain_pred
    if domain_pred not in schemas_dict:
        schema_key = 'others'  # 预测领域若没有schema，则使用'others'对应的schema抽取信息
        print('There is no corresponding schema in this field!')

    # 信息抽取
    file_MG_level, have_sensitive_words = extract_information(file_name, file_txt, log_base_dir, field=domain_pred, schemas=schemas_dict[schema_key], uie_dict=uie_dict, ori_filename=ori_filename)

    if print_info:
        print('文件名：{}'.format(ori_filename))
        print('提取文件名：{}'.format(file_name))
        if file_txt == '':
            print('是否成功提取文字内容：未提取到文字!')
        else:
            print('是否成功提取文字内容：成功！')
        print('预测领域：{}'.format(domain_pred))
        print('敏感等级：{}'.format(file_MG_level))
        print('是否包含机密词：{}'.format(have_sensitive_words))
        print('文本关键词：{}'.format(words))
        print('领域匹配信息：{}'.format(domain_matched))

    if file_txt == '':
        return False
    else:
        return True


def main_for_list(file_path_list, log_base_dir, embeddings, domain_keywords, intersection, table_extract=False, table_dir='', print_info=True):
    """
        执行一组文件的信息抽取
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

    for file_path in file_path_list:
        try:
            have_words = main(file_path, log_base_dir, embeddings, domain_keywords, intersection, ocr, table_engine, uie_dict, table_extract=table_extract, table_dir=table_dir, print_info=print_info)

            if not have_words:
                print('[未提取到内容]: ' + file_path.split('/')[-1])
        except Exception as e:
            print(file_path.split('/')[-1] + '[Main Error] : ' + str(e))
            continue


def main_for_multiprocess(file_dir, log_base_dir, embeddings, domain_keywords, intersection, ocr, table_engine, uie_dict, table_extract=False, table_dir='', print_info=True):
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
    global  total_num
    filenames = os.listdir(file_dir)
    pool = multiprocessing.Pool(processes=min(10, len(filenames)))  # 最多初始化10个进程

    if not os.path.isdir(os.path.join(file_dir, filenames[0])):
        # 若file_dir下为文件，则将文件分组，每组使用一个进程处理
        file_paths = list(map(lambda x: os.path.join(file_dir, x), filenames))
        split_file_paths = []
        num_in_group = max(1, math.ceil(len(file_paths) / 10))  # 最大分为10组时，每组内文件个数
        for i in range(0, len(file_paths), num_in_group):
            split_file_paths.append(file_paths[i: i+num_in_group])

        total_num = len(split_file_paths)  # 总共要执行的线程个数

        # 多进程处理
        for i in range(len(split_file_paths)):
            pool.apply_async(main_for_list, (split_file_paths[i], log_base_dir, embeddings, domain_keywords, intersection, table_extract, table_dir, print_info), callback=callback, error_callback=error_callback)
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
            pool.apply_async(main_for_list, (real_file_paths, real_log_base_dir, embeddings, domain_keywords, intersection, table_extract, real_table_dir, print_info), callback=callback, error_callback=error_callback)
        pool.close()
        pool.join()
##########


if __name__ == '__main__':
    file_dir = r'/usr/local/etc/ie_flow/partfile/运输'  # 文件路径/文件夹路径
    log_base_dir = r'/usr/local/etc/ie_flow/tiaoshi/integration_test/test2/运输'  # 结果输出路径
    table_dir = r'/usr/local/etc/ie_flow/tiaoshi/integration_test/test2/运输'  # 表格抽取结果输出路径（若无需抽取表格，则不用填）

    embedding_path = r'100000-small-modi.txt'  # 词嵌入文件路径，对应100000-small-modi.txt文件
    keywords_path = r'/usr/local/etc/ie_flow/tiaoshi/integration_test/domain_keywords.txt'  # 领域关键词文件路径，domain_keywords.txt
    intersection_path = r'/usr/local/etc/ie_flow/tiaoshi/integration_test/intersection.txt'  # 领域间关键词交集文件路径，intersection.txt

    table_extract = True  # 是否抽取表格
    print_info = True  # 是否输出每个文件的结果信息

    # 初始化
    ## 目录创建
    os.makedirs(log_base_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    ## 词表读入
    embeddings = get_embedding_table(embedding_path)  # 读入词嵌入，对应100000-small-modi.txt文件
    domain_keywords = get_domain_keywords(keywords_path)  # 读入领域关键词，domain_keywords.txt
    intersection = get_intersection(intersection_path)  # 读入交集关键词，intersection.txt

    ## 模型加载
    ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="ch",
                              det_model_dir="../configuration/ocr/ch_PP-OCRv4_det_infer",
                              rec_model_dir="../configuration/ocr/ch_PP-OCRv4_rec_infer",
                              cls_model_dir="../configuration/ocr/ch_ppocr_mobile_v2.0_cls_infer")  # 初始化ocr模型
    table_engine = PPStructure(table=False, ocr=False, show_log=True,
                               layout_model_dir="../configuration/ocr/layout/picodet_lcnet_x1_0_fgd_layout_cdla_infer",
                               table_model_dir="../configuration/ocr/table/ch_ppstructure_mobile_v2.0_SLANet_infer",
                               det_model_dir="../configuration/ocr/ch_PP-OCRv4_det_infer",
                               rec_model_dir="../configuration/ocr/ch_PP-OCRv4_rec_infer")  # 初始化版面识别模型
    uie_dict = {}  # 初始化UIE模型
    for domain in schemas_dict.keys():
        uie_dict[domain] = {}
        for schema_type in schemas_dict[domain].keys():
            uie_dict[domain][schema_type] = Taskflow("information_extraction", model='uie-base', schema=schemas_dict[domain][schema_type])

    # 执行
    if os.path.isdir(file_dir):
        # 多文件信息抽取
        main_for_multiprocess(file_dir, log_base_dir, embeddings, domain_keywords, intersection, ocr, table_engine, uie_dict, table_extract, table_dir, print_info)
    else:
        # 单个文件信息抽取
        main(file_dir, log_base_dir, embeddings, domain_keywords, intersection, ocr, table_engine, uie_dict, table_extract=table_extract, table_dir=table_dir, print_info=True)

