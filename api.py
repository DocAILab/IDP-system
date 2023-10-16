from paddlenlp import Taskflow

import pickle
import random
import string

import numpy as np
from numpy import array, uint64
import logging
import math
import multiprocessing
import utils
from utils import *
import pathlib

import difflib
from multiprocessing import Pool, Manager
import hashlib
from nltk import ngrams
from datasketch import MinHash, MinHashLSH
import time


from paddleocr import PPStructure, paddleocr

from update_keywords_table import save, create, add, delete, delete2, change, check_shuju
import math
import faiss
import numpy as np
import jieba


class Update_Keywords:
    def __init__(self, file_path, save_path=r'./configuration/', embedding_path=r'./configuration/100000-small-modi.txt',
                 keywords_path=r'./configuration/domain_keywords.txt', intersection_path=r'./configuration/intersection.txt'):
        self.file_paths = file_path
        # {'Q': r'/usr/local/etc/ie_flow/partfile/医疗',  # industry personal sechool government
        #               'C': r'/usr/local/etc/ie_flow/partfile/工业',
        #               'P': r'/usr/local/etc/ie_flow/partfile/学校',
        #               'F': r'/usr/local/etc/ie_flow/partfile/运输',
        #               'S': r'/usr/local/etc/ie_flow/partfile/政府数据'}  # 文件夹路径
        self.save_path =save_path# 结果存储文件夹
        self.embedding_path = embedding_path  # 词嵌入文件目录，对应100000-small-modi.txt文件
        self.keywords_path = keywords_path # 领域关键词文件路径，domain_keywords.txt
        self.intersection_path = keywords_path  # 领域间关键词交集文件路径，intersection.txt

    def creat(self):
        # 创建领域词库
        print('[create] Init...')
        embeddings = get_embedding_table(self.embedding_path)
        table_engine = PPStructure(table=False, ocr=False, show_log=False)
        ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="ch")

        print('Processing...')
        domain_keywords, intersection = create(self.file_paths, embeddings, ocr=ocr, table_engine=table_engine)
        save(self.save_path, domain_keywords, intersection)

        print('领域个数：{}'.format(len(domain_keywords)))
        print([(domain, len(words)) for domain, words in domain_keywords.items()])
        print('Done!')

    def add(self,file_path):
        # 增加新领域/扩充旧领域词库
        print('[add] Init...')
       self.file_paths = file_path
        embeddings = get_embedding_table(self.embedding_path)
        domain_keywords = get_domain_keywords(self.keywords_path)
        intersection = get_intersection(self.intersection_path)
        table_engine = PPStructure(table=False, ocr=False, show_log=False)
        ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="ch")

        print('Processing...')
        print('更新前领域个数：{}'.format(len(domain_keywords)))
        print([(domain, len(words)) for domain, words in domain_keywords.items()])

        # 更新领域词
        new_domain_keywords, new_intersection = add(self.file_paths, embeddings, domain_keywords, intersection, ocr=ocr,
                                                    table_engine=table_engine)
        save(self.save_path, new_domain_keywords, new_intersection)

        print('更新后领域个数：{}'.format(len(new_domain_keywords)))
        print([(domain, len(words)) for domain, words in new_domain_keywords.items()])
        print('Done!')

    def delete1(self,delete_domain):
        # 删除某个领域
        print('[delete1] Init...')
        domain_keywords = get_domain_keywords(self.keywords_path)
        intersection = get_intersection(self.intersection_path)
        if delete(domain_keywords, delete_domain):
            save(self.save_path, domain_keywords, intersection)
        else:
            print('领域名：“{}”不存在！'.format(delete_domain))

        print('更新后领域个数：{}'.format(len(domain_keywords)))
        print([(domain, len(words)) for domain, words in domain_keywords.items()])
        print('Done!')

    def delete2(self,delete_domain,delete_keywords):
        # 删除某个领域内的领域词
        print('[delete2] Init...')
        domain_keywords = get_domain_keywords(self.keywords_path)
        intersection = get_intersection(self.intersection_path)
        if delete2(domain_keywords,intersection, delete_domain,delete_keywords):
            save(self.save_path, domain_keywords, intersection)
        else:
            print('领域名：“{}”不存在！'.format(delete_domain))

        print('更新后领域个数：{}'.format(len(domain_keywords)))
        print([(domain, len(words)) for domain, words in domain_keywords.items()])
        print('Done!')

    def change(self,new_domain,old_domain):
        # 修改领域名
        print('[change] Init...')
        domain_keywords = get_domain_keywords(self.keywords_path)
        intersection = get_intersection(self.intersection_path)
        if change(domain_keywords, old_domain,new_domain ):
            save(self.save_path, domain_keywords, intersection)
        else:
            print('领域名：“{}”不存在！'.format(old_domain))

        print('更新后领域个数：{}'.format(len(domain_keywords)))
        print([(domain, len(words)) for domain, words in domain_keywords.items()])
        print('Done!')

    def check(self):
        # 查看领域信息
        print('[check] Init...')
        domain_keywords = get_domain_keywords(self.keywords_path)
        intersection = get_intersection(self.intersection_path)
        domain_num, intersection_num, domain_info =check_shuju(domain_keywords, intersection)

        print('领域个数：{}'.format(domain_num))
        print('交集词个数：{}'.format(intersection_num))
        print('各领域下关键词个数：{}'.format(domain_info))

"""
获取某标题的领域
creat_index 依据领域词数据集创建索引index，保存领域词id所属领域的词典key_d
get_domain_type 获取领域
索引index的使用：
搜索相邻词：D,I=index.search(nq, 5)，nq是要查询的矩阵，5表示每个词返回相似度前5的词，返回D相似度矩阵，I搜素到的领域词id矩阵
nq的生成：每个要查询的词向量占一行，为使D返回余弦相似度，之后要对该矩阵正则化
正则化：faiss.normalize_L2(nq)
"""
def creat_index(domain_keywords):
    """
        基于领域词数据集初始化faiss索引
    :param domain_keywords: 领域关键词
    :return:
        index: faiss索引
        key_d: { ‘A’:[1,2,3],'B':[5,6,7]}  用于确定领域词位置
    """
    vector_list = list()
    key_d = dict()
    idx = 0
    zeronum = 0
    for domain, keywords in domain_keywords.items():
        key_d[domain] = list()
        for k, v in keywords.items():
            flag = False
            for i in v:
                if not math.isclose(i, 0):
                    flag = True
                    break
            if flag:
                vector_list.append(np.array(v).astype(np.float32))
                key_d[domain].append(idx)
                idx = idx + 1
            else:
                zeronum += 1
    v_matrix = np.array(vector_list).astype('float32')
    index = faiss.IndexFlatIP(200)  # 初始化faiss，200维向量，索引方式为计算余弦相似度
    # 正则化
    v_matrix = v_matrix.copy() / np.linalg.norm(v_matrix)
    faiss.normalize_L2(v_matrix)
    # 注入索引矩阵
    index.add(v_matrix)
    return index, key_d


def get_domain_type(text, embeddings, domain_keywords, intersection, index, key_d):
    """
        基于文件名/标题文本的领域分类。
    :param text: 文件名/标题
    :param embeddings: 词嵌入表
    :param domain_keywords: 领域关键词
    :param intersection: 领域关键词交集
    :param index: faiss创建的索引
    :param key_d: key_d: { ‘A’:[1,2,3],'B':[5,6,7]}  用于确定领域词位置
    :return:
        domain_pred_res: 领域预测结果，str
        simi_sum / match_num: 预测相似度/置信度，float
        words: 文件名/标题关键词提取结果，set
    """

    flag = False
    # 分词，过滤，删交集
    words = set()
    for word in jieba.lcut(text, cut_all=True):
        if len(word) > 1:
            words.add(word)
    words = words - intersection
    # # 创建索引
    # # 将数据集转化为索引矩阵v_matrix,key_d是确定关键词属于哪个领域的字典 例:{A:[1,2,3],B:[4,5,6]}
    # vector_list = list()
    # key_d = dict()
    # idx = 0
    # zeronum = 0
    # for domain, keywords in domain_keywords.items():
    #     key_d[domain] = list()
    #     for k, v in keywords.items():
    #         flag = False
    #         for i in v:
    #             if not math.isclose(i, 0):  # 去除零向量
    #                 flag = True
    #                 break
    #         if flag:
    #             vector_list.append(np.array(v).astype(np.float32))
    #             key_d[domain].append(idx)
    #             idx = idx + 1
    #         else:
    #             zeronum += 1
    # v_matrix = np.array(vector_list).astype('float32')
    #
    # index = faiss.IndexFlatIP(200)  # 初始化faiss，200维向量，索引方式为计算余弦相似度
    # # 正则化
    # v_matrix = v_matrix.copy() / np.linalg.norm(v_matrix)
    # faiss.normalize_L2(v_matrix)
    # # 注入索引矩阵
    # index.add(v_matrix)

    # 计算与各领域的匹配情况
    # 每个领域都get最大的(一个或多个)，然后保留最大的一个或多个作为匹配结果
    domain_pred = {}  # 领域：[匹配个数，总相似度]
    domain_matched = {}  # 领域：[匹配到的关键词]
    for domain in domain_keywords.keys():
        domain_pred[domain] = [0, 0]
        domain_matched[domain] = []
    nq = list()  # 需要进行查询的向量
    for word in words:
        # 首先尝试直接完全匹配
        fully_matched = False
        for domain, keywords in domain_keywords.items():
            if word in keywords:
                domain_pred[domain][0] += 1  # 只按个数
                domain_pred[domain][1] += 1  # 个数乘以相似度
                fully_matched = True
                continue

        # 若完全匹配上，则其他领域也仅需判断是否可完全匹配即可，无需计算其他的相似度
        # 否则，收集该词进入下一步faiss相似度匹配
        if not fully_matched:
            word_vec, matched = get_embedding(word, embeddings)
            if matched:  # 可向量化时，再继续计算相似度，否则跳过
                xb = list(word_vec)
                nq.append(xb)
                flag = True

    if flag:
        # faiss相似度查询
        nq = np.array(nq).astype('float32')
        nq = nq.copy() / np.linalg.norm(nq)
        # 正则化
        faiss.normalize_L2(nq)
        D, I = index.search(nq, 5)  # 返回最近1个向量的距离和索引
        for i, row in enumerate(I):
            for j, item in enumerate(row):
                for k, v in key_d.items():
                    if item in v and D[i][j] > 0.5:
                        domain_pred[k][0] += 1
                        domain_pred[k][1] += D[i][j]

    # 确定类别。匹配相似度大者优先，各领域均为0时，返回“others”
    domain_pred_res = 'others'
    match_num = -1
    simi_sum = 0
    for k, v in domain_pred.items():
        if v[1] == 0:
            continue
        if v[1] > simi_sum:
            match_num = v[0]
            simi_sum = v[1]
            domain_pred_res = k

    return domain_pred_res, simi_sum / match_num, words


def encryption(mode, index):
    """
    计算单个文件的hash值。
    :param mode: 使用的hash函数 int
    :param index: 对应文件数据在ans中的下标 int
    :return: 填充之后的ans元素 hash3字典。
    """
    global ans
    if mode == 1:
        hashKernel = hashlib.md5()
    elif mode == 2:
        hashKernel = hashlib.sha1()
    elif mode == 3:
        hashKernel = hashlib.sha3_256()

    fingerprint1 = generate_ngram_lsh_fingerprint(ans[index].domain_pred)
    fingerprint2 = generate_ngram_lsh_fingerprint(str(ans[index].domain_pred_match_num))
    ans[index].hash1 = fingerprint1
    ans[index].hash2 = fingerprint2
    print(fingerprint1)
    tmp = {}
    for ly in ans[index].domain_matched:
        for word in ans[index].domain_matched[ly]:
            if ly not in tmp:
                tmp[ly] = [[generate_ngram_lsh_fingerprint(word[0]),
                            generate_ngram_lsh_fingerprint(word[1])]]
            else:
                tmp[ly].append(
                    [generate_ngram_lsh_fingerprint(word[0]), generate_ngram_lsh_fingerprint(word[1])])
    return ans[index], tmp


def sort(file_hash1, file_hash2, dict1, dict2):
    """
    计算两个文件之间的总相似度。
    :param file_hash1: 存储了第一个文件相关hash信息的块 FileHash
    :param file_hash2: 存储了第二个文件相关hash信息的块 FileHash
    :param dict1: 存储了第一个文件相关hash3信息的字典 dict
    :param dict1: 存储了第二个文件相关hash3信息的字典 dict
    :return: 相似度。
    """
    similarity1 = calculate_similarity(file_hash1.hash1, file_hash2.hash1)
    similarity2 = calculate_similarity(file_hash1.hash2, file_hash2.hash2)
    three = 0
    three_all = 0
    four = 0
    op1 = 0
    op2 = 0
    for hash_item1 in dict1:
        for hash_item2 in dict2:
            if hash_item1 == hash_item2:
                a = len(dict1[hash_item1])
                b = len(dict2[hash_item2])
                op1 = op1 + (a <= b and a or b)
                op2 = op2 + (a >= b and a or b)
                for tmp1 in dict1[hash_item1]:
                    for tmp2 in dict2[hash_item2]:
                        if (tmp1[0] == tmp2[0]).all():
                            three += calculate_similarity(tmp1[1], tmp2[1])
                            print(three)
                            three_all += 1
    if op2 != 0:
        four = op1 / op2

    if three_all != 0:
        ro = (similarity1 * 3) + (similarity2 * 1) + 1.0 * (three * 4) / three_all + four
        ro = ro / 9
    else:
        ro = (similarity1 * 3) + (similarity2 * 1) + four
        ro = ro / 5
    print(similarity1)
    print(similarity2)
    print(three)
    print(three_all)
    print(four)
    print("两文件的相似度为 :")
    print(ro)

    if ro > 0.75:
        return "相似"
    elif ro > 0.5:
        return "存在关联"
    else:
        return "无关系"

def get_file_finger(file_dir, out_dir):
    """
        计算文件指纹
    :param file_dir: 输入文件路径，可为单个文件 str
    :param out_dir: 输出路径，为目录 str
    :return: 无
    """
    manager = Manager()
    ans = manager.list()
    log_base_dir = r'D:\OCR\OCR_test\OCR_test\data'  # 结果输出路径
    table_dir = r'D:\OCR\OCR_test\OCR_test\data'  # 表格抽取结果输出路径（若无需抽取表格，则不用填）

    embedding_path = r'D:\OCR\OCR_test\OCR_test\configuration\100000-small-modi.txt'  # 词嵌入文件路径，对应100000-small-modi.txt文件
    keywords_path = r'D:\OCR\OCR_test\OCR_test\configuration\domain_keywords.txt'  # 领域关键词文件路径，domain_keywords.txt
    intersection_path = r'D:\OCR\OCR_test\OCR_test\configuration\intersection.txt'  # 领域间关键词交集文件路径，intersection.txt

    table_extract = False  # 是否抽取表格
    print_info = True  # 是否输出每个文件的结果信息
    os.makedirs(log_base_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    embeddings = get_embedding_table(embedding_path)  # 读入词嵌入，对应100000-small-modi.txt文件
    domain_keywords = get_domain_keywords(keywords_path)  # 读入领域关键词，domain_keywords.txt
    intersection = get_intersection(intersection_path)  # 读入交集关键词，intersection.txt

    ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="ch")  # 初始化ocr模型
    table_engine = PPStructure(table=False, ocr=False, show_log=True)  # 初始化版面识别模型
    uie_dict = {}  # 初始化UIE模型
    for domain in schemas_dict.keys():
        uie_dict[domain] = {}
        for schema_type in schemas_dict[domain].keys():
            uie_dict[domain][schema_type] = Taskflow("information_extraction", model='uie-base',
                                                     schema=schemas_dict[domain][schema_type])
    if os.path.isdir(file_dir):
        # 多文件信息抽取
        main_for_multiprocess(file_dir, log_base_dir, embeddings, domain_keywords, intersection, ocr, table_engine,
                              uie_dict, table_extract, table_dir, print_info)
    else:
        # 单个文件信息抽取
        main(file_dir, log_base_dir, embeddings, domain_keywords, intersection, ocr, table_engine, uie_dict,
             table_extract=table_extract, table_dir=table_dir, print_info=True)

    ans[0], dict1 = encryption(0, 0)
    ans[1], dict2 = encryption(0, 1)
    sort(ans[0], ans[1], dict1, dict2)


def check_file_finger(file_dir, check_file_dir, out_dir):
    """
        检查一个文件是否与某指纹相似
    :param file_dir: 输入文件路径，可为单个文件 str
    :param check_file_dir: 输入已保存指纹的路径 str
    :param out_dir: 输出路径，为目录 str
    :return: 是否为敏感文件 bool
    """
    manager = Manager()
    ans = manager.list()
    log_base_dir = r'D:\OCR\OCR_test\OCR_test\data'  # 结果输出路径
    table_dir = r'D:\OCR\OCR_test\OCR_test\data'  # 表格抽取结果输出路径（若无需抽取表格，则不用填）

    embedding_path = r'D:\OCR\OCR_test\OCR_test\configuration\100000-small-modi.txt'  # 词嵌入文件路径，对应100000-small-modi.txt文件
    keywords_path = r'D:\OCR\OCR_test\OCR_test\configuration\domain_keywords.txt'  # 领域关键词文件路径，domain_keywords.txt
    intersection_path = r'D:\OCR\OCR_test\OCR_test\configuration\intersection.txt'  # 领域间关键词交集文件路径，intersection.txt

    table_extract = False  # 是否抽取表格
    print_info = True  # 是否输出每个文件的结果信息
    os.makedirs(log_base_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    embeddings = get_embedding_table(embedding_path)  # 读入词嵌入，对应100000-small-modi.txt文件
    domain_keywords = get_domain_keywords(keywords_path)  # 读入领域关键词，domain_keywords.txt
    intersection = get_intersection(intersection_path)  # 读入交集关键词，intersection.txt

    ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="ch")  # 初始化ocr模型
    table_engine = PPStructure(table=False, ocr=False, show_log=True)  # 初始化版面识别模型
    uie_dict = {}  # 初始化UIE模型
    for domain in schemas_dict.keys():
        uie_dict[domain] = {}
        for schema_type in schemas_dict[domain].keys():
            uie_dict[domain][schema_type] = Taskflow("information_extraction", model='uie-base',
                                                     schema=schemas_dict[domain][schema_type])
    if os.path.isdir(file_dir):
        # 多文件信息抽取
        main_for_multiprocess(file_dir, log_base_dir, embeddings, domain_keywords, intersection, ocr, table_engine,
                              uie_dict, table_extract, table_dir, print_info)
    else:
        # 单个文件信息抽取
        main(file_dir, log_base_dir, embeddings, domain_keywords, intersection, ocr, table_engine, uie_dict,
             table_extract=table_extract, table_dir=table_dir, print_info=True)
    # 读取 + 遍历对比
    ans[0], dict1 = encryption(0, 0)
    ans.append(FileHash())
    ans[1].hash1 = data['hash1']
    ans[1].hash2 = data['hash2']
    dict2 = data['hash3']
    sort(ans[0], ans[1], dict1, dict2)


def file_file_check(file_dir):
    """
        计算两个文件相似度
    :param file_dir: 输入文件路径，应当为两个文件所处的目录 str
    :return: 相似度 float
    """
    manager = Manager()
    ans = manager.list()

    file_dir = r'D:\OCR\OCR_test\OCR_test\data'  # 文件路径/文件夹路径
    log_base_dir = r'D:\OCR\OCR_test\OCR_test\data'  # 结果输出路径
    table_dir = r'D:\OCR\OCR_test\OCR_test\data'  # 表格抽取结果输出路径（若无需抽取表格，则不用填）

    embedding_path = r'D:\OCR\OCR_test\OCR_test\configuration\100000-small-modi.txt'  # 词嵌入文件路径，对应100000-small-modi.txt文件
    keywords_path = r'D:\OCR\OCR_test\OCR_test\configuration\domain_keywords.txt'  # 领域关键词文件路径，domain_keywords.txt
    intersection_path = r'D:\OCR\OCR_test\OCR_test\configuration\intersection.txt'  # 领域间关键词交集文件路径，intersection.txt

    table_extract = False  # 是否抽取表格
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
    ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="ch")  # 初始化ocr模型
    table_engine = PPStructure(table=False, ocr=False, show_log=True)  # 初始化版面识别模型
    uie_dict = {}  # 初始化UIE模型
    for domain in schemas_dict.keys():
        uie_dict[domain] = {}
        for schema_type in schemas_dict[domain].keys():
            uie_dict[domain][schema_type] = Taskflow("information_extraction", model='uie-base',
                                                     schema=schemas_dict[domain][schema_type])

    main_for_multiprocess(file_dir, log_base_dir, embeddings, domain_keywords, intersection, ocr, table_engine,
                          uie_dict, table_extract, table_dir, print_info)
    ans[0], dict1 = encryption(0, 0)
    ans[1], dict2 = encryption(0, 1)
    sort(ans[0], ans[1], dict1, dict2)


def write_pickle(file_directory, data):
    """
    将数据存储为 pickle 格式的文件，随机生成文件名。

    :param file_directory:
    :param data: 要存储的数据。
    :return: 存储的文件路径和文件名。
    """
    file_name = ''.join(random.choice(string.ascii_letters) for _ in range(10)) + '.pkl'
    file_path = os.path.join(file_directory, file_name)

    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
    return file_path, file_name


# 定义函数，用于从 pickle 格式的文件中读取数据
def read_pickle(file_path):
    """
    从 pickle 格式的文件中读取数据。

    :param file_path: 要读取的文件路径。
    :return: 读取到的数据。
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def read_file(file_path, ocr, table_engine, table_extract=False, table_dir=None):
    """
        具体函数在utils.py里，这里只进行一个封装
        读入pdf/docx文件，获取文字内容
    :param file_path: 文件绝对路径
    :param ocr: OCR模型
    :param table_engine: 版面识别模型
    :param table_extract: 是否提取表格到excel
    :param table_dir: 表格存储目录
    :return: 原始文件名，str；提取文件名，str；文字读取结果，str
    """
    return utils.read_file(file_path, ocr, table_engine, table_extract=table_extract, table_dir=table_dir)


class Info_Extraction:
    def __init__(self, schemas_dict= schemas_dict_education, model= 'uie-m-base', task_path= None, use_fast= True):
        """
        初始化UIE模型字典
        :param schemas_dict: 领域词字典
        :param model: 使用paddlenlp模型名
        :param task_path: 使用微调或自定义模型路径，使用该项model设置无效
        :param use_fast: 使用FastTokenizer库加速(需要安装fast-tokenizer-python包)
        """

        self.model = model
        self.task_path = task_path
        self.use_fast = use_fast
        
        # 整合schema
        assemble_schema = True # 是否整合schema,为减小显存占用而整合
        if assemble_schema:
            self.schema_assemble_dict = self.assemble_schema(schemas_dict)
        else:
            self.schema_assemble_dict = schemas_dict
        
        # 初始化模型
        self.uie_dict = {}

        load_all_uie = False # 是否一次性加载所有uie模型
        if load_all_uie:
            self.load_all_uie()
    
    @classmethod
    def assemble_schema(cls, schemas_dict):
        """
        整合schema
        :param schemas_dict: 领域词字典
        :return: 返回集成后的schema字典（减小显存占用）
        """
        schema_assemble_dict = {}
        for domain, domian_schemas in schemas_dict.items():
            tmp = set()
            for schema in domian_schemas.values():
                tmp.update(schema)
            schema_assemble_dict[domain] = {'assemble_schemas': list(tmp)}
        return schema_assemble_dict
    
    def load_all_uie(self):
        for field in self.schema_assemble_dict.keys():
            self.update_uie(field)

    def update_uie(self, field):
        """
        增量加载uie模型
        :param field: 文档所属领域
        :return: 返回是否加载成功
        """
        # 增量加载uie模型
        if field not in self.uie_dict.keys():
            if field in self.schema_assemble_dict.keys():
                if self.task_path:
                    self.uie_dict[field] = {}
                    for schema_type in self.schema_assemble_dict[field].keys():
                        self.uie_dict[field][schema_type] = Taskflow("information_extraction", 
                                                                task_path= self.task_path, 
                                                                schema=self.schema_assemble_dict[field][schema_type],
                                                                use_fast= self.use_fast)
                else:
                    self.uie_dict[field] = {}
                    for schema_type in self.schema_assemble_dict[field].keys():
                        self.uie_dict[field][schema_type] = Taskflow("information_extraction", 
                                                                model=self.model, 
                                                                schema=self.schema_assemble_dict[field][schema_type],
                                                                use_fast= self.use_fast)
                return True
            else:
                print("领域不存在")
                return False
    
    def __call__(self, text_str, field, print_info= False):
        """
        输入纯文本和对应领域的schema字典（目前同领域下多个文件类型各有一个schema）
        field_uie为uie_dict下对应领域的uie模型
        :param text_str: 文档识别出来的纯本文字符串
        :param field: 文档所属领域
        :param print_info: 运行时是否打印信息
        :return: 返回是否是敏感文件、匹配到的schema:关键词列表组成的字典
        """
        
        # 增量加载uie模型
        self.update_uie(field)

        extract_num = 0
        extract_result = {}
        if text_str != '':
            if print_info:
                print("关键词和匹配情况")
            for schema_type in self.schema_assemble_dict[field].keys():
                ie_result = self.uie_dict[field][schema_type](text_str)
                #if print_info:
                    #print("文件类型: ", schema_type)
                if(ie_result[0]):
                    schema_num = 0
                    for schema, match_list in ie_result[0].items():
                        extract_result[schema] = []
                        schema_num += len(match_list)
                        if print_info:
                            print("    ", schema, ":", end=" ", sep="")
                            for each_match in match_list:
                                print(each_match['text'], end=" ")
                            print()
                        for each_match in match_list:
                            extract_result[schema].append(each_match['text'])
                    extract_num += schema_num
                    if print_info:
                        print("该类型下匹配到个数: ", schema_num, "\n")
                else:
                    if print_info:
                        print("该类型下未匹配到schema\n")
        if print_info:
            print("总共匹配到的个数: ", extract_num)
        
        # 判断是否敏感
        schema_num = sum([len(schemas) for schemas in self.schema_assemble_dict[field].values()])
        schema_num_avg = schema_num / len(self.schema_assemble_dict[field].keys())
        rate = len(extract_result) / schema_num_avg
        if print_info:
            print("平均匹配率: ", rate)
        mg = True if (extract_num > 10 or rate > 0.5) else False

        return extract_result, mg

def sensitive_word(file_txt, print_info= False):
    # 基于文件内容的信息抽取
    ## 机密词匹配
    have_sensitive_words = any((word in file_txt) for word in sensitive_words)
    if print_info:
        if have_sensitive_words:
            print('是否包含机密词 : 是')
        else:
            print('是否包含机密词 : 否')
    return have_sensitive_words

def extraction_classify(field, extract_result, schema_dict= schemas_dict_education_D, print_info= False):
    """
    根据抽取结果和scheam分类
    :param field: 文档所属领域
    :param extract_result: 抽取结果
    :param schema_dict: 领域词字典
    :return: 返回分类结果
    """

    classify_res = {}
    max_rate = 0
    for schema_type, schemas in schema_dict[field].items():
        match_num = 0
        repeat_num = 0
        for match_schema, match_words in extract_result.items():
            if match_schema in schemas:
                match_num += 1
                repeat_num += len(match_words)/10
        #classify_res[schema_type] = match_num + repeat_num
        classify_res[schema_type] = match_num / len(schemas) + repeat_num
        if classify_res[schema_type] > max_rate:
            max_rate = classify_res[schema_type]
    
    classify = []
    for schema_type, rate in classify_res.items():
        if rate == max_rate:
            classify.append(schema_type)

    if print_info:
        print("分类匹配")
        for schema_type, rate in classify_res.items():
            print("   ", schema_type, "匹配值: ", rate)
        print("分类结果: ", *classify, sep= " ")
    return classify

def uie_example():
    """
    uie接口使用示例（Info_Extraction、sensitive_word、extraction_classify）
    """
    # debug测试以及使用示例
    
    # 初始化uie模型,加载字典
    uie = Info_Extraction()

    # 获得领域和文本（由上一步ocr得到）
    field = '财务域'
    text = "有效张数51，作废张数2，实开金额1000，金额限制1000，票本序号123456，姓名小明"

    field2 = '人员域'
    text2 = "姓名小明，性别男，年龄18，身份证号123456789，电话号码123456789，家庭住址北京市海淀区"
    
    # 函数使用示例
    extract_result, mg = uie(text, field, print_info= True)
    have_sensitive_words = sensitive_word(text, print_info= True)
    classify = extraction_classify(field, extract_result, print_info= True)

    extract_result2, mg2 = uie(text2, field2)
    have_sensitive_words2 = sensitive_word(text2)
    classify2 = extraction_classify(field2, extract_result2)

    # 打印结果
    from pprint import pprint
    #pprint(uie.schema_assemble_dict)
    print("是否敏感:", "是" if mg else "否")
    #pprint(extract_result)
    print("是否含有敏感关键词:", "是"if have_sensitive_words else "否")
    print("分类结果:", *classify, sep= " ")

    print()

    print("是否敏感:", "是" if mg2 else "否")
    #pprint(extract_result2)
    print("是否含有敏感关键词:", "是" if have_sensitive_words2 else "否")
    print("分类结果:", *classify2, sep= " ")
