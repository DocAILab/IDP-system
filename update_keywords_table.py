"""
更新领域词和交集词表（创建领域词库、旧领域添加新词、添加新领域、删除某领域）。
输入：
    operation: 要进行的操作，支持：create：创建领域词库；add：增加新领域/扩充旧领域词库；delete：删除某领域；check：查看领域信息
    delete_domain: 要删除的领域词（operation为delete时输入）
    file_paths: 添加的文件夹路径（operation为create、add时输入）
    save_path: 结果存储文件夹（operation为create、add、delete时输入）
    embedding_path: 词嵌入文件目录，对应100000-small-modi.txt文件（operation为create、add时输入）
    keywords_path: 领域关键词文件路径，domain_keywords.txt（operation为add、delete、check时输入）
    intersection_path: 领域间关键词交集文件路径，intersection.txt（operation为add、delete、check时输入）
输出：
    base.log: 记录基本信息的文件，保存有各领域非交关键词提取个数、交集词提取个数
    domain_keywords.txt: 记录各领域关键词（及对应词向量）的文件，格式为：dict{domain{keyword: [embedding]}}
    intersection.txt: 记录领域间交集词的文件，格式为：set(keywords)
"""

import os
import paddleocr
from paddleocr import PPStructure
import jieba
from utils import get_embedding_table, get_embedding, get_filename, get_domain_keywords, get_intersection

def get_all_filename(paths, ocr, table_engine):
    """
        分别读入各领域所有文件名/标题。
    :param paths: 各领域文件的文件路径
                eg: paths = {'industry': '/home/a',
                             'hospital': '/home/b',
                             'government': '/home/c'}
    :param ocr: OCR模型
    :param table_engine: 版面识别模型
    :return: 各领域下的文件名/标题列表，dict{domain: [titles]}
    """
    filenames = {}
    for domain, path in paths.items():
        filenames[domain] = []
        for fname in os.listdir(path):
            _, fname = get_filename(os.path.join(path, fname), ocr, table_engine)
            filenames[domain].append(fname)

    return filenames


def create(file_paths, embeddings, ocr, table_engine):
    """
        创建领域词库：分词，过滤，求交集，删交集，剩余作为领域关键词，并向量化。
    :param file_paths: 文件路径表，dict{domain: str(path)}
    :param embeddings: 词嵌入模型
    :param ocr: OCR模型
    :param table_engine: 版面识别模型
    :return: 各领域关键词表，dict{domain{keyword: [embedding]}}；领域间交集词表，set(keywords)
    """
    filenames_dict = get_all_filename(file_paths, ocr, table_engine)
    domain_keywords = {}
    # 分词，过滤
    for domain, filenames in filenames_dict.items():
        words = set()
        for filename in filenames:
            for word in jieba.lcut(filename, cut_all=True):
                if len(word) > 1:
                    words.add(word)
        domain_keywords[domain] = words

    # 求所有领域的共有交集。即剩余元素仍有可能在多个领域出现，但不可能同时在所有领域出现
    intersection = set()
    for i, words in enumerate(domain_keywords.values()):
        if i == 0:
            intersection = words
        else:
            intersection = intersection & words

    # 删交集
    for domain, words in domain_keywords.items():
        domain_keywords[domain] = words - intersection

    # 向量化
    for domain, words in domain_keywords.items():
        words_vec_dict = {}
        for word in words:
            words_vec_dict[word], _ = get_embedding(word, embeddings)

        domain_keywords[domain] = words_vec_dict

    return domain_keywords, intersection


def add(file_paths, embeddings, domain_keywords, intersection, ocr, table_engine):
    """
        添加新词：
            读入新领域&文件列表（同时一个或多个领域）
            非中文文件名的需要标题抽取
            分词、去停用词、去重、去交集词 -> 合并到对应领域
            更新交集词，后再更新领域词
    :param file_paths: 文件路径表，dict{domain: str(path)}
    :param embeddings: 词嵌入表
    :param domain_keywords: 领域关键词表
    :param intersection: 交集词表
    :param ocr: OCR模型
    :param table_engine: 版面识别模型
    :return: 更新后的领域关键词表和交集词表
    """
    
    domain_keywords_set = {}  # 领域关键词集合
    new_domain_keywords_set = {}  # 新领域关键词集合
    new_intersection = {}  # 新交集词集合

    filenames_dict = get_all_filename(file_paths, ocr, table_engine)  # 读入文件名/标题

    # 读入领域关键词集合（只保留关键词，不读入词向量）
    for domain, keywords in domain_keywords.items():
        domain_keywords_set[domain] = set(keywords.keys())

    # 分词，过滤，创建新领域词集合
    for domain, filenames in filenames_dict.items():
        words = set()
        for filename in filenames:
            for word in jieba.lcut(filename, cut_all=True):
                if len(word) > 1 and word not in intersection:
                    words.add(word)
        new_domain_keywords_set[domain] = words

    # 与旧领域词集合并
    for domain, words in new_domain_keywords_set.items():
        if domain in domain_keywords_set:
            domain_keywords_set[domain] |= words
        else:
            domain_keywords_set[domain] = words

    # 获取新交集词
    for i, words in enumerate(domain_keywords_set.values()):
        if i == 0:
            new_intersection = words.copy()
        else:
            new_intersection &= words

    # 删除领域词中的新交集词
    for domain, words in domain_keywords_set.items():
        domain_keywords_set[domain] = words - new_intersection

    # 更新交集词
    new_intersection |= intersection

    # 向量化
    new_domain_keywords = {}
    for domain, words in domain_keywords_set.items():
        words_vec_dict = {}
        for word in words:
            words_vec_dict[word], _ = get_embedding(word, embeddings)

        new_domain_keywords[domain] = words_vec_dict

    return new_domain_keywords, new_intersection


def delete(domain_keywords, domain_name):
    """
        删除指定领域的词表。
    :param domain_keywords: 领域关键词表
    :param domain_name: 要删除的领域，str
    :return: 若领域存在并删除成功，则返回True；否则返回False
    """
    if domain_name not in domain_keywords:
        return False
    del domain_keywords[domain_name]
    return True


def check(domain_keywords, intersection):
    """
        获取领域个数、交集词个数、各领域下关键词个数列表。
    :param domain_keywords: 领域词表
    :param intersection: 交集词表
    :return: 领域个数，int；交集词个数，int；各领域下关键词个数列表，[(domain, words_num)]
    """
    return len(domain_keywords), len(intersection), [(domain, len(words)) for domain, words in domain_keywords.items()]


def save(save_path, domain_keywords, intersection):
    """
        记录基本信息，并保存领域关键词和交集词到文件中。
    :param save_path: log存储路径
    :param domain_keywords: 领域关键词表
    :param intersection: 交集词表
    :return: 存储基本信息、关键词表、交集词表到本地，无返回值
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 记录基础信息：使用的各领域文件个数，各领域关键词个数，交集个数
    with open(os.path.join(save_path, 'base.log'), 'w', encoding='utf-8') as f:
        f.write('各领域非交关键词个数：\n')
        for domain, keywords in domain_keywords.items():
            f.write('{}: {}'.format(domain, len(keywords)) + '\n')
        f.write('\n')

        f.write('领域间交集关键词个数：' + str(len(intersection)) + '\n')

    # 记录领域关键词
    with open(os.path.join(save_path, 'domain_keywords.txt'), 'w', encoding='utf-8') as f:
        f.write(str(domain_keywords))

    # 记录交集
    with open(os.path.join(save_path, 'intersection.txt'), 'w', encoding='utf-8') as f:
        f.write(str(intersection))
# Ａ农、林、牧、渔业；
# Ｂ采矿业；
# Ｃ制造业；
# Ｄ电力、热力、燃气及水生产和供应业；
# Ｅ建筑业；
# Ｆ交通运输、仓储和邮政业；
# Ｇ信息传输、软件和信息技术服务业；
# Ｈ批发和零售业；
# Ｉ住宿和餐饮业；
# Ｊ金融业；
# Ｋ房地产业；
# Ｌ租赁和商务服务业；
# Ｍ科学研究和技术服务业；
# Ｎ水利、环境和公共设施管理业；
# Ｏ居民服务、修理和其他服务业；
# Ｐ教育；
# Ｑ卫生和社会工作；
# Ｒ文化、体育和娱乐业；
# Ｓ公共管理、社会保障和社会组织；
# Ｔ国际组织。
#     file_paths = {'A': r'/usr/local/etc/ie_flow/partfile/分级分类20230612/A',#industry personal sechool government
#                   'B': r'/usr/local/etc/ie_flow/partfile/分级分类20230612/B',
#                   'C': r'/usr/local/etc/ie_flow/partfile/分级分类20230612/C',
#                   'D': r'/usr/local/etc/ie_flow/partfile/分级分类20230612/D',
#                   'E': r'/usr/local/etc/ie_flow/partfile/分级分类20230612/E',
#                   'F': r'/usr/local/etc/ie_flow/partfile/分级分类20230612/F',
#                   'G': r'/usr/local/etc/ie_flow/partfile/分级分类20230612/G',
#                   'H': r'/usr/local/etc/ie_flow/partfile/分级分类20230612/H',
#                   'I': r'/usr/local/etc/ie_flow/partfile/分级分类20230612/I',
#                   'J': r'/usr/local/etc/ie_flow/partfile/分级分类20230612/J',
#                   'K': r'/usr/local/etc/ie_flow/partfile/分级分类20230612/K',
#                   'L': r'/usr/local/etc/ie_flow/partfile/分级分类20230612/L',
#                   'M': r'/usr/local/etc/ie_flow/partfile/分级分类20230612/M',
#                   'N': r'/usr/local/etc/ie_flow/partfile/分级分类20230612/N',
#                   'O': r'/usr/local/etc/ie_flow/partfile/分级分类20230612/O',
#                   'P': r'/usr/local/etc/ie_flow/partfile/分级分类20230612/P',
#                   'Q': r'/usr/local/etc/ie_flow/partfile/分级分类20230612/Q',
#                   'R': r'/usr/local/etc/ie_flow/partfile/分级分类20230612/R',
#                   'S': r'/usr/local/etc/ie_flow/partfile/分级分类20230612/S',
#                   'T': r'/usr/local/etc/ie_flow/partfile/分级分类20230612/T'}

    # file_paths = {'Q': r'/usr/local/etc/ie_flow/partfile/医疗',#industry personal sechool government
    #               'C': r'/usr/local/etc/ie_flow/partfile/工业',
    #               'P': r'/usr/local/etc/ie_flow/partfile/学校',
    #               'F': r'/usr/local/etc/ie_flow/partfile/运输',
    #               'S': r'/usr/local/etc/ie_flow/partfile/政府数据'}  # 文件夹路径
if __name__ == "__main__":
    operation = 'add'  # 要进行的操作，支持：create：创建领域词库；add：增加新领域/扩充旧领域词库；delete：删除某领域；check：查看领域信息
    delete_domain = r''  # operation为delete时，输入要删除的领域词
    file_paths = {'Q': r'/usr/local/etc/ie_flow/partfile/医疗',#industry personal sechool government
                  'C': r'/usr/local/etc/ie_flow/partfile/工业',
                  'P': r'/usr/local/etc/ie_flow/partfile/学校',
                  'F': r'/usr/local/etc/ie_flow/partfile/运输',
                  'S': r'/usr/local/etc/ie_flow/partfile/政府数据'}  # 文件夹路径
    save_path = r'./configuration/'  # 结果存储文件夹

    embedding_path = r'./configuration/100000-small-modi.txt'  # 词嵌入文件目录，对应100000-small-modi.txt文件
    keywords_path = r'./configuration/domain_keywords.txt'  # 领域关键词文件路径，domain_keywords.txt
    intersection_path = r'./configuration/intersection.txt'  # 领域间关键词交集文件路径，intersection.txt

    if operation == 'create':
        # 创建领域词库
        print('[create] Init...')
        embeddings = get_embedding_table(embedding_path)
        table_engine = PPStructure(table=False, ocr=False, show_log=False)
        ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="ch")

        print('Processing...')
        domain_keywords, intersection = create(file_paths, embeddings, ocr=ocr, table_engine=table_engine)
        save(save_path, domain_keywords, intersection)

        print('领域个数：{}'.format(len(domain_keywords)))
        print([(domain, len(words)) for domain, words in domain_keywords.items()])
        print('Done!')

    elif operation == 'add':
        # 增加新领域/扩充旧领域词库
        print('[add] Init...')
        embeddings = get_embedding_table(embedding_path)
        domain_keywords = get_domain_keywords(keywords_path)
        intersection = get_intersection(intersection_path)
        table_engine = PPStructure(table=False, ocr=False, show_log=False)
        ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="ch")

        print('Processing...')
        print('更新前领域个数：{}'.format(len(domain_keywords)))
        print([(domain, len(words)) for domain, words in domain_keywords.items()])

        # 更新领域词
        new_domain_keywords, new_intersection = add(file_paths, embeddings, domain_keywords, intersection, ocr=ocr,
                                                    table_engine=table_engine)
        save(save_path, new_domain_keywords, new_intersection)

        print('更新后领域个数：{}'.format(len(new_domain_keywords)))
        print([(domain, len(words)) for domain, words in new_domain_keywords.items()])
        print('Done!')

    elif operation == 'delete':
        # 删除某个领域
        print('[delete] Init...')
        domain_keywords = get_domain_keywords(keywords_path)
        intersection = get_intersection(intersection_path)
        if delete(domain_keywords, delete_domain):
            save(save_path, domain_keywords, intersection)
        else:
            print('领域名：“{}”不存在！'.format(delete_domain))

        print('更新后领域个数：{}'.format(len(domain_keywords)))
        print([(domain, len(words)) for domain, words in domain_keywords.items()])
        print('Done!')

    elif operation == 'check':
        # 查看领域信息
        print('[check] Init...')
        domain_keywords = get_domain_keywords(keywords_path)
        intersection = get_intersection(intersection_path)
        domain_num, intersection_num, domain_info = check(domain_keywords, intersection)

        print('领域个数：{}'.format(domain_num))
        print('交集词个数：{}'.format(intersection_num))
        print('各领域下关键词个数：{}'.format(domain_info))

    else:
        print('输入的操作无效：{}'.format(operation))

