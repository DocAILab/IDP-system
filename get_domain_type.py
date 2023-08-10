import math
import faiss
import numpy as np
import jieba
def get_embedding(word, embeddings):
    """
        获取特定词向量。若匹配到了指定词，返回对应的词向量，否则，返回默认词向量（全0向量）。
    :param word: 待获取词向量的词，str
    :param embeddings: 词嵌入表，dict
    :return: 词嵌入的向量，list；是否匹配到了该词，bool；
    """
    if word in embeddings:
        return embeddings[word], True  # matched = True
    else:
        return embeddings['mean_vector'], False  # matched = False


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
