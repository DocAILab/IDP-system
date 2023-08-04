# 基于关键词的文本分类
# 标题分词，去除停用词
# 统计不同行业间分词的交集
# 删除标题中交集词，其他词作为行业领域的关键词，并向量化
# 对于新标题，删除停用词、交集词，计算与各行业的关键词相似度，确定行业类别

import os
import jieba
from random import shuffle
import paddle
import paddle.nn.functional as F
import math
import time

# 读入词嵌入表
def get_embedding_table(embedding_path):
    # 腾讯词向量精简版，10万词，200维
    embeddings = {}
    mean_vec = [0.0] * 200  # 计算平均向量，作为OOV词语的向量。平均不太行，偶尔会出现一个词和全部的OOV词的相似度大于0.5。直接设0即可
    with open(embedding_path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f.readlines():
            line = line.strip().split()
            word = line[0]
            embedding = list(map(lambda x: float(x), line[1:]))
            embeddings[word] = embedding
            # mean_vec = [i + j for i, j in zip(mean_vec, embedding)]
    
    # lens = len(embeddings)
    # mean_vec = [i / lens for i in mean_vec]
    embeddings['mean_vector'] = mean_vec

    return embeddings

# 获取特定词向量
def get_embedding(word, embeddings):
    if word in embeddings:
        return (embeddings[word], True)  # matched = True
    else:
        return (embeddings['mean_vector'], False)  # matched = False

# 分别读入各领域所有文件名/标题
def get_all_title(paths):
    # 各领域文件的文件路径
    # paths = {'industry': '/home/a',
    #          'hospital': '/home/b',
    #          'government': '/home/c'}
    titles = {}
    for domain, path in paths.items():
        titles[domain] = list(map(lambda x: os.path.splitext(x)[0], os.listdir(path)))
    
    return titles

# 取一部分提取词库，剩余部分作为测试集
def get_dataset(titles, train_ratio=0.7):
    train = {}
    test = {}
    for domain, title_list in titles.items():
        shuffle(title_list)
        num = int(len(title_list) * train_ratio)
        train[domain] = title_list[:num]
        test[domain] = title_list[num:]
    
    return (train, test)

# 前一部分分词，过滤，求交集，删交集，剩余作为领域关键词，并向量化
def get_domain_keywords(train, embeddings, intersection_mode='mid'):
    """
    intersection_mode: 求交集的模式
                       small: 最小交集模式，即法1，求所有领域的共有交集
                       full: 最大交集模式，即法2，求任意两领域间的交集，再取并集
                       mid: 折中模式，即法3，限定元素最多可出现n次，大于n次的作为交集
    """
    domain_keywords = {}
    # 分词，过滤
    for domain, title_list in train.items():
        words = set()
        for title in title_list:
            for word in jieba.lcut(title, cut_all=True):
                if len(word) > 1:
                    words.add(word)
        domain_keywords[domain] = words
    
    # 求交集
    intersection = set()
    if intersection_mode == 'small':
        # 法1，求所有领域的共有交集。即剩余元素仍有可能在多个领域出现，但不可能同时在所有领域出现
        # 适用于领域数少（2，3个领域）和classify2
        for i, words in enumerate(domain_keywords.values()):
            if i == 0:
                intersection = words
            else:
                intersection = intersection & words
    elif intersection_mode == 'full':
        # 法2，求任意两领域间的交集，再取并集。即剩余任何元素都只能出现一次
        # 适用于classify
        keys = list(domain_keywords.keys())
        for i in range(len(keys) - 1):
            for j in range(i + 1, len(keys)):
                intersection = intersection | (domain_keywords[keys[i]] & domain_keywords[keys[j]])
    elif intersection_mode == 'mid':
        # 法3，限定元素最多可出现n次，大于n次的作为交集。属于上述两方法的折中，且当n<=3时，等效于法1
        # 适用于任意数量领域和classify2
        threshold = math.ceil(len(domain_keywords) / 2)  # 限定元素最多可出现的次数，目前为领域个数/2，再向上取整
        word_times = {}  # 统计每个词出现的次数
        for words in domain_keywords.values():
            for w in words:
                if w in word_times:
                    word_times[w] += 1
                else:
                    word_times[w] = 1
        for k, v in word_times.items():
            if v > threshold:
                intersection.add(k)
    else:
        raise ValueError('invalid intersection mode!', intersection_mode)
    
    # 删交集
    for domain, words in domain_keywords.items():
        domain_keywords[domain] = words - intersection

    # 向量化
    for domain, words in domain_keywords.items():
        words_vec_dict = {}
        for word in words:
            words_vec_dict[word], _ = get_embedding(word, embeddings)
        
        domain_keywords[domain] = words_vec_dict
    
    return (domain_keywords, intersection)

# 测试文本分词，过滤，删交集，剩余词向量化
# 和领域关键词求相似度，依据相似度值或某领域下匹配个数确定类别（单分类或多分类）
def classify(text, embeddings, domain_keywords, intersection, match_mode='full'):
    """
    match_mode: 词语匹配模式
                greedy: 贪心模式，在某领域下匹配到即停止，不再与其他领域算相似度，对应法1
                full: 全匹配模式，与各领域均进行匹配计算，最后选取相似度最大的一个或多个结果，对应法2
    """
    # 分词，过滤，删交集
    words = set()
    for word in jieba.lcut(text, cut_all=True):
        if len(word) > 1:
            words.add(word)
    words = words - intersection

    # 计算与各领域的匹配情况
    domain_pred = {}  # 领域：[匹配个数，总相似度]
    domain_matched = {}  # 领域：[匹配到的关键词]
    for domain in domain_keywords.keys():
        domain_pred[domain] = [0, 0]
        domain_matched[domain] = []
    
    if match_mode == 'greedy':
        # 法1，当某个词在某领域完全匹配后，结束继续匹配，否则计算与所有关键词的相似度，选择相似度最大的一个。适用于领域间无交集情况
        for word in words:
            max_simi = -1
            max_domain = ''
            max_word = ''

            # 首先尝试直接完全匹配
            fully_matched = False
            for domain, keywords in domain_keywords.items():
                if word in keywords:
                    max_simi = 1
                    max_domain = domain
                    max_word = word
                    fully_matched = True
                    break
            
            # 没有匹配时，计算相似度
            if not fully_matched:
                word_vec, matched = get_embedding(word, embeddings)
                if matched:  # 可向量化时，再继续计算相似度，否则跳过
                    for domain, keywords in domain_keywords.items():
                        for k, v in keywords.items():
                            simi = F.cosine_similarity(paddle.to_tensor(word_vec), paddle.to_tensor(v), axis=0).tolist()[0]
                            if simi > max_simi:
                                max_simi = simi
                                max_domain = domain
                                max_word = k
            
            # 获取最终匹配结果
            if max_simi > 0.5:  # 阈值设置为0.5
                domain_pred[max_domain][0] += 1
                domain_pred[max_domain][1] += max_simi
                domain_matched[max_domain].append([word, max_word, max_simi])
    elif match_mode == 'full':
        # 法2，每个领域都get最大的(一个或多个)，然后保留最大的一个或多个作为匹配结果
        for word in words:
            domain_max = {}
            for k in domain_keywords.keys():
                domain_max[k] = {'max_simi': -1, 'max_word': []}
            
            # 首先尝试直接完全匹配
            fully_matched = False
            for domain, keywords in domain_keywords.items():
                if word in keywords:
                    domain_max[domain]['max_simi'] = 1
                    domain_max[domain]['max_word'] = [word]
                    fully_matched = True
                    continue
            
            # 若完全匹配上，则其他领域也仅需判断是否可完全匹配即可，无需计算其他的相似度
            # 否则，计算与各领域关键词的相似度
            if not fully_matched:
                word_vec, matched = get_embedding(word, embeddings)
                if matched:  # 可向量化时，再继续计算相似度，否则跳过
                    for domain, keywords in domain_keywords.items():
                        for k, v in keywords.items():
                            simi = F.cosine_similarity(paddle.to_tensor(word_vec), paddle.to_tensor(v), axis=0).tolist()[0]
                            if simi > domain_max[domain]['max_simi']:
                                domain_max[domain]['max_simi'] = simi
                                domain_max[domain]['max_word'] = [k]
                            elif simi == domain_max[domain]['max_simi']:
                                domain_max[domain]['max_word'].append(k)
            
            # 获取最大相似度
            max_simi = -1
            for v in domain_max.values():
                if v['max_simi'] > max_simi:
                    max_simi = v['max_simi']
            
            # 获取最终匹配结果
            for k, v in domain_max.items():
                if v['max_simi'] > 0.5 and v['max_simi'] == max_simi:  # 阈值设为0.5
                    domain_pred[k][0] += len(v['max_word'])
                    domain_pred[k][1] += (len(v['max_word']) * max_simi)
                    for w in v['max_word']:
                        domain_matched[k].append([word, w, max_simi])
    else:
        raise ValueError('invalid match mode!', match_mode)
    
    # 确定类别。匹配个数多者优先，相同时相似度大者优先，各领域均为0时，返回“others”
    domain_pred_res = 'others'
    match_num = -1
    simi_sum = -1
    for k, v in domain_pred.items():
        if v[0] == 0:
            continue
        if v[0] > match_num:
            match_num = v[0]
            simi_sum = v[1]
            domain_pred_res = k
        elif v[0] == match_num and v[1] > simi_sum:
            simi_sum = v[1]
            domain_pred_res = k
    
    return (domain_pred_res, simi_sum / match_num, words, domain_matched)

# 测试集结果
def get_test_res(test, embeddings, domain_keywords, intersection, match_mode='full'):
    labels = {}
    confuse_matrix = []   # labels_num行（真实标签），labels_num+1列（预测标签），每个元素为[预测个数，总相似度]
    errors = []  # 二维矩阵，[错误标题，正确类别，预测类别，分词结果，各领域下匹配到的关键词]
    for i, k in enumerate(test.keys()):
        labels[k] = i
    for i in range(len(labels)):
        confuse_matrix.append([])
        for j in range(len(labels) + 1):
            confuse_matrix[-1].append([0, 0])
    labels['others'] = len(test.keys())
    
    # 预测
    for domain, titles in test.items():
        for title in titles:
            pred_res, mean_simi, all_words, domain_matched = classify(title, embeddings, domain_keywords, intersection, match_mode)
            confuse_matrix[labels[domain]][labels[pred_res]][0] += 1
            confuse_matrix[labels[domain]][labels[pred_res]][1] += mean_simi
            if pred_res != domain:
                errors.append([title, domain, pred_res, all_words, domain_matched])
    
    # 计算平均相似度
    for i in range(len(confuse_matrix)):
        for j in range(len(confuse_matrix) + 1):
            if confuse_matrix[i][j][0] != 0:
                confuse_matrix[i][j][1] /= confuse_matrix[i][j][0]
    
    # 计算准确率
    total_num = 0
    for v in test.values():
        total_num += len(v)
    acc_num = 0
    for i in range(len(confuse_matrix)):
        acc_num += confuse_matrix[i][i][0]
    acc = acc_num / total_num if total_num != 0 else 0
    
    return (acc, confuse_matrix, labels, errors)

# 记录log文件
def output_log(log_path, test, train_ratio, acc, confuse_matrix, labels, errors, domain_keywords, intersection, intersection_mode, match_mode, run_time):
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    # 记录基础信息：测试集各领域下文件个数，测试集占比，领域标签个数及内容，各领域关键词个数，交集个数，准确率
    ## 获取领域标签
    domain_labels = []
    for k in domain_keywords.keys():
        domain_labels.append(k)
    domain_labels.append('others')
    ## 获取各领域提取的非交关键词个数
    domain_keywords_num = []
    for k, v in domain_keywords.items():
        domain_keywords_num.append('{}：{}'.format(k, len(v)))
    ## 获取领域间交集关键词个数
    intersection_num = len(intersection)
    ## 获取测试集占比
    test_num = '%.1f' % ((1 - train_ratio) * 100) + '%'
    ## 获取测试集中各领域文件个数
    test_domain_file_num = []
    for k in domain_keywords.keys():
        test_domain_file_num.append('{}：{}'.format(k, len(test[k])))
    ## 写入文件
    with open(os.path.join(log_path, 'base.log'), 'w', encoding='utf-8') as f:
        f.write('领域标签：' + '、'.join(domain_labels) + '\n')
        f.write('\n')
        f.write('求交方式(intersection_mode)：' + intersection_mode + '\n')
        f.write('匹配方式(match_mode)：' + match_mode + '\n')
        f.write('\n')
        f.write('各领域提取的非交关键词个数：\n')
        for t in domain_keywords_num:
            f.write(t + '\n')
        f.write('\n')
        f.write('领域间交集关键词个数：' + str(intersection_num) + '\n')
        f.write('\n')
        f.write('测试集占比：' + test_num + '\n')
        f.write('\n')
        f.write('测试集中各领域文件个数：\n')
        for t in test_domain_file_num:
            f.write(t + '\n')
        f.write('\n')
        f.write('测试集预测准确率：' + str(acc) + '\n')
        f.write('\n')
        f.write('测试集运行时长：{:.2f}s'.format(run_time))

    
    # 记录领域关键词
    domain_keywords_only_words = {}
    for k, v in domain_keywords.items():
        domain_keywords_only_words[k] = list(v.keys())
    
    with open(os.path.join(log_path, 'domain_keywords.log'), 'w', encoding='utf-8') as f:
        f.write(str(domain_keywords_only_words))
    
    # 记录交集
    with open(os.path.join(log_path, 'intersection.log'), 'w', encoding='utf-8') as f:
        f.write(str(intersection))
    
    # 记录混淆矩阵
    labels = [i[0] for i in sorted(labels.items(), key=lambda x: x[1])]
    with open(os.path.join(log_path, 'confuse_matrix.log'), 'w', encoding='utf-8') as f:
        f.write('预测数量\t' + '\t'.join(labels) + '\n')
        for i in range(len(labels) - 1):
            f.write(labels[i] + '\t' + '\t'.join([str(j[0]) for j in confuse_matrix[i]]) + '\n')
        f.write('\n')
        f.write('平均相似度\t' + '\t'.join(labels) + '\n')
        for i in range(len(labels) - 1):
            f.write(labels[i] + '\t' + '\t'.join([str(j[1]) for j in confuse_matrix[i]]) + '\n')
    
    # 记录错误分类结果
    with open(os.path.join(log_path, 'errors_pred.log'), 'w', encoding='utf-8') as f:
        f.write('错误分类标题\t正确类别\t预测类别\n')
        for e in errors:
            f.write('\t'.join(e[:3]) + '\n')
            f.write('all words: ' + str(e[3]) + '\n')
            f.write('matched res: ' + str(e[4]) + '\n')
            f.write('\n')


if __name__ == "__main__":
    embedding_path = r'/home/aistudio/100000-small-modi.txt'  # 词嵌入文件目录，对应100000-small-modi.txt文件
    title_paths = {'ai': r'/home/aistudio/work/ai',
             'education': r'/home/aistudio/work/education',
             'industry': r'/home/aistudio/work/industry'}  # 领域文件夹目录，选择文件名比较长的，如.txt文件或.log文件对应的文件夹
    log_path = r'/home/aistudio/work/'  # 实验结果的log文件夹路径
    train_ratio = 0.5

    embeddings = get_embedding_table(embedding_path)
    titles = get_all_title(title_paths)
    train, test = get_dataset(titles, train_ratio)

    # 实验1
    intersection_mode = 'small'
    match_mode = 'full'
    full_log_path = os.path.join(log_path, 'experiment1-log-{}-{}'.format(train_ratio, ''.join(['%02d' % i for i in time.localtime()[3:6]])))
    domain_keywords, intersection = get_domain_keywords(train, embeddings, intersection_mode)
    tic = time.time()
    acc, confuse_matrix, labels, errors = get_test_res(test, embeddings, domain_keywords, intersection, match_mode)
    run_time = time.time() - tic
    output_log(full_log_path, test, train_ratio, acc, confuse_matrix, labels, errors, domain_keywords, intersection, intersection_mode, match_mode, run_time)

    # 实验2
    intersection_mode = 'mid'
    match_mode = 'full'
    full_log_path = os.path.join(log_path, 'experiment2-log-{}-{}'.format(train_ratio, ''.join(['%02d' % i for i in time.localtime()[3:6]])))
    domain_keywords, intersection = get_domain_keywords(train, embeddings, intersection_mode)
    tic = time.time()
    acc, confuse_matrix, labels, errors = get_test_res(test, embeddings, domain_keywords, intersection, match_mode)
    run_time = time.time() - tic
    output_log(full_log_path, test, train_ratio, acc, confuse_matrix, labels, errors, domain_keywords, intersection, intersection_mode, match_mode, run_time)

    # 实验3
    intersection_mode = 'full'
    match_mode = 'greedy'
    full_log_path = os.path.join(log_path, 'experiment3-log-{}-{}'.format(train_ratio, ''.join(['%02d' % i for i in time.localtime()[3:6]])))
    domain_keywords, intersection = get_domain_keywords(train, embeddings, intersection_mode)
    tic = time.time()
    acc, confuse_matrix, labels, errors = get_test_res(test, embeddings, domain_keywords, intersection, match_mode)
    run_time = time.time() - tic
    output_log(full_log_path, test, train_ratio, acc, confuse_matrix, labels, errors, domain_keywords, intersection, intersection_mode, match_mode, run_time)