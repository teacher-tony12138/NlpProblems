#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author : "Zhitao Wang"
# @file : main.py
# @date : 2020/3/17 4:14 下午
# @contact : 18600064502@163.com

import json
from nltk.corpus import stopwords
import re
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
import numpy as np
from scipy import sparse

# 问答对文件路径
qa_file_path = "train-v2.0.json"
# 词干提取
snowball_stemmer = SnowballStemmer("english")
# 停用词
sWords = stopwords.words('english')

def read_corpus(file_path):
    """
    读取给定的语料库，并把问题列表和答案列表分别写入到 qlist, alist 里面。 在此过程中，不用对字符换做任何的处理（这部分需要在 Part 2.3里处理）
    qlist = ["问题1"， “问题2”， “问题3” ....]
    alist = ["答案1", "答案2", "答案3" ....]
    务必要让每一个问题和答案对应起来（下标位置一致）
    :param file_path: str, 问答对文件路径
    :return:
            qlist: list 问题列表
            alist: list 答案列表
    """
    qlist = []
    alist = []
    # 读取文件并将json解析为dict
    with open(file_path) as f:
        json_str = f.read()
    qa_dic = json.loads(json_str)
    # 解析dict，的到q-a对
    for data in qa_dic['data']:
        for para in data['paragraphs']:
            for qas in para['qas']:
                qlist.append(qas['question'])
                if len(qas['answers']) == 0:
                    alist.append(qas['plausible_answers'][0]['text'])
                else:
                    alist.append(qas['answers'][0]['text'])

    assert len(qlist) == len(alist)  # 确保长度一样
    return qlist, alist

def word_stat(qlist, alist):
    '''
    词频统计
    :param qlist: list 问题列表
    :param alist: list 答案列表
    :return: dict 词频字典
    '''
    res_dict = dict()
    length = len(qlist)
    for i in range(length):
        qes = re.sub(r'[^\w\s]','', qlist[i]).split(" ")
        ans = re.sub(r'[^\w\s]','', alist[i]).split(" ")
        for word in qes + ans:
            if word not in res_dict:
                res_dict[word] = 1
            else:
                res_dict[word] += 1

    return res_dict


def word_filter(sentence, qa_word_dict):
    '''
    单词预处理
    :param sentence: str 需要进行预处理的句子，空格分隔
    :param qa_word_dict: 词频词典
    :return: 处理完毕的句子，空格分隔
    '''
    # 处理时临时存储数组
    tmp = []
    # 过滤无用符号
    sen = re.sub(r'[^\w\s]','', sentence).split(" ")
    # 频率在5的以下忽略
    threshold = 5
    # 去掉停用词, 低频词, 转换成小写, 统一数字字符
    for w in sen:
        #去除停用此和低频词
        if w not in sWords and w in qa_word_dict and qa_word_dict[w] >= threshold:
            if w.isdigit():
                #将数字字符统一成'#number'
                tmp.append("#number")
            else:
                #转化小写
                cur = w.lower()
                cur = snowball_stemmer.stem(cur)
                tmp.append(cur)
    sen = " ".join(tmp)
    return sen


def train_tfidf(blist):
    '''
    训练tf-idf模型
    :param blist: list 语料库，每一行是一个空格分隔的句子
    :return:
            tfidf_matrix：tf-idf矩阵
            vectorizer：sklean-build-in sklearn中向量化模型
            transformer：sklean-build-in sklearn中tf-idf模型
    '''
    vectorizer = CountVectorizer()
    count = vectorizer.fit_transform(blist)

    transformer = TfidfTransformer()
    tfidf_matrix = transformer.fit_transform(count)
    return tfidf_matrix, vectorizer, transformer


def get_inverted_ind(org_list):
    '''
    生成倒排表
    :param org_list: list语料库，每一行是一个空格分隔的句子
    :return: dict 倒排表
    '''
    inverted_dict = dict()
    for ind, q in enumerate(org_list):
        for w in q.split(" "):
            if w not in inverted_dict:
                inverted_dict[w] = [ind]
            else:
                inverted_dict[w].append(ind)
    return inverted_dict


# 调节阈值
def threshold_check(candidates, st_th):
    '''
    动态计算阈值，调节倒排表中匹配的数量
    阈值：两个句子有多少单词相同
    :param candidates: dict 备选问题字典
    :param st_th: int 初始阈值
    :return:
            st_th: int 调节后阈值
            num: int 调节后备选集数量
    '''
    # 允许进行距离计算的最大备选集数量
    num_limit = 1000
    values = list(candidates.values())
    num = num_limit + 1
    while num > num_limit:
        st_th += 1
        num = sum(list(map(lambda x: 1 if x >= st_th else 0, values)))
    return st_th, num


def qlist_filter(processed_input, inverted_dict):
    '''
    根据倒排表在问题集中筛选备选问题
    :param processed_input: str 预处理后空格分隔的句子
    :param inverted_dict: dict 倒排表
    :return: list 备选问题列表
    '''
    # 初始筛选阈值(两个句子有多少单词相同)
    threshold = 1
    candidate = dict()
    for w in processed_input.split(" "):
        if w in inverted_dict:
            inds = inverted_dict[w]
            for ind in inds:
                if ind in candidate:
                    candidate[ind] += 1
                else:
                    candidate[ind] = 1
    res = []
    th, num = threshold_check(candidate, threshold)
    if num == 0:
        raise Exception("No enough answer!")
    print("threshlod is {} and candidates amount is {}".format(th, num))
    for ind, count in candidate.items():
        if count >= th:
            res.append(ind)
    return res


def cal_distance(cur_input, filtered_ind, vec, trans, tfidf_matrix):
    '''
    计算当前句子和备选问题集合的余弦相似度，并返回最近的5个问题在备选集合中的索引
    :param cur_input: str 当前输入句子
    :param filtered_ind: list 备选问题索引列表
    :param vec: sklean-build-in 用所有问题集合训练好的向量化模型
    :param trans: sklean-build-in 用所有问题集合训练好的tf-idf模型
    :param tfidf_matrix: list 用所有问题集合训练好的ti-idf数据
    :return: list 最近的5个问题在备选集合中的索引列表
    '''
    print("calculating tf-idf...")
    time_start = time.time()
    new_count = vec.transform([cur_input])
    new_tfidf = trans.transform(new_count).toarray()[0]
    time_end = time.time()
    time_cost_tf_idf = time_end - time_start
    print('time cost', time_cost_tf_idf, 's')

    tfidf_array = tfidf_matrix.toarray()
    filter_tfidf_array = []
    for ind in filtered_ind:
        filter_tfidf_array.append(tfidf_array[ind])
    filter_tfidf_array.append(new_tfidf)

    print("calculating similarity...")
    time_start = time.time()
    sparse_tfidf = sparse.csr_matrix(np.array(filter_tfidf_array))
    cos_s = cosine_similarity(sparse_tfidf)
    time_end = time.time()
    time_cost_similarity = time_end - time_start
    print('time cost', time_cost_similarity, 's')

    print('Totoal time cost', time_cost_tf_idf + time_cost_similarity)

    input_s_list = cos_s[-1][0: -1]
    s_dict = dict()
    for ind, v in enumerate(input_s_list):
        s_dict[ind] = v
    sorted_similar = sorted(s_dict.items(), key=lambda d: d[1], reverse=True)[0: 5]
    return sorted_similar


def top5results_invidx(input_q, inverted_idx, vec, trans, tfidf_matrix, alist, qa_word_dict):
    '''
    组合以上内容，根据输入的句子，返回输入问题的答案
    :param input_q: str 输入的问题
    :param inverted_idx: dict 倒排表
    :param vec: sklean-build-in 用所有问题集合训练好的向量化模型
    :param trans: sklean-build-in 用所有问题集合训练好的tf-idf模型
    :param tfidf_matrix: list 用所有问题集合训练好的ti-idf数据
    :param alist: list 答案集合
    :param qa_word_dict: dict 词频字典
    :return: str 对应答案
    '''
    # 输入预处理
    processed_input = word_filter(input_q, qa_word_dict)

    # 利用到排表过滤备选问题
    candidates = qlist_filter(processed_input, inverted_idx)

    # 计算相似度
    top5 = cal_distance(processed_input, candidates, vec, trans, tfidf_matrix)
    top_idxs = [x[0] for x in top5]

    # 利用倒排表还原问题索引
    org_ind = [candidates[x] for x in top_idxs]

    # 获取问题答案
    cur_alist = [alist[x] for x in org_ind][0]

    return cur_alist


if __name__ == '__main__':
    qes1 = 'How many volts does a track lighting system usually use?'
    qes2 = 'How many volts we need if we want to build a track lighting system?'

    print("加载问答对并配对...")
    qlist, alist = read_corpus(qa_file_path)

    qa_word_dict = word_stat(qlist, alist)
    processed_qlist = []
    print("对问题进行预处理...")
    for qes in qlist:
        processed_qlist.append(word_filter(qes, qa_word_dict))

    print("根据问题集合训练tf-idf...")
    tfidf_matrix, vec, trans = train_tfidf(processed_qlist)

    print("生成倒排表...")
    inverted_idx = get_inverted_ind(processed_qlist)

    print("运行测试...")
    print(top5results_invidx(qes1, inverted_idx, vec, trans, tfidf_matrix, alist, qa_word_dict))
    print('*' * 50)
    print(top5results_invidx(qes1, inverted_idx, vec, trans, tfidf_matrix, alist, qa_word_dict))







