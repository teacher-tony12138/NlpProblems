#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author : "Zhitao Wang"
# @file : main.py
# @date : 2020/3/18 11:43 上午
# @contact : 18600064502@163.com

# 以下是每一个单词出现的概率。为了问题的简化，我们只列出了一小部分单词的概率。 在这里没有出现的的单词但是出现在词典里的，统一把概率设置成为0.00001
# 比如 p("学院")=p("概率")=...0.00001
word_prob = {"北京": 0.03, "的": 0.08, "天": 0.005, "气": 0.005, "天气": 0.06, "真": 0.04, "好": 0.05, "真好": 0.04, "啊": 0.01,
             "真好啊": 0.02,
             "今": 0.01, "今天": 0.07, "课程": 0.06, "内容": 0.06, "有": 0.05, "很": 0.03, "很有": 0.04, "意思": 0.06, "有意思": 0.005,
             "课": 0.01,
             "程": 0.005, "经常": 0.08, "意见": 0.08, "意": 0.01, "见": 0.005, "有意见": 0.02, "分歧": 0.04, "分": 0.02, "歧": 0.005}
# 单词表文件路径
vacab_file_path = "chinese_word_base.csv"


def read_vacab(file_path):
    '''
    读取单词表文件并返回
    :param file_path: str 单词表文件路径
    :return: dict 单词表字典
    '''
    dic_words = dict()
    with open(file_path) as f:
        for line in f.readlines():
            word = line.strip().split(",")[0]
            if word not in dic_words:
                dic_words[word] = 1
            else:
                dic_words[word] += 1
    return dic_words


def check_existence(words, dic_words):
    '''
    检查单词是否在词典中
    :param words: str 待检验单词
    :param dic_words: dict 单词表
    :return: bool 是否存在词典
    '''
    flag = True
    for w in words:
        if w not in dic_words:
            flag = False
            break
    return flag


def generate_all_segment(input_str, word_dict):
    '''
    生成所有可能的划分，并检查所有单词是否都在词典中
    :param input_str: str 需要分词的句子
    :param word_dict: dict 词典
    :return: list 所有合法的划分
    '''
    str_len = len(input_str)
    if str_len == 0:
        return
    max_count = 2 ** (str_len - 1)
    all_seg = []
    for i in range(1, max_count):
        ind = 0
        cur_seg = []
        cur_list = []
        ci = i
        while ci > 0:
            cur_list.append(input_str[ind])
            cmod = ci % 2
            if cmod == 1:
                cur_seg.append("".join(cur_list))
                cur_list = []
            ind += 1
            ci = int(ci / 2)
        if ind < str_len:
            cur_seg.append(input_str[ind:])
        if check_existence(cur_seg, word_dict):
            all_seg.append(cur_seg)
    return all_seg


import math


def cal_unigram_prob(prob_dict, cur_str):
    '''
    计算当前分词的一个划分方法的总概率(unigram)
    :param prob_dict: dict unigram概率
    :param cur_str: list 当前划分
    :return: float 当前划分总概率的映射(进过大小和对数变换)
    '''
    res = 0.0
    for s in cur_str:
        cur_prob = 0.0000001
        if s in prob_dict:
            cur_prob = prob_dict[s]
        # 从找最大变成找最小
        res += (math.log(cur_prob) * -1)
    return res


def word_segment_naive(input_str, word_dict):
    '''
    1. 对于输入字符串做分词，并返回所有可行的分词之后的结果。
    2. 针对于每一个返回结果，计算句子的概率
    3. 返回概率最高的最作为最后结果

    input_str: 输入字符串   输入格式：“今天天气好”
    best_segment: 最好的分词结果  输出格式：["今天"，"天气"，"好"]

    :param input_str: str 待分词的句子
    :param word_dict: dict 词典
    :return: list 最优划分
    '''
    # 第一步： 计算所有可能的分词结果，要保证每个分完的词存在于词典里，这个结果有可能会非常多。
    segments = generate_all_segment(input_str, word_dict)  # 存储所有分词的结果。如果次字符串不可能被完全切分，则返回空列表(list)
    # 格式为：segments = [["今天"，“天气”，“好”],["今天"，“天“，”气”，“好”],["今“，”天"，“天气”，“好”],...]

    # 第二步：循环所有的分词结果，并计算出概率最高的分词结果，并返回
    best_segment = ""
    best_score = float("inf")
    for seg in segments:
        cur_score = cal_unigram_prob(word_prob, seg)
        if cur_score < best_score:
            best_score = cur_score
            best_segment = seg

    return best_segment


def generate_dag(matrix, input_str, word_prob, dic_words):
    '''
    根据输入生成有向图(邻接矩阵)
    :param matrix: list 有向图容器
    :param input_str: str 待分词的句子
    :param word_prob: dict unigram概率
    :param dic_words: dict 词典
    :return: 有向图
    '''
    str_len = len(input_str)
    max_len_dic = max(list(map(lambda x: len(x), list(dic_words.keys()))))
    scan_len = max_len_dic if max_len_dic < str_len else str_len
    for i in range(1, scan_len + 1):
        for j in range(str_len - i + 1):
            cur_words = "".join(input_str[j: j + i])
            if cur_words in dic_words:
                matrix[j][j + i] = (-1 * math.log(word_prob[cur_words])) if cur_words in word_prob else (
                            -1 * math.log(0.0000001))

    return matrix


def dp_on_dag(input_str, word_prob, dic_words):
    '''
    在构建的有向图上使用维特比算法(dp)求解最优解
    :param input_str: str 待分词的句子
    :param word_prob: dict unigram概率
    :param dic_words: dict 词典
    :return: list 最优划分点的序号列表
    '''
    dp_len = len(input_str) + 1
    matrix = [[0 for _ in range(dp_len)] for _ in range(dp_len)]
    matrix = generate_dag(matrix, input_str, word_prob, dic_words)
    dp_list = [0] * dp_len
    parent = [0] * dp_len
    for i in range(1, dp_len):
        min_edge = float("inf")
        min_ind = -1
        for j in range(0, i):
            if matrix[j][i] > 0:
                if matrix[j][i] + dp_list[j] < min_edge:
                    min_edge = matrix[j][i] + dp_list[j]
                    min_ind = j
        dp_list[i] = min_edge
        parent[i] = min_ind
    return parent


def get_seg(input_str, word_prob, dic_words):
    '''
    使用维特比算法的到最优划分点列表，还原最优划分
    :param input_str: str 待分词的句子
    :param word_prob: dict unigram概率
    :param dic_words: dict 词典
    :return: list 最优划分
    '''
    parents = dp_on_dag(input_str, word_prob, dic_words)
    dp_len = len(parents) - 1
    segment = []
    res = []
    ind = dp_len
    while ind != 0:
        res.append(ind)
        ind = parents[ind]
    res.append(0)
    res.reverse()
    for i in range(len(res) - 1):
        l = res[i]
        r = res[i + 1]
        segment.append("".join(input_str[l: r]))
    return segment


def word_segment_viterbi(input_str, dic_words):
    '''
    1. 基于输入字符串，词典，以及给定的unigram概率来创建DAG(有向图）。
    2. 编写维特比算法来寻找最优的PATH
    3. 返回分词结果

    input_str: 输入字符串   输入格式：“今天天气好”
    best_segment: 最好的分词结果  输出格式：["今天"，"天气"，"好"]

    :param input_str: str 待分词的句子
    :param dic_words: dict 词典
    :return: list 最优划分
    '''
    # 第一步：根据词典，输入的句子，以及给定的unigram概率来创建带权重的有向图（Directed Graph） 参考：课程内容
    # 有向图的每一条边是一个单词的概率（只要存在于词典里的都可以作为一个合法的单词），这些概率在 word_prob，如果不在word_prob里的单词但在
    # 词典里存在的，统一用概率值0.00001。
    best_segment = get_seg(input_str, word_prob, dic_words)

    # 利用维特比算法来找出最好的PATH， 这个PATH是P(sentence)最大或者 -log P(sentence)最小的PATH。

    # 根据最好的PATH, 返回最好的切分

    return best_segment


if __name__ == '__main__':
    dic_words = read_vacab(vacab_file_path)
    # 测试，分别使用枚举划分和维特比算法，注意这里unigram概率都已经给出，如果需要测试其他语句，需要自行生成unigram概率
    print(word_segment_naive("北京的天气真好啊", dic_words), word_segment_viterbi("北京的天气真好啊", dic_words))
    print(word_segment_naive("今天的课程内容很有意思", dic_words), word_segment_viterbi("今天的课程内容很有意思", dic_words))
    print(word_segment_naive("经常有意见分歧", dic_words), word_segment_viterbi("经常有意见分歧", dic_words))
