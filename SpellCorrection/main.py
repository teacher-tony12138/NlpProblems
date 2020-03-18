#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author : "Zhitao Wang"
# @file : main.py
# @date : 2020/3/17 2:57 下午
# @contact : 18600064502@163.com

# 英文拼写错误！！！
# 数据下载地址见项目根目录下的README.md文件
# 单词表数据路径
vacab_file_path = "vocab.txt"
# 拼写错误数据路径，表示常常被拼写错误的单词和对应的错误拼写(如apple：appl, aple)
spell_error_prob_file_path = "spell-errors.txt"
# 测试数据路径
test_data_file_path = "testdata.txt"


def read_vacab(file_path):
    '''
    读入词汇表
    :param file_path: str, 词汇表文件路径
    :return: dict 词汇表
    '''
    vab_dict = dict()
    with open(file_path) as f:
        for line in f.readlines():
            vab_dict[line.rstrip()] = 1
    return vab_dict


def read_spell_error_prob(file_path):
    '''
    读入拼接错误概率(简化为等概率，实际上根据日志统计, 比如搜索引擎常常提示你，你要搜的是不是：xxxx)
    :param file_path: str, 拼写错误文件
    :return: dict<dict> 拼写错误概率(apple -> {appl: 0.5, aple: 0.5})
    '''
    spell_error_prob = dict()
    with open(file_path) as f:
        for line in f.read().split("\n"):
            hits = line.rstrip().split(": ")
            correct = hits[0]
            spell_error_prob[correct] = dict()
            errors = hits[1].split(", ")
            for error in errors:
                spell_error_prob[correct][error] = 1.0 / len(errors)
    return spell_error_prob


def read_test_data(file_path):
    '''
    读取测试数据
    :param file_path: str
    :return: dict, 测试数据
    '''
    test_data = dict()
    with open(file_path) as f:
        for line in f.readlines():
            hits = line.rstrip().split("\t")
            test_data[float(hits[0])] = [float(hits[1]), hits[2]]
    return test_data


def generate_bigram_LM(corpus_list):
    '''
    构建bigram语言模型
    :param corpus_list: list, 语料库(英文，每个句子是列表形式)
    :return: one_term_count dict, unigram 词频
             two_term_count dict, bigram 词频
    '''
    one_term_count = dict()
    two_term_count = dict()

    for doc in corpus_list:
        doc = ['<s>'] + doc + ['<e>']
        for i in range(len(doc) - 1):

            one_term = doc[i]
            two_term = ' '.join(doc[i: i + 2])

            if one_term in one_term_count:
                one_term_count[one_term] += 1
            else:
                one_term_count[one_term] = 1

            if two_term in two_term_count:
                two_term_count[two_term] += 1
            else:
                two_term_count[two_term] = 1

    return one_term_count, two_term_count


def generate_edit_distance_word_equal_one(word, vacab_list, check_valid):
    '''
    生成编辑距离为1的有效单词
    :param word: str 原单词
    :param vacab_list: list, 单词库
    :param check_valid: bool 是否检查单词合法性
    :return: list 所有编辑距离为1的单词
    '''
    res = []
    # 分割原单词
    word_s = [(word[: i], word[i:]) for i in range(len(word) + 1)]
    letters = 'abcdefghijklmnopqrstuvwxyz'

    # 生成三种操作下的所有单词

    insert = [L + l + R for L, R in word_s for l in letters]

    delete = [L + R[1:] for L, R in word_s if R]

    replace = [L + l + R[1:] for L, R in word_s if R for l in letters]

    # 合并去重
    if check_valid:
        return list(set(cur for cur in replace + insert + delete if cur in vacab_list and cur != word))
    else:
        return list(set(cur for cur in replace + insert + delete if cur != word))


def generate_valid_edit_distance_word(word, dist, vacab_list):
    '''
    生成制定编辑距离的单词
    在求编辑距离为2时，中间过程生成的单词可能不合法，如app -> appl -> apple
    :param word: str 原单词
    :param dist: int 编辑距离
    :param vacab_list: list 单词表
    :return: list 所有编辑距离为dist的单词
    '''
    cur_dist = 1
    cur_list = generate_edit_distance_word_equal_one(word, vacab_list, False)
    res = []
    while cur_dist < dist:
        for cur_w in cur_list:
            res += generate_edit_distance_word_equal_one(cur_w, vacab_list, False)
        cur_list = res
        cur_dist += 1
    # 校验合法性(是否在词库中)
    return list(set(w for w in cur_list if w in vacab_list and w != word))


def word_format(w):
    '''
    定义规则对单词特殊处理如Tom's, wouldn't，需要还原单词，但这里和stemming有区别
    :param w: str 单词
    :return: str 处理后的单词
    '''
    w = w if w[-1] not in [',', '.', '!'] else w[:-1]
    w = w if ''.join(w[-2:]) not in ["'s"] else w[:-2]
    w = w if ''.join(w[-2:]) not in ["'t"] else w[:-3]
    return w


if __name__ == '__main__':
    from nltk.corpus import reuters
    import numpy as np

    # 读取nltk语料库
    categories = reuters.categories()
    corpus = reuters.sents(categories=categories)

    # 准备所需数据
    test_data = read_test_data(test_data_file_path)
    vacab_list = read_vacab(vacab_file_path)
    spell_prob = read_spell_error_prob(spell_error_prob_file_path)
    one_term_dict, two_term_dict = generate_bigram_LM(corpus)

    # 对测试数据进行拼写纠错

    # 定义最小默认概率
    noisy_channel_default_prob = np.log(0.00001)
    # unigarm 词表大小，注意这里不是通用的vacab_list
    V = len(list(one_term_dict.keys()))

    for sen_idx, line in test_data.items():
        count = line[0]
        # 因为是bigram，需要对句子前后进行填充
        sentense = '<s>' + line[1] + ' ' + '<e>'
        sp = list(filter(lambda x: len(x) > 0, sentense.split(' ')))
        error_count = 0
        for i in range(1, len(sp) - 1):
            # 单词不在词库里，视为拼写错误
            word = sp[i]
            word = word_format(word)
            if word not in vacab_list:

                # 所有概率集合
                prob_list = []

                init_edit_dist = 1
                # 生成所有编辑距离不大于2的候选单词(处于性能和统计考虑，由于大于1的编辑距离的的中间过程的单词可能不合法，所以编辑距离大于3时性能有明显下降，而且大多数拼写错误不会大于两个字母)
                candidates = generate_valid_edit_distance_word(word, init_edit_dist, vacab_list)
                while len(candidates) == 0:
                    init_edit_dist += 1
                    candidates = generate_valid_edit_distance_word(word, init_edit_dist, vacab_list)
                    if init_edit_dist >= 2:
                        break
                # 忽略编辑距离为2时没有候选单词的情况
                if len(candidates) == 0:
                    continue
                # 对于每个候选单词计算条件概率
                for cand in candidates:
                    # 前后词
                    pre_bi = sp[i - 1] + ' ' + cand
                    next_bi = cand + ' ' + sp[i + 1]

                    # nosiy channel model probability
                    cur_noisy_ch_prob = np.log(spell_prob[cand][word]) if cand in spell_prob and word in spell_prob[
                        cand] else noisy_channel_default_prob

                    # 语言模型概率
                    # 前向bigram概率
                    pre_bigram_prob = 0.0
                    if cand in one_term_dict and pre_bi in two_term_dict:
                        pre_bigram_prob = np.log((two_term_dict[pre_bi] + 1) / (one_term_dict[cand] + V))
                    else:
                        pre_bigram_prob = np.log(1.0 / V)

                    # 后向bigram概率
                    next_bigram_prob = 0.0
                    if cand in one_term_dict and next_bi in two_term_dict:
                        next_bigram_prob = np.log((two_term_dict[next_bi] + 1) / (one_term_dict[cand] + V))
                    else:
                        next_bigram_prob = np.log(1.0 / V)

                    sum_prob = cur_noisy_ch_prob + pre_bigram_prob + cur_noisy_ch_prob + next_bigram_prob
                    prob_list.append(sum_prob)

                max_idx = prob_list.index(max(prob_list))
                print("{}: {} -> {}".format(sen_idx, word, candidates[max_idx]))
        print("*" * 50)
