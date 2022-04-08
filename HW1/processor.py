import jieba
import chardet
import math
import os

from HW1 import file_reader


def get_tf(words):
    tf_dic = {}
    for w in words:
        tf_dic[w] = tf_dic.get(w, 0) + 1
    return tf_dic


def get_bigram_tf(words):
    tf_dic = {}
    for i in range(len(words) - 1):
        tf_dic[(words[i], words[i + 1])] = tf_dic.get((words[i], words[i + 1]), 0) + 1
    return tf_dic


def get_trigram_tf(words):
    tf_dic = {}
    for i in range(len(words) - 2):
        tf_dic[((words[i], words[i+1]), words[i+2])] = tf_dic.get(((words[i], words[i+1]), words[i+2]), 0) + 1
    return tf_dic


def calculate_entropy(word, words_len):
    return -word / words_len * math.log(word / words_len, 2)


def get_split_words(file):
    with open(file, mode='rb') as f:
        print(file)
        text = f.read()
        real_encode = chardet.detect(text)['encoding']
        text = text.decode(real_encode, errors='ignore')
        # split_words = [x for x in jieba.cut(text)]
        split_words = [x for x in text]
        return split_words


def deal_unigram_entropy(file):
    split_words = get_split_words(file)
    entropy_list = [calculate_entropy(word[1], len(split_words)) for word in get_tf(split_words).items()]
    return sum(entropy_list)


def deal_bigram_entropy(file):
    split_words = get_split_words(file)
    bi_tf = get_bigram_tf(split_words).items()
    word_tf = get_tf(split_words)
    tf_length = sum([dic[1] for dic in bi_tf])
    entropy_list = []
    for tf in bi_tf:  # sentence yx...
        pxy = tf[1] / tf_length  # p(x,y)
        pxcy = tf[1] / word_tf[tf[0][0]]  # p(x|y)
        entropy_list.append(-pxy * math.log(pxcy, 2))
    return sum(entropy_list)


def deal_trigram_entropy(file):
    split_words = get_split_words(file)
    bi_tf = get_bigram_tf(split_words)
    tri_tf = get_trigram_tf(split_words).items()
    tf_length = sum([dic[1] for dic in tri_tf])
    entropy_list = []
    for tf in tri_tf:  # sentence zyx...
        pxy = tf[1] / tf_length  # p(x,y,z)
        pxcy = tf[1] / bi_tf[tf[0][0]]  # p(x|y,z)
        entropy_list.append(-pxy * math.log(pxcy, 2))
    return sum(entropy_list)


if __name__ == '__main__':
    d = {}
    for file in file_reader.get_files():
        simple_name = str(file).split(os.sep)[1].split('.')[0]
        print(simple_name)
        # d[simple_name] = deal_unigram_entropy(file)
        # d[simple_name] = deal_bigram_entropy(file)
        d[simple_name] = deal_trigram_entropy(file)
    print(d)
