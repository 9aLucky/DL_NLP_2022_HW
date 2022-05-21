import json
import os
import random
from collections import Counter
from random import randint

import gensim
import jieba
from gensim import corpora
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence, PathLineSentences
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.svm import SVC
import matplotlib.pyplot as plt

from HW4.preprocessor import get_texts, stopwords_list

para_num = 250
per_para_words = 600
corpus_context_dict = {}
id_corpus_dict = {}
topic_num = 100
GEN_DATA = False


def is_chinese(uchar):
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    else:
        return False


def get_dataset():
    data = []
    for i in range(para_num):
        context = corpus_context_dict[id_corpus_dict[i % 16]]
        rand_value = randint(0, len(context) - per_para_words)
        new_context = context[rand_value:rand_value + per_para_words]
        cut_list = list(jieba.cut(new_context, cut_all=False))
        filter_list = []
        for c in cut_list:
            if stopwords_list.__contains__(c):
                continue
            filter_list.append(c)
        data.append((i % 16, filter_list))
    return data


def cluster(model):
    all_words = []
    for file in os.listdir('train_data'):
        with open('train_data/' + file, 'r') as f:
            all_words.extend(f.read().split(" "))
    word_times_dict = Counter(all_words)

    highest_words = []
    for k, v in word_times_dict.items():
        if v > 80:
            highest_words.append(k)

    word_vectors = []
    for tmp_word in highest_words:
        word_vectors.append(model.wv[tmp_word])
    tSNE = TSNE()
    word_embeddings = tSNE.fit_transform(word_vectors)
    classifier = KMeans(n_clusters=16)
    classifier.fit(word_embeddings)
    labels = classifier.labels_

    min_left = min(word_embeddings[:, 0])
    max_right = max(word_embeddings[:, 0])
    min_bottom = min(word_embeddings[:, 1])
    max_top = max(word_embeddings[:, 1])

    markers = ["bo", "go", "ro", "co", "mo", "yo", "ko", "bx", "gx", "rx", "cx", "mx", "yx", "kx", "b>", "g>"]

    for i in range(len(word_embeddings)):
        plt.plot(word_embeddings[i][0], word_embeddings[i][1], markers[labels[i]])
    plt.axis([min_left, max_right, min_bottom, max_top])
    plt.savefig("./kmeans_result.png")



if __name__ == '__main__':
    if GEN_DATA:
        corpus_context_dict, id_corpus_dict = get_texts()
        dataset = get_dataset()
        for name in corpus_context_dict.keys():
            words = jieba.cut(corpus_context_dict[name], cut_all=False)
            with open('train_data/' + name, 'w', encoding='utf-8') as f:
                for w in words:
                    f.write(w)
                    f.write(" ")
    else:
        #corpus_context_dict, id_corpus_dict = get_texts()
        # word2vec_model_cb = Word2Vec(sentences=PathLineSentences('train_data'), hs=1, min_count=10, window=5, vector_size=200, sg=0, workers=16, epochs=10)
        # word2vec_model_sg = Word2Vec(sentences=PathLineSentences('train_data'), hs=1, min_count=10, window=5, vector_size=200, sg=1, workers=16, epochs=10)
        word2vec_model = Word2Vec.load('skip_gram.model')
        cluster(word2vec_model)
        # test_name = ['郭靖', '华山派', '蛤蟆功', '九阴真经']
        # for name in test_name:
        #     print(name)
        #     for result in word2vec_model.wv.similar_by_word(name, topn=10):
        #         print(result[0], '{:.3f}'.format(result[1]))
        # word2vec_model_cb.save('cbow.model')
        # word2vec_model_sg.save('skip_gram.model')


