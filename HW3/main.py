import json
import random
from random import randint

import gensim
import jieba
from gensim import corpora
import numpy as np
from sklearn.svm import SVC

from HW3.preprocessor import get_texts, stopwords_list

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


if __name__ == '__main__':
    ratio = 0.9
    train_data, train_label, test_data, test_label = [], [], [], []
    if GEN_DATA:
        corpus_context_dict, id_corpus_dict = get_texts()
        dataset = get_dataset()
        # print(dataset)
        for i in range(int(len(dataset) * ratio)):
            train_data.append(dataset[i][1])
            train_label.append(dataset[i][0])
        for i in range(int(len(dataset) * ratio), len(dataset)):
            test_data.append(dataset[i][1])
            test_label.append(dataset[i][0])
        # print(train_data)

        with open('dataset.json', 'w', encoding='utf-8') as f:
            json.dump(dataset, fp=f, indent=4)
    else:
        dataset = []
        with open('dataset.json', 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        # random.shuffle(dataset)
        print(len(dataset))
        print(int(len(dataset) * ratio))
        for i in range(0, int(len(dataset) * ratio)):
            train_data.append(dataset[i][1])
            train_label.append(dataset[i][0])
        for i in range(int(len(dataset) * ratio), len(dataset)):
            test_data.append(dataset[i][1])
            test_label.append(dataset[i][0])
    # print(train_data)

    print("Trainng Dictionary model...")
    dictionary = corpora.Dictionary(train_data)
    lda_corpus_train = [dictionary.doc2bow(tmp_doc) for tmp_doc in train_data]
    # print(dictionary)
    # print(lda_corpus_train)
    print("Trainng LDA model...")
    lda = gensim.models.LdaModel(corpus=lda_corpus_train, id2word=dictionary, num_topics=topic_num)
    topics = lda.print_topics(16)
    print(topics)



    print(len(test_data))
    lda_corpus_test = [dictionary.doc2bow(tmp_doc) for tmp_doc in test_data]
    topics = lda.get_document_topics(lda_corpus_test)
    for i in range(len(test_data)):
        print(topics[i])

    train_topic_results = lda.get_document_topics(lda_corpus_train)
    train_features = np.zeros((len(train_data), topic_num))
    for i in range(len(lda_corpus_train)):
        for topic_no, freq in train_topic_results[i]:
            train_features[i][topic_no] = freq


    print("训练SVM分类器")
    train_label = np.array(train_label)
    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(train_features, train_label)
    print("训练集的精确度为： {:.4f}.".format(sum(classifier.predict(train_features) == train_label) / len(train_label)))


    lda_corpus_test = [dictionary.doc2bow(tmp_doc) for tmp_doc in test_data]
    test_topic_results = lda.get_document_topics(lda_corpus_test)
    test_features = np.zeros((len(test_data), topic_num))
    for i in range(len(test_topic_results)):
        for topic_no, freq in test_topic_results[i]:
            test_features[i][topic_no] = freq
    test_label = np.array(test_label)
    print("测试集的精确度为 {:.4f}.".format(sum(classifier.predict(test_features) == test_label) / len(test_label)))



