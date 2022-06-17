# -*- coding: utf-8 -*-
import argparse
import torch
from torch import nn
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
import os
import time
import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"

train_novel_path = './decode/天龙八部demo'
test_novel_path = './decode/天龙八部test'
is_Train = False
char_key_dict_path = 'chinese_characters_3500.txt'
model_save_path = "./Model/novel_creat_model.pkl"
model_save_path_pth = "./Model/novel_creat.pth"
save_pred_novel_path = "./create_novel/pred_novel_" + str(int(round(time.time() * 1000000))) + ".txt"
pred_novel_start_text = '包三先生笑道：“司马卫的儿子徒弟，都是这么一批脓包货色，除了暗箭伤人，什么都不会。”\n'

use_gpu = torch.cuda.is_available()
print('torch.cuda.is_available() == ', use_gpu)
device = torch.device('cpu')


def dictGet(dict1, index):
    length1 = len(dict1)

    if index >= 0 and index < length1:
        return dict1[index]
    else:
        return dict1[0]


def dictGetValue(dict1, indexZifu):
    if indexZifu in dict1:
        return dict1[indexZifu]
    else:
        return dict1['*']


def getNotSet(list1):
    '''
    返回一个新列表,如何删除列表中重复的元素且保留原顺序
    例子
        list1 = 1 1 2 3 3 5
        return 1 2 3 5
    '''
    l3 = []
    for i in list1:
        if i not in l3:
            l3.append(i)
    return l3


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, ):
        self.args = args
        self.words = self.load_words()

        self.uniq_words = self.get_uniq_words()
        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}
        # self.words_list = list( self.words )

        # 把小说的 字 转换成 int
        self.words_indexes = []

        # 把字典里没有的字符 用'*'表示，也就是Chinese_characters_3500.txt没有的字符
        for w in self.words:
            if not (w in self.word_to_index):
                self.words_indexes.append(1482)  # 1482 =='*'
                # print(w,'= *',)
            else:
                self.words_indexes.append(self.word_to_index[w])
                # print(w,'= ',self.word_to_index[w])

    def load_words(self):
        """加载数据集"""
        if is_Train:
            with open(train_novel_path, encoding='UTF-8') as f:
                corpus_chars = f.read()
        else :
            with open(test_novel_path, encoding='UTF-8') as f:
                corpus_chars = f.read()
        print('length', len(corpus_chars))
        # corpus_chars = corpus_chars[1000:5000]
        return corpus_chars

    def get_uniq_words(self):
        with open(char_key_dict_path, 'r', encoding='utf-8') as f:
            text = f.read()
        idx_to_char = list(text)  # 不能使用 set(self.words) 函数 ,因为每次启动随机,只能用固定的
        return idx_to_char

    def __len__(self):
        return len(self.words_indexes) - self.args.sequence_length

    def __getitem__(self, index):
        return (
            # torch.tensor(self.words_indexes[index:index + self.args.sequence_length]).cuda(),
            torch.tensor(self.words_indexes[index:index + self.args.sequence_length]).cpu(),
            # torch.tensor(self.words_indexes[index + 1:index + self.args.sequence_length + 1]).cuda(),
            torch.tensor(self.words_indexes[index + 1:index + self.args.sequence_length + 1]).cpu(),
        )


class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.input_size = 128
        self.hidden_size = 256
        self.embedding_dim = self.input_size
        self.num_layers = 2

        n_vocab = len(dataset.uniq_words)
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.rnn = nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )
        self.rnn.cpu()
        self.fc = nn.Linear(self.hidden_size, n_vocab).cpu()

    def forward(self, x, prev_state):
        # embed = self.embedding(x).cuda()
        embed = self.embedding(x).cpu()

        output, state = self.rnn(embed, prev_state)
        logits = self.fc(output)

        return logits, state

    def init_state(self, sequence_length):
        return torch.zeros(self.num_layers, sequence_length, self.hidden_size).cpu()


def train(dataset, model, args):
    model.to(device)
    model.train()

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(args.max_epochs):
        state = model.init_state(args.sequence_length)

        for batch, (x, y) in enumerate(dataloader):
            # print(x,y)
            optimizer.zero_grad()
            # x = x.cuda()
            x = x.cpu()
            # y = y.cuda()
            y = y.cpu()
            y_pred, state = model(x, state)

            loss = criterion(y_pred.transpose(1, 2), y)
            loss = loss.to(device)
            state = state.detach()

            loss.backward()
            optimizer.step()

            if batch % 1000 == 0:
                torch.save(model, model_save_path)
                torch.save(model.state_dict(), model_save_path_pth)

            print({'epoch': epoch, 'batch': batch, 'loss': loss.item()})


def predict(dataset, model, text, next_words=20):
    # words = text.split(' ')
    words = list(text)
    model.eval()

    device = 'cpu'
    model.to(device)
    state = model.init_state(len(words))

    for i in range(0, next_words):
        # x = torch.tensor([[dictGetValue(dataset.word_to_index, w) for w in words[i:]]]).cuda()
        x = torch.tensor([[dictGetValue(dataset.word_to_index, w) for w in words[i:]]]).cpu()
        y_pred, state = model(x, state)

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        # p = torch.from_numpy(p).cuda(0)
        # p = torch.from_numpy(p).cpu()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dictGet(dataset.index_to_word, word_index))

    return "".join(words)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rnn')
    parser.add_argument('--max-epochs', type=int, default=5)  # 训练多少遍 总的文本  , default=20)
    parser.add_argument('--batch-size', type=int, default=256)  # default=256)
    parser.add_argument('--sequence-length', type=int, default=50)  # sequence-length 每次训练多长的句子, default=20)
    args = parser.parse_args([])
    dataset = Dataset(args)
    if os.path.exists(model_save_path):
        model = torch.load(model_save_path)
        print('发现有保存的Model,load model ....\n------开始训练----------')
    else:
        print('没保存的Model,Creat model .... \n------开始训练----------')
        model = Model(dataset)

    if is_Train:
        print(model)
        train(dataset, model, args)
        torch.save(model, model_save_path)
        torch.save(model.state_dict(), model_save_path_pth)
        print("训练完成")
    else:
        neirong = predict(dataset, model, pred_novel_start_text, 500)
        print(neirong)
        # for i in range(1, 30):
        #     neirong = predict(dataset, model, pred_novel_start_text, 3000)
        #     with open(save_pred_novel_path, 'a+', buffering=1073741824, encoding='utf-8') as wf:
        #         wf.write(neirong)