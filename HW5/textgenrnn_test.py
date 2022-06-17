from textgenrnn import textgenrnn

if __name__ == '__main__':
    textgen = textgenrnn(name="novel")   # 给模型起个名字
    textgen.reset()
    # 从数据文件训练模型
    textgen.train_from_file(file_path='./decode/天龙八部demo', # 文件路径
                            new_model=True, # 训练新模型
                            batch_size=5,
                            rnn_bidirectional=True, # 是否使用Bi-LSTM
                            rnn_size=128,
                            word_level=False, # True:词级别，False:字级别
                            dim_embeddings=256,
                            num_epochs=3, # 训练轮数
                            max_length=30, # 一条数据的最大长度
                            verbose=1,
                            multi_gpu = False)

    textgen.save()

    textgen_2 = textgenrnn(weights_path='novel_weights.hdf5',
                            vocab_path='novel_vocab.json',
                            config_path='novel_config.json')

    textgen_2.generate_samples()