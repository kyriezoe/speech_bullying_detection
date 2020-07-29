import re

import numpy as np
import keras
import json
import random
from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords

import copy
from keras.layers import Dense, Activation, Embedding, Bidirectional, GRU, Flatten, Dropout
from keras import regularizers
from keras.callbacks import ModelCheckpoint


class TrainGRU(object):
    # 数据预处理：分词、标点、小写、stemming、缩写转换
    f = open('../data/train.json')
    TrollsData = []  # label:1
    NonTrollsData = []  # label:0
    for i in f:
        temp = json.loads(i)
        content = temp['content']
        # content = content.lower()
        # wordnet_lemmatizer = WordNetLemmatizer()
        lancaster = LancasterStemmer()
        for token in content:
            # token = wordnet_lemmatizer.lemmatize(token)
            token = lancaster.stem(token)
        content = content = re.sub("'re", " are", content)
        content = re.sub("'s", " is", content)
        content = re.sub("'m", " am", content)
        # content = re.sub("[+.!/,-_?,$%^*(+\"\']+|[+——！...、~@#￥&*()]+", "", content)
        # print(content)
        label = int(temp['annotation']['label'][0])
        Data = {
            "content": content,
            "annotation": label
        }
        if label == 0:
            NonTrollsData.append(Data)
        else:
            TrollsData.append(Data)
    f.close()
    print('数据总数：%d, 欺凌数据个数：%d, 友好数据个数：%d' % (len(TrollsData) + len(NonTrollsData), len(TrollsData), len(NonTrollsData)))
    # print(json.dumps(data))

    # 数据不均衡，重复采样
    new_temp_TrollsData = copy.deepcopy(TrollsData)
    random.shuffle(new_temp_TrollsData)

    for i in range(len(new_temp_TrollsData)):
        content = TrollsData[i]['content'].split(' ')
        content_len = len(content)
        r = random.randint(0, content_len - 1)
        new_temp_TrollsData[i]['content'] += ' ' + content[r]
        r = random.randint(0, content_len - 1)
        new_temp_TrollsData[i]['content'] += ' ' + content[r]

    TrollsData += new_temp_TrollsData
    TrollsData = TrollsData[:11178]
    Data = TrollsData + NonTrollsData
    random.shuffle(Data)
    random.shuffle(Data)
    print('数据总数：%d, 欺凌数据个数：%d, 友好数据个数：%d' % (len(TrollsData) + len(NonTrollsData), len(TrollsData), len(NonTrollsData)))

    # 创建词表：出现5次以上的词
    WordFre = {}  # 词频
    my_stopwords = set(stopwords.words('english'))
    for d in Data:
        # 分词、去标点、小写
        content = keras.preprocessing.text.text_to_word_sequence(d['content'])
        content = [w for w in content if w not in my_stopwords]
        for c in content:
            if c == '':
                continue
            word = c.lower()
            if WordFre.get(word, None) is None:
                WordFre[word] = 0
            WordFre[word] += 1
    threshold = 5
    WordIdx = {}
    indx = 0
    for w in sorted(WordFre.items(), key=lambda x: x[1], reverse=True):
        key = w[0]
        fre = w[1]
        if fre < threshold:
            continue
        indx += 1
        WordIdx[key] = indx
    WordIdx['_PAD_'] = indx + 1,
    WordIdx['_UNK_'] = indx + 2
    WordIdxLen = len(WordIdx)
    json.dump(WordIdx, open('word.json', 'w'))

    # 模型：词表->embedding->GRU->全连接(relu)->dropout->全连接(relu)->全连接(sigmoid)
    model = keras.models.Sequential()
    model.add(Embedding(input_dim=WordIdxLen + 1, output_dim=50, mask_zero=True))
    model.add(Bidirectional(GRU(64, return_sequences=False, dropout=0.4)))
    model.add(Dense(units=32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.2, noise_shape=None, seed=None))
    model.add(Dense(units=16, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['binary_accuracy'],
    )
    model.summary()
    print(model.input_shape)
    model.predict(np.asarray([
        [1, 2, 3, 4, 5, 6, 7],
        [2, 3, 4, 5, 6, 7, 8]
    ])).shape

    # 填充字符串
    max_str_len = 0
    for d in Data:
        d['content'] = keras.preprocessing.text.text_to_word_sequence(d['content'])
        print(d['content'])
        if len(d['content']) > max_str_len:
            max_str_len = len(d['content'])
    TrainData = []
    TrainLabel = []
    for d in Data:
        vec = [WordIdx.get(i, WordIdx['_UNK_']) for i in d['content']]
        pad = WordIdx.get('_PAD_')
        temp = keras.preprocessing.sequence.pad_sequences(sequences=[vec], maxlen=max_str_len, value=pad)
        TrainData.append(temp[0])
        TrainLabel.append(d['annotation'])

    # 分割训练和测试集，训练模型并保存
    split = 1
    trainNum = int(len(TrainData) * split)
    finalTrainData = TrainData[0:trainNum]
    finalTrainLable = TrainLabel[0:trainNum]
    finalTestData = TrainData[trainNum:]
    finalTestLabel = TrainLabel[trainNum:]
    filepath = "weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    model.fit(np.asarray(finalTrainData), np.asarray(finalTrainLable), validation_split=0.1, epochs=500, batch_size=512,
              callbacks=callbacks_list, verbose=0)
