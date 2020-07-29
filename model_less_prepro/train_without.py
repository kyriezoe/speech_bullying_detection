import numpy as np
import keras
import json
import random

import copy
from keras.layers import Dense, Activation, Embedding, Bidirectional, GRU, Flatten, Dropout
from keras import regularizers

from keras.callbacks import ModelCheckpoint

# 数据处理
f = open('./dataset.json')
TrollsData = []  # label:1
NonTrollsData = []  # label:0
for i in f:
    temp = json.loads(i)
    content = temp['content']
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

# 创建词表
WordFre = {}  # 词频
for d in Data:
    content = keras.preprocessing.text.text_to_word_sequence(d['content'])
    for c in content:
        if c == '':
            continue
        word = c.lower()
        if WordFre.get(word, None) is None:
            WordFre[word] = 0
        WordFre[word] += 1
threshold = 10
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

# 分割训练和测试集
split = 1
trainNum = int(len(TrainData) * split)
finalTrainData = TrainData[0:trainNum]
finalTrainLable = TrainLabel[0:trainNum]
finalTestData = TrainData[trainNum:]
finalTestLabel = TrainLabel[trainNum:]

#
# val_loss: 1.0881 - val_binary_accuracy: 0.6203
# checkpoint
filepath = "weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# Fit the model
model.fit(np.asarray(finalTrainData), np.asarray(finalTrainLable), validation_split=0.1, epochs=500, batch_size=512, callbacks=callbacks_list, verbose=0)
#
#
# 保存模型
model.save('model.h5')

# # 评估模型效果
# accuracy = model.evaluate(np.asarray(finalTestData), np.asarray(finalTestLabel))
# print('accuracy', accuracy)
#
# # 模型预测
# preds = model.predict(np.asarray(finalTestData))
# for i in range(len(preds)):
#     if preds[i] > 0.5:
#         preds[i] = 1
#     else:
#         preds[i] = 0
#
# y_predict = [int(item) for item in preds]
# print('num of test predictions')
# print(len(y_predict))
#
# print('acc')
# acc = accuracy_score(y_predict, np.asarray(finalTestLabel))
# print(acc)
#
# print('precision')
# precision = precision_score(y_predict, np.asarray(finalTestLabel), average='weighted')
# print(precision)
#
# print('recall')
# recall = recall_score(y_predict, np.asarray(finalTestLabel), average='weighted')
# print(recall)
#
# print('matrix')
# matrix = confusion_matrix(y_predict, np.asarray(finalTestLabel))
# print(matrix)
