import numpy as np
import keras
import json
import csv

from keras.layers import Dense, Activation, Embedding, Bidirectional, GRU, Flatten, Dropout
from keras import regularizers
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


# 数据处理
TrollsData = []  # label:1
NonTrollsData = []  # label:0
with open('../data/new.csv', 'r', encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        content = row['tweet']
        label = int(row['class'])
        if label == 0 or label == 1:
            label = 1
        else:
            label = 0
        Data = {
            "content": content,
            "annotation": label
        }
        if label == 0:
            NonTrollsData.append(Data)
        else:
            TrollsData.append(Data)
print('数据总数：%d, 欺凌数据个数：%d, 友好数据个数：%d' % (len(TrollsData) + len(NonTrollsData), len(TrollsData), len(NonTrollsData)))
# print(json.dumps(data))

Data = TrollsData + NonTrollsData
WordIdx = json.load(open('word.json'))
WordIdxLen = len(WordIdx)

model = keras.models.Sequential()
model.add(Embedding(input_dim=WordIdxLen + 1, output_dim=50, mask_zero=True))
model.add(Bidirectional(GRU(64, return_sequences=False, dropout=0.4)))
model.add(Dense(units=32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.2, noise_shape=None, seed=None))
model.add(Dense(units=16, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(units=1, activation='sigmoid'))
model.load_weights("weights.best.hdf5")
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

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
finalTestData = TrainData[0:trainNum]
finalTestLabel = TrainLabel[0:trainNum]

# 评估模型效果
loss, accuracy = model.evaluate(np.asarray(finalTestData), np.asarray(finalTestLabel))
print('accuracy', accuracy)
print('loss', loss)

# 模型预测
preds = model.predict(np.asarray(finalTestData))
for i in range(len(preds)):
    if preds[i] > 0.5:
        preds[i] = 1
    else:
        preds[i] = 0

y_predict = [int(item) for item in preds]
print('num of test predictions')
print(len(y_predict))

print('acc')
acc = accuracy_score(y_predict, np.asarray(finalTestLabel))
print(acc)

print('precision')
precision = precision_score(y_predict, np.asarray(finalTestLabel), average='weighted')
print(precision)

print('recall')
recall = recall_score(y_predict, np.asarray(finalTestLabel), average='weighted')
print(recall)

print('matrix')
matrix = confusion_matrix(y_predict, np.asarray(finalTestLabel))
print(matrix)
