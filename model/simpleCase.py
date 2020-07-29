from tensorflow import keras
import numpy as np
import json
from keras.layers import Dense, Activation, Embedding, Bidirectional, GRU, Flatten, Dropout
from keras import regularizers
import keras

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


def Bully(text):
    s_q = keras.preprocessing.text.text_to_word_sequence(text)
    s_v = [WordIdx.get(i, WordIdx['_UNK_']) for i in s_q]
    res = model.predict(np.asarray([s_v]))
    value = res[0][0]
    if value > 0.5:
        return text, value, '恶意言论'
    else:
        return text, value, '不是恶意言论'


print(Bully('fuck you!'))
print(Bully('Its fucking great!'))
