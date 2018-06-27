import os
import numpy as np
from gensim.models import word2vec
from keras.utils import to_categorical
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

E = {'Japan_Travel': 0, 'KR_ENTERTAIN': 1, 'Makeup': 2, 'Tech_Job': 3, 'WomenTalk': 4, 'babymother': 5, 'e-shopping': 6, 'graduate': 7, 'joke': 8, 'movie': 9}

max_length = 1000
word2Vec = word2vec.Word2Vec.load("word2vec2.model")
train_word, train_index = [], []
for i in E.keys():
	for j in os.listdir('./training/{key}_cut/'.format(key=i)):
		train_word.append(open("./training/{key}_cut/{file}".format(key=i, file=j), 'r', encoding='utf-8').read().strip().split(' / '))
		train_index.append(E[i])
test_word = [open('./testing/{index}.txt'.format(index=str(index)), 'r', encoding='utf-8').read().strip().split(' / ') for index in range(1000)]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_word + test_word)
train_x = tokenizer.texts_to_sequences(train_word)
train_x = pad_sequences(train_x, maxlen=max_length)
train_y = to_categorical(np.asarray(train_index))
test_x = tokenizer.texts_to_sequences(test_word)
test_x = pad_sequences(test_x, maxlen=max_length)

model = load_model('owo.model')

test_y = model.predict(x=test_x, batch_size=10, verbose=1)

with open('test.txt', 'w') as file:
	file.write('id,category\n')
	for count, i in zip(range(1000), test_y):
		file.write('{},{}\n'.format(count, i.argmax()))