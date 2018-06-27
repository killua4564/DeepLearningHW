import os
import random
import datetime
import numpy as np
import pandas as pd
from gensim.models import word2vec
from keras import optimizers, initializers
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Activation, Dense, Embedding, RNN, SimpleRNN, GRU, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

E = {'Japan_Travel': 0, 'KR_ENTERTAIN': 1, 'Makeup': 2, 'Tech_Job': 3, 'WomenTalk': 4, 'babymother': 5, 'e-shopping': 6, 'graduate': 7, 'joke': 8, 'movie': 9}

if __name__ == '__main__':
	max_length = 300
	now = datetime.datetime.now()
	word2Vec = word2vec.Word2Vec.load("word2vec2.model")

	train_list = []
	for i in E.keys():
		for j in os.listdir('./training/{key}_cut/'.format(key=i)):
			train_word = open("./training/{key}_cut/{file}".format(key=i, file=j), 'r', encoding='utf-8').read().strip().split(' / ')
			train_list.append([train_word, E[i]])
	train_df = pd.DataFrame(train_list, columns=["text", "category"]).sample(frac=1)
	test_word = [open('./testing/{index}.txt'.format(index=str(index)), 'r', encoding='utf-8').read().strip().split(' / ') for index in range(1000)]

	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(list(train_df["text"]) + test_word)

	train_x = tokenizer.texts_to_sequences(train_word)
	train_x = pad_sequences(train_x, maxlen=max_length)

	train_y = to_categorical(np.asarray(list(train_df["category"])))

	test_x = tokenizer.texts_to_sequences(test_word)
	test_x = pad_sequences(test_x, maxlen=max_length)

	word_index = tokenizer.word_index

	embedding_matrix = np.zeros((len(word_index)+1, word2Vec.vector_size))
	for word, i in word_index.items():
		try:
			embedding_matrix[i] = word2Vec[word]
		except:
			continue

	model = Sequential()
	model.add(Embedding(input_dim=len(word_index)+1, output_dim=word2Vec.vector_size, weights=[embedding_matrix], input_length=max_length, trainable=False))
	# model.add(LSTM(units=200, kernel_initializer=initializers.RandomUniform(minval=-1, maxval=1, seed=1), activation='tanh', return_sequences=True, recurrent_activation='hard_sigmoid', unroll=True))
	# model.add(GRU(units=120, kernel_initializer=initializers.RandomUniform(minval=-1, maxval=1, seed=1), activation='relu', return_sequences=True, recurrent_activation='hard_sigmoid', unroll=True))
	# model.add(SimpleRNN(units=60, kernel_initializer=initializers.RandomUniform(minval=-1, maxval=1, seed=1), activation='tanh', return_sequences=False, unroll=True))
	
	# model.add(LSTM(units=16, kernel_initializer=initializers.RandomUniform(minval=-1, maxval=1, seed=1), activation='tanh', return_sequences=False, recurrent_activation='hard_sigmoid', unroll=True))
	# model.add(Dense(units=100, kernel_initializer=initializers.RandomUniform(minval=-1, maxval=1, seed=1), activation='relu'))
	# model.add(Dense(units=100, kernel_initializer=initializers.RandomUniform(minval=-1, maxval=1, seed=1), activation='relu'))
	# model.add(Dense(units=10, kernel_initializer=initializers.RandomUniform(minval=-1, maxval=1, seed=1), activation='softmax'))
	# model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=.0012626), metrics=['accuracy'])
	
	model.add(LSTM(units=16, kernel_initializer=initializers.glorot_uniform(seed=1), activation='tanh', dropout=.2, recurrent_dropout=.1))
	model.add(Dense(units=100, kernel_initializer=initializers.glorot_uniform(seed=1), activation='relu'))
	model.add(Dense(units=100, kernel_initializer=initializers.glorot_uniform(seed=1), activation='relu'))
	model.add(Dense(units=10, kernel_initializer=initializers.glorot_uniform(seed=1), activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=.00087), metrics=['accuracy'])
	model.summary()

	epochs = 100

	callback = model.fit(x=train_x, y=train_y, epochs=epochs, validation_split=.3, batch_size=100, verbose=1).history
	
	test_y = model.predict(x=test_x, batch_size=10, verbose=1)

	seconds = str((datetime.datetime.now() - now).seconds)

	print(seconds)

	model.save('{seconds}.model'.format(seconds=seconds))

	with open('record{seconds}.log'.format(seconds=seconds), 'w') as file:
		file.write('\t'.join(['index', 'loss\t\t', 'acc\t\t\t', 'val_loss\t', 'val_acc']) + '\n')
		for index, loss, acc, val_loss, val_acc in zip(range(1, epochs+1), callback['loss'], callback['acc'], callback['val_loss'], callback['val_acc']):
			file.write('\t'.join([str(index) + '\t', '{:.12f}'.format(loss), '{:.12f}'.format(acc), '{:.12f}'.format(val_loss), '{:.12f}'.format(val_acc)]) + '\n')

	with open('test{seconds}.txt'.format(seconds=seconds), 'w') as file:
		file.write('id,category\n')
		for count, i in zip(range(1000), test_y):
			file.write('{},{}\n'.format(count, i.argmax()))
			