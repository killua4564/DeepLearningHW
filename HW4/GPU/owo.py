import os
import random
import datetime
import numpy as np
import pandas as pd
from gensim.models import word2vec
from keras import backend, optimizers, initializers
from keras.utils import to_categorical
from keras.models import Model, Sequential
from keras.layers import BatchNormalization, Conv1D, concatenate, Dense, Dropout, Embedding, Flatten, GRU, Input, Lambda, LeakyReLU, MaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

E = ['AllTogether', 'Baseball', 'Boy-Girl', 'CVS', 'C_chat', 'GameSale', 'GetMarry', 'Lifeismoney', 'LoL', 'MH', 'MLB', 'Mobilecomm', 'movie', 'MuscleBeach', 'NBA', 'SENIORHIGH', 'Stock', 'Tennis', 'Tos', 'WomenTalk']

def loss(y_true, y_pred):
	return backend.mean(backend.abs(y_true - y_pred))

def build_w2v(min_count):
	sentences = [open('./Testing/{index}.txt'.format(index=str(index)), 'r', encoding='utf-8').read().strip().split(' / ') for index in range(1000)]
	for i in E:
		for j in os.listdir('./Training/{key}_cut/'.format(key=i)):
			sentences.append(open("./Training/{key}_cut/{file}".format(key=i, file=j), 'r', encoding='utf-8').read().split(' / '))
	random.shuffle(sentences)
	word2Vec = word2vec.Word2Vec(sentences, min_count=min_count, size=270, iter=10, sg=1, workers=10)
	word2Vec.save("word2vec{count}.model".format(count=min_count))

if __name__ == '__main__':
	build_w2v(3)

	max_length = 300
	now = datetime.datetime.now()
	word2Vec = word2vec.Word2Vec.load("word2vec2.model")

	train_list = []
	for category, i in enumerate(E):
		category_index = open("./Training/{key}.txt".format(key=i), 'r', encoding='utf-8').read().split('\n')
		for j in range(1800):
			train_word = open("./Training/{key}_cut/{key}_cut{index}.txt".format(key=i, index=str(j)), 'r', encoding='utf-8').read().strip().split(' / ')
			train_up, train_down = category_index[j].split('推')[-1].split('噓')
			train_list.append([category, train_word, train_up, train_down])
	test_list = [[open('./Testing/{index}.txt'.format(index=str(index)), 'r', encoding='utf-8').read().strip().split(' / ')] for index in range(4000)]
	
	train_df = pd.DataFrame(train_list, columns=["category", "text", "good", "bad"]).sample(frac=1)
	test_df = pd.DataFrame(test_list, columns=["text"])

	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(train_df['text'].append(test_df['text']))

	train_x = tokenizer.texts_to_sequences(train_df['text'])
	train_x = pad_sequences(train_x, maxlen=max_length)

	train_y_good = train_df["good"]
	train_y_bad = train_df["bad"]
	# train_y = pd.DataFrame(train_df, columns=["good", "bad"])

	test_x = tokenizer.texts_to_sequences(test_df['text'])
	test_x = pad_sequences(test_x, maxlen=max_length)

	word_index = tokenizer.word_index
	
	embedding_matrix = np.zeros((len(word_index)+1, word2Vec.vector_size))
	for word, i in word_index.items():
		try:
			embedding_matrix[i] = word2Vec[word]
		except:
			continue

	inputs = Input(shape=(max_length,))
	model = Embedding(name="embedding", input_dim=len(word_index)+1, output_dim=word2Vec.vector_size, weights=[embedding_matrix], input_length=max_length)(inputs)
	
	good = Conv1D(name="conv1D_good", filters=128, kernel_size=3)(model)
	good = BatchNormalization(name="batchNormalization_good", epsilon=.000001, momentum=.5)(good)
	good = MaxPooling1D(name="maxPooling1D_good", pool_size=3, strides=1)(good)
	good = Flatten(name="flatten_good")(good)
	good = Dense(name="dense_good_1", output_dim=64, kernel_initializer=initializers.glorot_uniform(seed=1))(good)
	good = Dense(name="dense_good_2", output_dim=64, kernel_initializer=initializers.glorot_uniform(seed=1))(good)
	# good = Dropout(name="dropout_good", rate=.2)(good)
	good = Dense(name="good", output_dim=1, kernel_initializer=initializers.glorot_uniform(seed=1), activation='relu')(good)

	bad = Conv1D(name="conv1D_bad", filters=128, kernel_size=3)(model)
	bad = BatchNormalization(name="batchNormalization_bad", epsilon=.000001, momentum=.5)(bad)
	bad = Lambda(name="lambda_bad", function=lambda x: -x)(bad)
	bad = MaxPooling1D(name="maxPooling1D_bad", pool_size=3, strides=1)(bad)
	bad = Flatten(name="flatten_bad")(bad)
	bad = Dense(name="dense_bad_1", output_dim=64, kernel_initializer=initializers.glorot_uniform(seed=1))(bad)
	bad = Dense(name="dense_bad_2", output_dim=64, kernel_initializer=initializers.glorot_uniform(seed=1))(bad)
	# bad = Dropout(name="dropout_bad", rate=.2)(bad)
	bad = Dense(name="bad", output_dim=1, kernel_initializer=initializers.glorot_uniform(seed=1), activation='relu')(bad)

	model = Model(inputs=inputs, outputs=[good, bad])
	model.compile(loss='mae', optimizer=optimizers.Adam(lr=.001), metrics=[])
	model.summary()

	epochs = 10
	callback = model.fit(x=train_x, y=[train_y_good, train_y_bad], epochs=epochs, validation_split=.3, batch_size=20, verbose=1).history

	test_y = np.rint(model.predict(x=test_x, batch_size=10, verbose=1)).astype('int')

	seconds = str((datetime.datetime.now() - now).seconds)

	print(seconds)

	with open('test{seconds}.txt'.format(seconds=seconds), 'w') as file:
		file.write('id,good,bad\n')
		for index, good, bad in zip(range(4000), test_y[0], test_y[1]):
			file.write('{},{},{}\n'.format(index, good[0], bad[0]))
	
	with open('record{seconds}.log'.format(seconds=seconds), 'w') as file:
		file.write('result\t\n\n')
		file.write('\t'.join(['index', 'loss\t\t', 'good_loss\t', 'bad_loss\t', 'val_loss\t', 'val_good_loss', 'val_bad_loss']) + '\n')
		for index, loss, good_loss, bad_loss, val_loss, val_good_loss, val_bad_loss in zip(range(1, epochs+1), callback['loss'], callback['good_loss'], callback['bad_loss'], callback['val_loss'], callback['val_good_loss'], callback['val_bad_loss']):
			file.write('\t'.join([str(index) + '\t', '{:.12f}'.format(loss), '{:.12f}'.format(good_loss), '{:.12f}'.format(bad_loss), '{:.12f}'.format(val_loss), '{:.12f}'.format(val_good_loss), '{:.12f}'.format(val_bad_loss)]) + '\n')
		file.write('\nmax_length={max_length}\nmin_count=2, size=270, iter=10, sg=1, workers=10\n'.format(max_length=max_length))

	import matplotlib.pyplot as plt
	fig = plt.figure()
	plt.plot(callback['loss'])
	plt.plot(callback['good_loss'])
	plt.plot(callback['bad_loss'])
	plt.plot(callback['val_loss'])
	plt.plot(callback['val_good_loss'])
	plt.plot(callback['val_bad_loss'])
	plt.title('model loss')
	plt.ylabel('loss (mae)')
	plt.xlabel('epoch')
	plt.legend(['train', 'train_good', 'train_bad', 'test', 'test_good', 'test_bad'], loc='upper right')
	fig.savefig('{seconds}.png'.format(seconds=seconds), dpi=fig.dpi)