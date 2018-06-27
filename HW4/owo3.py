import os
import random
import datetime
import numpy as np
import pandas as pd
from gensim.models import word2vec
from keras import backend, optimizers, initializers
from keras.models import Model
from keras.layers import Conv1D, Dense, Embedding, Flatten, Input, MaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

E = ['AllTogether', 'Baseball', 'Boy-Girl', 'CVS', 'C_chat', 'GameSale', 'GetMarry', 'Lifeismoney', 'LoL', 'MH', 'MLB', 'Mobilecomm', 'movie', 'MuscleBeach', 'NBA', 'SENIORHIGH', 'Stock', 'Tennis', 'Tos', 'WomenTalk']

def loss(y_true, y_pred):
	x = backend.abs(y_true - y_pred)
	return backend.mean(x + (1/2)*x**2 + (1/6)*x**3)

def build_w2v(min_count):
	sentences = [open('./Testing/{index}.txt'.format(index=str(index)), 'r', encoding='utf-8').read().strip().split(' / ') for index in range(1000)]
	for i in E:
		for j in os.listdir('./Training/{key}_cut/'.format(key=i)):
			sentences.append(open("./Training/{key}_cut/{file}".format(key=i, file=j), 'r', encoding='utf-8').read().split(' / '))
	random.shuffle(sentences)
	word2Vec = word2vec.Word2Vec(sentences, min_count=min_count, size=270, iter=10, sg=1, workers=8)
	word2Vec.save("word2vec{count}.model".format(count=min_count))

def train(word2Vec, train_df, test_df, max_length, filters, kernel_size, pool_size, dense):
	r = random.randint(1, 10000)
	now = datetime.datetime.now()

	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(word2Vec.wv.vocab.keys())

	train_x = tokenizer.texts_to_sequences(train_df['text'])
	train_x = pad_sequences(train_x, maxlen=max_length)

	# train_y_good = train_df["good"]
	# train_y_bad = train_df["bad"]
	train_y = pd.DataFrame(train_df, columns=["good", "bad"])

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
	model = Conv1D(name="conv1D", filters=filters, kernel_size=kernel_size, kernel_initializer=initializers.glorot_uniform(seed=1), padding='same')(model)
	model = MaxPooling1D(name="maxPooling1D", pool_size=pool_size, strides=1, padding='same')(model)
	model = Flatten(name="flatten")(model)
	model = Dense(name="dense", output_dim=dense, kernel_initializer=initializers.glorot_uniform(seed=1), activation='relu')(model)
	model = Dense(name="output", output_dim=2, kernel_initializer=initializers.glorot_uniform(seed=1), activation='relu')(model)
	model = Model(inputs=inputs, outputs=model)
	model.compile(loss='mae', optimizer=optimizers.Adam(lr=.001), metrics=['mse'])
	model.summary()

	epochs = 15
	callback = model.fit(x=train_x, y=train_y, epochs=epochs, validation_split=.3, batch_size=20, verbose=1).history

	test_y = np.rint(model.predict(x=test_x, batch_size=10, verbose=1)).astype('int')

	seconds = str((datetime.datetime.now() - now).seconds)

	with open('test{seconds}_{r}.txt'.format(seconds=seconds, r=r), 'w') as file:
		file.write('id,good,bad\n')
		for index, data in enumerate(test_y):
			file.write('{},{},{}\n'.format(index, data[0], data[1]))
	
	with open('record{seconds}_{r}.log'.format(seconds=seconds, r=r), 'w') as file:
		file.write('result\t\n\n')
		file.write('\t'.join(['index', 'loss\t\t', 'mse\t\t\t', 'val_loss\t\t', 'val_mse\t']) + '\n')
		for index, loss, mse, val_loss, val_mse in zip(range(1, epochs+1), callback['loss'], callback['mean_squared_error'], callback['val_loss'], callback['val_mean_squared_error']):
			file.write('\t'.join([str(index) + '\t', '{:.12f}'.format(loss), '{:.12f}'.format(mse), '{:.12f}'.format(val_loss), '{:.12f}'.format(val_mse)]) + '\n')
		file.write('\nmax_length={max_length}\nmin_count={min_count}, size=270, iter=10, sg=1, workers=10\n'.format(max_length=max_length, min_count=min_count))
		file.write('inputs = Input(shape=(max_length,)\n')
		file.write('model = Embedding(name="embedding", input_dim=len(word_index)+1, output_dim=word2Vec.vector_size, weights=[embedding_matrix], input_length=max_length)(inputs)\n')
		file.write('model = Conv1D(name="conv1D_good", filters={filters}, kernel_size={kernel_size}, kernel_initializer=initializers.glorot_uniform(seed=1), padding="same")(model)\n'.format(filters=filters, kernel_size=kernel_size))
		file.write('model = MaxPooling1D(name="maxPooling1D", pool_size={pool_size}, strides=1, padding="same")(model)\n'.format(pool_size=pool_size))
		file.write('model = Dense(name="dense", output_dim={dense}, kernel_initializer=initializers.glorot_uniform(seed=1), activation="relu")(model)\n'.format(dense=dense))
		file.write('model = Dense(name="output", output_dim=2, kernel_initializer=initializers.glorot_uniform(seed=1), activation="relu")(model)\n')
		file.write('model = Model(inputs=inputs, outputs=model)\n')
		file.write('model.compile(loss="mae", optimizer=optimizers.Adam(lr=.001), metrics=["mse"])\n')

	import matplotlib.pyplot as plt
	fig = plt.figure()
	plt.grid(True)
	plt.ylim(0, 40)
	plt.plot(callback['loss'])
	plt.plot(callback['mse'])
	plt.plot(callback['val_loss'])
	plt.plot(callback['val_mse'])
	plt.title('model loss')
	plt.ylabel('loss (mae)')
	plt.xlabel('epoch')
	plt.legend(['train_loss', 'train_mse', 'test_loss', 'test_mse'], loc='upper right')
	fig.savefig('{seconds}_{r}.png'.format(seconds=seconds, r=r), dpi=fig.dpi)

if __name__ == '__main__':
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

	for min_count in range(2, 4):
		# build_w2v(min_count)
		word2Vec = word2vec.Word2Vec.load("word2vec{min_count}.model".format(min_count=min_count))
		for max_length in [200, 250, 300, 350, 400]:
			for filters in range(2, 10):
				for kernel_size in range(2, 10):
					for pool_size in range(2, 10):
						for dense in range(10, 100, 10):
							train(word2Vec, train_df, test_df, max_length, filters, kernel_size, pool_size, dense)
