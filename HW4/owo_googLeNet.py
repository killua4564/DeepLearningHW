import os
import random
import datetime
import numpy as np
import pandas as pd
from gensim.models import word2vec
from keras import backend, optimizers, initializers
from keras.utils import to_categorical
from keras.models import Model, Sequential
from keras.layers import AveragePooling1D, Conv1D, concatenate, Dense, Dropout, Embedding, Flatten, GRU, Input, MaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

E = ['AllTogether', 'Baseball', 'Boy-Girl', 'CVS', 'C_chat', 'GameSale', 'GetMarry', 'Lifeismoney', 'LoL', 'MH', 'MLB', 'Mobilecomm', 'movie', 'MuscleBeach', 'NBA', 'SENIORHIGH', 'Stock', 'Tennis', 'Tos', 'WomenTalk']

def loss(y_true, y_pred):
	return backend.mean(backend.abs(y_true - y_pred))

def build_w2v():
	sentences = [open('./Testing/{index}.txt'.format(index=str(index)), 'r', encoding='utf-8').read().strip().split(' / ') for index in range(1000)]
	for i in E:
		for j in os.listdir('./Training/{key}_cut/'.format(key=i)):
			sentences.append(open("./Training/{key}_cut/{file}".format(key=i, file=j), 'r', encoding='utf-8').read().split(' / '))
	random.shuffle(sentences)
	for min_count in range(2, 3): # <--- min_count
		word2Vec = word2vec.Word2Vec(sentences, min_count=min_count, size=270, iter=10, sg=1, workers=10)
		word2Vec.save("word2vec{count}.model".format(count=min_count))

if __name__ == '__main__':
	# build_w2v()

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

	train_y_category = to_categorical(np.asarray(list(train_df["category"])))
	train_y_good = train_df["good"]
	train_y_bad = train_df["bad"]

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
	embedding = Embedding(name="embedding", input_dim=len(word_index)+1, output_dim=word2Vec.vector_size, weights=[embedding_matrix], input_length=max_length)(inputs)

	category_gru = GRU(name="category_gru", units=16, kernel_initializer=initializers.glorot_uniform(seed=1), activation='tanh', dropout=.2, recurrent_dropout=.1)(embedding)
	category_dense1 = Dense(name="category_dense1", units=100, kernel_initializer=initializers.glorot_uniform(seed=1), activation='relu')(category_gru)
	category_dense2 = Dense(name="category_dense2", units=100, kernel_initializer=initializers.glorot_uniform(seed=1), activation='relu')(category_dense1)
	category_output = Dense(name="category", units=20, kernel_initializer=initializers.glorot_uniform(seed=1), activation='softmax')(category_dense2)

	good_1x1 = Conv1D(name="good_1x1", filters=64, kernel_size=1, strides=1, kernel_initializer=initializers.glorot_uniform(seed=1), padding='same', activation='relu')(embedding)
	good_3x3_reduce = Conv1D(name="good_3x3_reduce", filters=96, kernel_size=1, strides=1, kernel_initializer=initializers.glorot_uniform(seed=1), padding='same', activation='relu')(embedding)
	good_3x3 = Conv1D(name="good_3x3", filters=64, kernel_size=3, strides=3, kernel_initializer=initializers.glorot_uniform(seed=1), padding='same', activation='relu')(good_3x3_reduce)
	good_5x5_reduce = Conv1D(name="good_5x5_reduce", filters=16, kernel_size=1, strides=1, kernel_initializer=initializers.glorot_uniform(seed=1), padding='same', activation='relu')(embedding)
	good_5x5 = Conv1D(name="good_5x5", filters=64, kernel_size=5, strides=5, kernel_initializer=initializers.glorot_uniform(seed=1), padding='same', activation='relu')(good_5x5_reduce)
	good_pool = MaxPooling1D(name="good_pool", pool_size=3, strides=1, padding='same')(embedding)
	good_pool_proj = Conv1D(name="good_pool_proj", filters=64, kernel_size=1, strides=1, kernel_initializer=initializers.glorot_uniform(seed=1), padding='same', activation='relu')(good_pool)
	good_merge = concatenate(name="good_merge", inputs=[good_1x1, good_3x3, good_5x5, good_pool_proj], axis=1)
	good_ave_pool = AveragePooling1D(name="good_ave_pool", pool_size=5, strides=3)(good_merge)
	good_conv = Conv1D(name="good_conv", filters=128, kernel_size=1, strides=1, kernel_initializer=initializers.glorot_uniform(seed=1), padding='same', activation='relu')(good_ave_pool)
	good_flat = Flatten(name="good_flat")(good_conv)
	good_dense = Dense(name="good_dense", output_dim=256, kernel_initializer=initializers.glorot_uniform(seed=1), activation='relu')(good_flat)
	good_drop = Dropout(name="good_drop", rate=.2)(good_dense)
	good_output = Dense(name="good", output_dim=1, kernel_initializer=initializers.glorot_uniform(seed=1), activation='relu')(good_drop)

	bad_1x1 = Conv1D(name="bad_1x1", filters=64, kernel_size=1, strides=1, kernel_initializer=initializers.glorot_uniform(seed=1), padding='same', activation='relu')(embedding)
	bad_3x3_reduce = Conv1D(name="bad_3x3_reduce", filters=96, kernel_size=1, strides=1, kernel_initializer=initializers.glorot_uniform(seed=1), padding='same', activation='relu')(embedding)
	bad_3x3 = Conv1D(name="bad_3x3", filters=64, kernel_size=3, strides=3, kernel_initializer=initializers.glorot_uniform(seed=1), padding='same', activation='relu')(bad_3x3_reduce)
	bad_5x5_reduce = Conv1D(name="bad_5x5_reduce", filters=16, kernel_size=1, strides=1, kernel_initializer=initializers.glorot_uniform(seed=1), padding='same', activation='relu')(embedding)
	bad_5x5 = Conv1D(name="bad_5x5", filters=64, kernel_size=5, strides=5, kernel_initializer=initializers.glorot_uniform(seed=1), padding='same', activation='relu')(bad_5x5_reduce)
	bad_pool = MaxPooling1D(name="bad_pool", pool_size=3, strides=1, padding='same')(embedding)
	bad_pool_proj = Conv1D(name="bad_pool_proj", filters=64, kernel_size=1, strides=1, kernel_initializer=initializers.glorot_uniform(seed=1), padding='same', activation='relu')(bad_pool)
	bad_merge = concatenate(name="bad_merge", inputs=[bad_1x1, bad_3x3, bad_5x5, bad_pool_proj], axis=1)
	bad_ave_pool = AveragePooling1D(name="bad_ave_pool", pool_size=5, strides=3)(bad_merge)
	bad_conv = Conv1D(name="bad_conv", filters=128, kernel_size=1, strides=1, kernel_initializer=initializers.glorot_uniform(seed=1), padding='same', activation='relu')(bad_ave_pool)
	bad_flat = Flatten(name="bad_flat")(bad_conv)
	bad_dense = Dense(name="bad_dense", output_dim=256, kernel_initializer=initializers.glorot_uniform(seed=1), activation='relu')(bad_flat)
	bad_drop = Dropout(name="bad_drop", rate=.2)(bad_dense)
	bad_output = Dense(name="bad", output_dim=1, kernel_initializer=initializers.glorot_uniform(seed=1), activation='relu')(bad_drop)

	model = Model(inputs=inputs, outputs=[category_output, good_output, bad_output])
	model.compile(loss='mae', optimizer=optimizers.Adam(lr=.001), metrics=[])
	model.summary()
	
	epochs = 100

	callback = model.fit(x=train_x, y=[train_y_category, train_y_good, train_y_bad], epochs=epochs, validation_split=.3, batch_size=1000, verbose=1).history

	test_y = model.predict(x=test_x, batch_size=10, verbose=1)

	test_y_good = np.rint(test_y[1]).astype('int')
	test_y_bad = np.rint(test_y[2]).astype('int')

	seconds = str((datetime.datetime.now() - now).seconds)

	print(seconds)

	with open('test{seconds}.txt'.format(seconds=seconds), 'w') as file:
		file.write('id,good,bad\n')
		for index, good, bad in zip(range(4000), test_y_good, test_y_bad):
			file.write('{},{},{}\n'.format(index, good[0], bad[0]))
	
	with open('record{seconds}.log'.format(seconds=seconds), 'w') as file:
		file.write('result\t\n\n')
		file.write('\t'.join(['index', 'loss\t\t', 'category_loss\t', 'good_loss\t\t', 'bad_loss\t\t', 'val_loss\t\t', 'val_category_loss\t', 'val_good_loss\t\t', 'val_bad_loss']) + '\n')
		for index, loss, category_loss, good_loss, bad_loss, val_loss, val_category_loss, val_good_loss, val_bad_loss in zip(range(1, epochs+1), callback['loss'], callback['category_loss'], callback['good_loss'], callback['bad_loss'], callback['val_loss'], callback['val_category_loss'], callback['val_good_loss'], callback['val_bad_loss']):
			file.write('\t'.join([str(index) + '\t', '{:.12f}'.format(loss), '{:.12f}'.format(category_loss), '{:.12f}'.format(good_loss), '{:.12f}'.format(bad_loss), '{:.12f}'.format(val_loss), '{:.12f}'.format(val_category_loss), '{:.12f}'.format(val_good_loss), '{:.12f}'.format(val_bad_loss)]) + '\n')
		file.write('\nmax_length={max_length}\nmin_count=2, size=270, iter=10, sg=1, workers=10\n'.format(max_length=max_length))