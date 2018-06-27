import numpy as np
import pandas as pd
from keras import optimizers
from keras import initializers
from keras.models import Sequential
from keras.layers import Activation, Dense
from sklearn.preprocessing import MinMaxScaler

def preprocessData(df):
	df['age'] = df['age'].fillna(df['age'].mean())
	df['fare'] = df['fare'].fillna(df['fare'].mean())
	df['boat'] = df['boat'].fillna(0)
	df['sex'] = df['sex'].map({'female': 0, 'male': 1}).astype(int)
	df['family'] = df['sibsp'] + df['parch']
	df['name'] = df['name'].apply(lambda x: 1 if 'Mr.' in x else 2 if 'Miss.' in x else 3 if 'Mrs.' in x else 4 if 'Master.' in x else 0)
	df['boat'] = df['boat'].apply(lambda x: 1 if x != 0 else 0)
	df = df.drop(['sibsp'], axis=1).drop('parch', axis=1)
	ndarray = pd.get_dummies(data=df, columns=["embarked"]).values
	label = ndarray[:, 0]
	features = ndarray[:, 1:]
	minmax_scale = MinMaxScaler(feature_range=(0, 1))
	features = minmax_scale.fit_transform(features)
	return features, label

if __name__ == '__main__':
	df = pd.read_excel('training data(1000).xlsx')
	train_Features, train_label = preprocessData(df.loc[:,['survived', 'pclass', 'name', 'sex', 'age', 'sibsp', 'parch', 'fare', 'boat', 'embarked']])
	df = pd.read_excel('testing data.xlsx')
	test_Features, test_label = preprocessData(df.loc[:,['survived', 'pclass', 'name', 'sex', 'age', 'sibsp', 'parch', 'fare', 'boat', 'embarked']])

	model = Sequential()
	model.add(Dense(units=80, input_dim=10, kernel_initializer=initializers.RandomUniform(minval=-1, maxval=1, seed=1)))
	model.add(Activation('relu')) # f(x) = max(x, 0)
	# model.add(Dense(units=60, kernel_initializer=initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=1)))
	# model.add(Activation('relu'))
	model.add(Dense(units=60, kernel_initializer=initializers.RandomUniform(minval=-1, maxval=1, seed=1)))
	model.add(Activation('relu'))
	# model.add(Dense(units=30, kernel_initializer=initializers.RandomUniform(minval=-1, maxval=1, seed=1)))
	# model.add(Activation('relu'))
	model.add(Dense(units=1, kernel_initializer=initializers.RandomUniform(minval=-1, maxval=1, seed=1)))
	model.add(Activation('sigmoid')) # f(x) = 1/(1+e^(-x))
	model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=.0008), metrics=['accuracy'])
	model.summary()

	model.fit(x=train_Features, y=train_label, epochs=500, validation_split=.1, batch_size=20, verbose=1)
	
	print(model.evaluate(x=train_Features, y=train_label))

	with open('test.txt', 'w') as file:
		file.write('id,survived\n')
		for count, i in zip(range(309), model.predict(x=test_Features, batch_size=10, verbose=0)):
			file.write('{},{}\n'.format(count, int(round(i[0]))))
