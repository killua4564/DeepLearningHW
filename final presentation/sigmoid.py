from keras import optimizers, initializers
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

training_set = ImageDataGenerator(rescale=1./255).flow_from_directory('dataset/training_set', target_size=(64, 64), batch_size=32, class_mode='binary')
testing_set = ImageDataGenerator(rescale=1./255).flow_from_directory('dataset/testing_set', target_size=(64, 64), batch_size=32, class_mode='binary')

classifier = Sequential()
classifier.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(64, 64, 3), kernel_initializer=initializers.glorot_uniform(seed=1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=1))
classifier.add(Flatten())
classifier.add(Dense(units=128, kernel_initializer=initializers.glorot_uniform(seed=1), activation='relu'))
classifier.add(Dense(units=1, kernel_initializer=initializers.glorot_uniform(seed=1), activation='sigmoid'))
classifier.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=.0001), metrics=['accuracy'])
classifier.summary()

classifier.fit_generator(generator=training_set, steps_per_epoch=5, epochs=2)
result = classifier.predict_generator(generator=testing_set)

print(result)