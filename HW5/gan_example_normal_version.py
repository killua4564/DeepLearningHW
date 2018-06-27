import os
import glob
import datetime
import numpy as np
import pandas as pd
from scipy.misc.pilutil import imread, imresize
from keras import backend
from keras.initializers import glorot_uniform
from keras.models import Model, Sequential
from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, LeakyReLU, UpSampling2D
from keras.layers.merge import Concatenate
from keras.optimizers import Adam, RMSprop

backend.set_image_data_format("channels_first")
os.environ["CUDA_VISIBLE_DEVICES"] = '-1' 

def Discriminator(input_shape):
    img_A, img_B = Input(input_shape), Input(input_shape)
    x = Concatenate(axis=1)([img_A, img_B])
    x = Conv2D(filters=4, kernel_size=(4, 4), strides=(2, 2), kernel_initializer=glorot_uniform(seed=1), padding='same', activation=LeakyReLU(alpha=.2))(x)
    x = Conv2D(filters=8, kernel_size=(4, 4), strides=(2, 2), kernel_initializer=glorot_uniform(seed=1), padding='same', activation=LeakyReLU(alpha=.2))(x)
    x = BatchNormalization(epsilon=.000001, momentum=.5)(x)
    x = Conv2D(filters=16, kernel_size=(4, 4), strides=(2, 2), kernel_initializer=glorot_uniform(seed=1), padding='same', activation=LeakyReLU(alpha=.2))(x)
    x = BatchNormalization(epsilon=.000001, momentum=.5)(x)
    x = Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), kernel_initializer=glorot_uniform(seed=1), padding='same', activation=LeakyReLU(alpha=.2))(x)
    x = BatchNormalization(epsilon=.000001, momentum=.5)(x)
    x = Conv2D(filters=1, kernel_size=(4, 4), strides=(1, 1), kernel_initializer=glorot_uniform(seed=1), padding='same')(x)
    return Model(inputs=[img_A, img_B], outputs=x)
    
def Generator(input_shape):
    img = Input(input_shape)
    x1 = Conv2D(filters=4, kernel_size=(4, 4), strides=(2, 2), kernel_initializer=glorot_uniform(seed=1), padding='same', activation=LeakyReLU(alpha=.2))(img)
    x2 = Conv2D(filters=8, kernel_size=(4, 4), strides=(2, 2), kernel_initializer=glorot_uniform(seed=1), padding='same', activation=LeakyReLU(alpha=.2))(x1)
    x2 = BatchNormalization(epsilon=.000001, momentum=.5)(x2)
    x3 = Conv2D(filters=16, kernel_size=(4, 4), strides=(2, 2), kernel_initializer=glorot_uniform(seed=1), padding='same', activation=LeakyReLU(alpha=.2))(x2)
    x3 = BatchNormalization(epsilon=.000001, momentum=.5)(x3)
    x4 = Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), kernel_initializer=glorot_uniform(seed=1), padding='same', activation=LeakyReLU(alpha=.2))(x3)
    x4 = BatchNormalization(epsilon=.000001, momentum=.5)(x4)
    x5 = Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), kernel_initializer=glorot_uniform(seed=1), padding='same', activation=LeakyReLU(alpha=.2))(x4)
    x5 = BatchNormalization(epsilon=.000001, momentum=.5)(x5)

    x5 = UpSampling2D(size=(2, 2))(x5)
    x5 = Conv2D(filters=32, kernel_size=(4, 4), strides=(1, 1), kernel_initializer=glorot_uniform(seed=1), padding='same', activation='relu')(x5)
    x5 = BatchNormalization(epsilon=.000001, momentum=.5)(x5)
    x4 = Concatenate(axis=1)([x5, x4])
    x4 = UpSampling2D(size=(2, 2))(x4)
    x4 = Conv2D(filters=16, kernel_size=(4, 4), strides=(1, 1), kernel_initializer=glorot_uniform(seed=1), padding='same', activation='relu')(x4)
    x4 = BatchNormalization(epsilon=.000001, momentum=.5)(x4)
    x3 = Concatenate(axis=1)([x4, x3])
    x3 = UpSampling2D(size=(2, 2))(x3)
    x3 = Conv2D(filters=8, kernel_size=(4, 4), strides=(1, 1), kernel_initializer=glorot_uniform(seed=1), padding='same', activation='relu')(x3)
    x3 = BatchNormalization(epsilon=.000001, momentum=.5)(x3)
    x2 = Concatenate(axis=1)([x2, x2])
    x2 = UpSampling2D(size=(2, 2))(x2)
    x2 = Conv2D(filters=4, kernel_size=(4, 4), strides=(1, 1), kernel_initializer=glorot_uniform(seed=1), padding='same', activation='relu')(x2)
    x2 = BatchNormalization(epsilon=.000001, momentum=.5)(x2)
    x1 = Concatenate(axis=1)([x2, x1])
    x1 = UpSampling2D(size=(2, 2))(x1)
    x1 = Conv2D(filters=1, kernel_size=(4, 4), strides=(1, 1), kernel_initializer=glorot_uniform(seed=1), padding='same', activation='tanh')(x1)
    return Model(inputs=img, outputs=x1)

def generator_img(real_list_dir, white_list_dir, resize=None, batch_size=32):
    batch_real_img = []
    batch_white_img = []
    for _ in range(batch_size):
        random_index = np.random.randint(len(real_list_dir))
        real_img = imread(real_list_dir[random_index], mode='L')
        white_img = imread(white_list_dir[random_index], mode='L')
        if resize:
            real_img = imresize(real_img, resize)
            white_img = imresize(white_img, resize)
        batch_real_img.append(real_img)
        batch_white_img.append(white_img)
    batch_real_img = np.array(batch_real_img) / 127.5 - 1
    batch_real_img = np.expand_dims(batch_real_img, axis=1)
    batch_white_img = np.array(batch_white_img) / 127.5 - 1
    batch_white_img = np.expand_dims(batch_white_img, axis=1)
    return batch_real_img, batch_white_img

if __name__ == '__main__':
    real_list_dir = glob.glob('./Training/Real/*')
    white_list_dir = glob.glob('./Training/White/*')

    batch_size = 64
    all_epoch = 5600
    channels, img_row, img_col = 1, 128, 128
    input_shape = (channels, img_row, img_col)
    
    G = Generator(input_shape)
    D = Discriminator(input_shape)
    D.compile(loss='mse', optimizer=Adam(lr=.0002, beta_1=.5), metrics=['accuracy'])
    D.summary()
    D.trainable = False
    
    img_A = Input(input_shape)
    img_B = Input(input_shape)
    fake = G(img_B)
    AM = Model(inputs=[img_A, img_B], outputs=[D([fake, img_B]), fake])
    AM.compile(loss=['mse', 'mae'], optimizer=RMSprop(lr=.0001, decay=3e-8), loss_weights=[1, 1], metrics=['accuracy'])
    AM.summary()

    valid = np.ones((batch_size, channels, 8, 8))
    fake = np.zeros((batch_size, channels, 8, 8))

    now = datetime.datetime.now()
    for now_iter in range(all_epoch):
        ori_img, white_img = generator_img(real_list_dir=real_list_dir, white_list_dir=white_list_dir, resize=(img_row, img_col), batch_size=batch_size)
        fake_img = G.predict(white_img) 

        D_loss_Real = D.train_on_batch([ori_img, white_img], valid)
        D_loss_Fake = D.train_on_batch([ori_img, fake_img], fake)
        D_loss = 0.5 * np.add(D_loss_Real, D_loss_Fake)
        G_loss = AM.train_on_batch([ori_img, white_img], [valid, ori_img])
        
        print("[Epoch {now_iter}/{all_epoch}] [D loss: {D_loss_0}, acc: {D_loss_1}] [G loss1: {G_loss_0}, loss2: {G_loss_1}] [time:{time}]".format(
            now_iter=now_iter,
            all_epoch=all_epoch,
            D_loss_0=D_loss[0],
            D_loss_1='{:.3f}'.format(D_loss[1]*100),
            G_loss_0=G_loss[0],
            G_loss_1=G_loss[1],
            time=datetime.datetime.now() - now
        ))

    import matplotlib.pyplot as plt
    n, r, c = 2, 2, 3
    plt.gray()
    plt.figure(figsize=(c*6, r*6))
    for i in range(r):
        ori_img, white_img = generator_img(real_list_dir=real_list_dir, white_list_dir=white_list_dir, resize=(img_row,img_col), batch_size=1)
        ax = plt.subplot(r, c, i*c + 1)
        plt.imshow(G.predict(white_img).reshape(img_row, img_col))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(r, c, i*c + 2)
        plt.imshow(ori_img.reshape(img_row, img_col))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(r, c, i*c + 3)
        plt.imshow(white_img.reshape(img_row, img_col))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)   
    plt.show()