import os
import glob
import datetime
import numpy as np
import pandas as pd
from imageio import imwrite
from scipy.misc.pilutil import imread, imresize
from keras import backend
from keras.initializers import glorot_uniform
from keras.models import Model, Sequential
from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Dense, Dropout, Flatten, Input, LeakyReLU, UpSampling2D
from keras.layers.merge import Concatenate
from keras.optimizers import Adam, RMSprop

backend.set_image_data_format("channels_first")
os.environ["CUDA_VISIBLE_DEVICES"] = '-1' 

def Discriminator(input_shape):
    img_A, img_B = Input(input_shape), Input(input_shape)
    x = Concatenate(axis=1)([img_A, img_B])
    x = Conv2D(filters=4, kernel_size=(4, 4), strides=(1, 1), kernel_initializer=glorot_uniform(seed=1), padding='same', activation=LeakyReLU(alpha=.2))(x)
    x = Conv2D(filters=8, kernel_size=(4, 4), strides=(2, 2), kernel_initializer=glorot_uniform(seed=1), padding='same', activation=LeakyReLU(alpha=.2))(x)
    x = BatchNormalization(epsilon=.001, momentum=.9)(x)
    x = Conv2D(filters=16, kernel_size=(4, 4), strides=(2, 2), kernel_initializer=glorot_uniform(seed=1), padding='same', activation=LeakyReLU(alpha=.2))(x)
    x = BatchNormalization(epsilon=.001, momentum=.9)(x)
    x = Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), kernel_initializer=glorot_uniform(seed=1), padding='same', activation=LeakyReLU(alpha=.2))(x)
    x = BatchNormalization(epsilon=.001, momentum=.9)(x)
    x = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), kernel_initializer=glorot_uniform(seed=1), padding='same', activation=LeakyReLU(alpha=.2))(x)
    x = BatchNormalization(epsilon=.001, momentum=.9)(x)
    x = Conv2D(filters=128, kernel_size=(4, 4), strides=(1, 1), kernel_initializer=glorot_uniform(seed=1), padding='same', activation=LeakyReLU(alpha=.2))(x)
    x = BatchNormalization(epsilon=.001, momentum=.9)(x)
    x = Conv2D(filters=1, kernel_size=(4, 4), strides=(1, 1), kernel_initializer=glorot_uniform(seed=1), padding='same')(x)
    return Model(inputs=[img_A, img_B], outputs=x)
    
def Generator(input_shape):
    img = Input(input_shape)
    x1 = Conv2D(filters=2, kernel_size=(4, 4), strides=(2, 2), kernel_initializer=glorot_uniform(seed=1), padding='same', activation=LeakyReLU(alpha=.2))(img)
    x1 = BatchNormalization(epsilon=.001, momentum=.9)(x1)
    x2 = Conv2D(filters=4, kernel_size=(4, 4), strides=(2, 2), kernel_initializer=glorot_uniform(seed=1), padding='same', activation=LeakyReLU(alpha=.2))(x1)
    x2 = BatchNormalization(epsilon=.001, momentum=.9)(x2)
    x3 = Conv2D(filters=8, kernel_size=(4, 4), strides=(2, 2), kernel_initializer=glorot_uniform(seed=1), padding='same', activation=LeakyReLU(alpha=.2))(x2)
    x3 = BatchNormalization(epsilon=.001, momentum=.9)(x3)
    x4 = Conv2D(filters=16, kernel_size=(4, 4), strides=(2, 2), kernel_initializer=glorot_uniform(seed=1), padding='same', activation=LeakyReLU(alpha=.2))(x3)
    x4 = BatchNormalization(epsilon=.001, momentum=.9)(x4)
    x5 = Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), kernel_initializer=glorot_uniform(seed=1), padding='same', activation=LeakyReLU(alpha=.2))(x4)
    x5 = BatchNormalization(epsilon=.001, momentum=.9)(x5)
    x6 = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), kernel_initializer=glorot_uniform(seed=1), padding='same', activation=LeakyReLU(alpha=.2))(x5)
    x6 = BatchNormalization(epsilon=.001, momentum=.9)(x6)
    x7 = Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), kernel_initializer=glorot_uniform(seed=1), padding='same', activation=LeakyReLU(alpha=.2))(x6)
    x7 = BatchNormalization(epsilon=.001, momentum=.9)(x7)

    x8 = Conv2D(filters=128, kernel_size=(4, 4), strides=(1, 1), kernel_initializer=glorot_uniform(seed=1), padding='same', activation=LeakyReLU(alpha=.2))(x7)
    x8 = BatchNormalization(epsilon=.001, momentum=.9)(x8)

    x7 = UpSampling2D(size=(2, 2))(x8)
    x7 = Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(1, 1), kernel_initializer=glorot_uniform(seed=1), padding='same', activation='relu')(x7)
    x7 = BatchNormalization(epsilon=.001, momentum=.9)(x7)
    x6 = Concatenate(axis=1)([x7, x6])
    x6 = UpSampling2D(size=(2, 2))(x6)
    x6 = Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(1, 1), kernel_initializer=glorot_uniform(seed=1), padding='same', activation='relu')(x6)
    x6 = BatchNormalization(epsilon=.001, momentum=.9)(x6)
    x5 = Concatenate(axis=1)([x6, x5])
    x5 = UpSampling2D(size=(2, 2))(x5)
    x5 = Conv2DTranspose(filters=32, kernel_size=(4, 4), strides=(1, 1), kernel_initializer=glorot_uniform(seed=1), padding='same', activation='relu')(x5)
    x5 = BatchNormalization(epsilon=.001, momentum=.9)(x5)
    x4 = Concatenate(axis=1)([x5, x4])
    x4 = UpSampling2D(size=(2, 2))(x4)
    x4 = Conv2DTranspose(filters=16, kernel_size=(4, 4), strides=(1, 1), kernel_initializer=glorot_uniform(seed=1), padding='same', activation='relu')(x4)
    x4 = BatchNormalization(epsilon=.001, momentum=.9)(x4)
    x3 = Concatenate(axis=1)([x4, x3])
    x3 = UpSampling2D(size=(2, 2))(x3)
    x3 = Conv2DTranspose(filters=8, kernel_size=(4, 4), strides=(1, 1), kernel_initializer=glorot_uniform(seed=1), padding='same', activation='relu')(x3)
    x3 = BatchNormalization(epsilon=.001, momentum=.9)(x3)
    x2 = Concatenate(axis=1)([x3, x2])
    x2 = UpSampling2D(size=(2, 2))(x2)
    x2 = Conv2DTranspose(filters=4, kernel_size=(4, 4), strides=(1, 1), kernel_initializer=glorot_uniform(seed=1), padding='same', activation='relu')(x2)
    x2 = BatchNormalization(epsilon=.001, momentum=.9)(x2)
    x1 = Concatenate(axis=1)([x2, x1])
    x1 = UpSampling2D(size=(2, 2))(x1)
    x1 = Conv2DTranspose(filters=1, kernel_size=(4, 4), strides=(1, 1), kernel_initializer=glorot_uniform(seed=1), padding='same', activation='tanh')(x1)
    return Model(inputs=img, outputs=x1)

def generator_train_img(real_list_dir, white_list_dir, resize, batch_size):
    batch_real_img = []
    batch_white_img = []
    for _ in range(batch_size):
        random_index = np.random.randint(len(real_list_dir))
        real_img = imresize(imread(real_list_dir[random_index], mode='L'), resize)
        white_img = imresize(imread(white_list_dir[random_index], mode='L'), resize)
        batch_real_img.append(real_img)
        batch_white_img.append(white_img)
    batch_real_img = np.array(batch_real_img) / 127.5 - 1
    batch_real_img = np.expand_dims(batch_real_img, axis=1)
    batch_white_img = np.array(batch_white_img) / 127.5 - 1
    batch_white_img = np.expand_dims(batch_white_img, axis=1)
    return batch_real_img, batch_white_img

def generator_test_img(list_dir, resize):
    output_training_img = [imresize(imread(i, mode='L'), resize) for i in list_dir]
    output_training_img = np.array(output_training_img) / 127.5 - 1
    output_training_img = np.expand_dims(output_training_img, axis=1) # (batch,img_row,img_col) ==> (batch,1,img_row,img_cok)
    return output_training_img

def numpy_to_csv(input_image, image_number, save_csv_name):
    save_image = np.zeros([int(input_image.size / image_number), image_number], dtype=np.float32)
    for image_index in range(image_number):
        save_image[:, image_index] = input_image[image_index, :, :].flatten()
    df = pd.DataFrame(save_image)
    df.index.name = 'index'
    df.columns = [('id' + str(i)) for i in range(image_number)]
    df.to_csv(save_csv_name)

if __name__ == '__main__':
    real_list_dir = glob.glob('./Training/Real/*')
    white_list_dir = glob.glob('./Training/White/*')
    test_data_dir = glob.glob('./Test/White/*')

    batch_size = 128
    all_epoch = 128
    channels, img_row, img_col = 1, 128, 128
    input_shape = (channels, img_row, img_col)
    
    G = Generator(input_shape)
    D = Discriminator(input_shape)
    D.trainable = False
    D.compile(loss='mse', optimizer=Adam(lr=.0001, beta_1=.5), metrics=['accuracy'])
    # D.summary()
    
    img_A = Input(input_shape)
    img_B = Input(input_shape)
    fake = G(img_B)
    AM = Model(inputs=[img_A, img_B], outputs=[D([fake, img_B]), fake])
    AM.compile(loss=['mse', 'mae'], optimizer=RMSprop(lr=.0001, decay=3e-8), loss_weights=[1, 1], metrics=['accuracy'])
    # AM.summary()

    valid = np.ones((batch_size, channels, 8, 8))
    fake = np.zeros((batch_size, channels, 8, 8))

    now = datetime.datetime.now()
    for now_iter in range(1, all_epoch+1):
        ori_img, white_img = generator_train_img(real_list_dir=real_list_dir, white_list_dir=white_list_dir, resize=(img_row, img_col), batch_size=batch_size)
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

    test_img = generator_test_img(list_dir=test_data_dir, resize=(img_row, img_col))
    test_img_ans = G.predict(test_img)
    for index, data in enumerate(((test_img_ans + 1) * 127.5).astype(np.uint8)):
        imwrite('./Test/Ans/{index}.jpg'.format(index=index), np.transpose(data, axes=(1, 2, 0)))
    image_array = (test_img_ans.squeeze(1) + 1) / 2
    numpy_to_csv(input_image=image_array, image_number=10, save_csv_name='predict.csv')

    import matplotlib.pyplot as plt
    plt.gray()
    fig = plt.figure(figsize=(18, 60))
    for i in range(10):
        ori_img, white_img = generator_train_img(real_list_dir=real_list_dir, white_list_dir=white_list_dir, resize=(img_row,img_col), batch_size=1)
        ax = plt.subplot(10, 3, 3*i+1)
        plt.imshow(G.predict(white_img).reshape(img_row, img_col))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(10, 3, 3*i+2)
        plt.imshow(ori_img.reshape(img_row, img_col))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(10, 3, 3*i+3)
        plt.imshow(white_img.reshape(img_row, img_col))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)   
    fig.savefig('training.png', dpi=fig.dpi)
