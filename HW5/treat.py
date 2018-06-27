import glob
import numpy as np
import pandas as pd
from imageio import imwrite
from scipy.misc.pilutil import imread, imresize

def generator_test_img(list_dir):
	output_training_img = [imresize(imread(i, mode='L'), (128, 128)) for i in list_dir]
	output_training_img = np.array(output_training_img) / 255.0
	return output_training_img

def numpy_to_csv(input_image, image_number):
	save_image = np.zeros([int(input_image.size / image_number), image_number], dtype=np.float32)
	for image_index in range(image_number):
		save_image[:, image_index] = input_image[image_index, :, :].flatten()
	df = pd.DataFrame(save_image)
	df.index.name = 'index'
	df.columns = [('id' + str(i)) for i in range(image_number)]
	return df

def get_dataframe(root):
	list_dir = sorted(glob.glob(root))
	image_array = generator_test_img(list_dir)
	# image_array = np.transpose(image_array, axes=(0, 2, 1))
	return numpy_to_csv(input_image=image_array, image_number=10)

if __name__ == '__main__':
	white_df = get_dataframe('./Test/White/*')
	test_df = pd.read_csv('./GPU/Ans.026/predict.csv')
	for i in range(10286, 14162, 128):
		for j in range(i, i+40):
			for k in range(10):
				if white_df.iloc[j][k] == 1.0:
					white_df.iloc[j][k] = test_df.iloc[j][k+1]

	white_df.to_csv("predict.csv")
