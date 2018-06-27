import numpy as np
import tensorflow as tf

def add_layer(name, inputs, input_tensors, output_tensors, initializer):
	weight = tf.get_variable(name=name + '_weight', shape=[input_tensors, output_tensors], dtype=tf.float32, initializer=initializer)
	bias = tf.get_variable(name=name + '_bias', shape=[output_tensors], dtype=tf.float32, initializer=initializer)
	layer = tf.add(tf.matmul(inputs, weight), bias)
	return layer

if __name__ == '__main__':
	ans = np.array([list(map(int, open("answer.txt", "r").read().split(' ')))])
	train = np.array([list(map(int, i.split(' '))) for i in open("data.txt", "r").read().strip('\n').split('\n')])
	ans = np.transpose(ans)

	train_input = tf.placeholder(tf.float32, shape=[None, 32])
	ans_input = tf.placeholder(tf.float32, shape=[None, 1])
	initializer = tf.truncated_normal_initializer(mean=.0, stddev=.1, seed=tf.set_random_seed(1))

	layer_1 = add_layer(name='layer_1', inputs=train_input, input_tensors=32, output_tensors=2, initializer=initializer)
	layer_2 = add_layer(name='layer_2', inputs=layer_1, input_tensors=2, output_tensors=1, initializer=initializer)

	loss = tf.losses.mean_squared_error(ans, layer_2)
	optimizer = tf.train.AdagradOptimizer(.103).minimize(loss)

	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		for _ in range(600):
			sess.run(optimizer, feed_dict={train_input: train, ans_input: ans})
			loss_scalar = sess.run(loss, feed_dict={train_input: train, ans_input: ans})
			# tf.summary.scalar('loss_scalar', loss_scalar)
			# tf.summary.histogram('loss_scalar', loss_scalar)
			print('{:.40f}'.format(loss_scalar))
		# tf.summary.merge_all()
		# tf.summary.FileWriter('./graph', sess.graph)
			
		
