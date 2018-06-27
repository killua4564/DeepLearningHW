import tensorflow as tf
import numpy as np

'''
sess = tf.InteractiveSession()

with tf.name_scope("name_scope"):
	initializer = tf.constant_initializer(value=3)
	value = tf.get_variable(name="get_value", shape=[], dtype=tf.float32, initializer=initializer)
	a = tf.Variable(name="a", initial_value=2.0, dtype=tf.float32)
	with tf.name_scope("owo"):
		b = tf.Variable(name="b", initial_value=3.1, dtype=tf.float32)
		c = tf.Variable(name="c", initial_value=4.2, dtype=tf.float32)
		with tf.Session() as sess:
			tf.global_variables_initializer().run()
			print(a.name)
			print(b.name)
			print(c.name)
			print(sess.run(a))
			print(sess.run(b))
			print(sess.run(c))

		# writer = tf.summary.FileWriter('./graphs', sess.graph)
'''

# multiply
# [[1 2]    [[1 3]     [[1 6]
#  [3 4]] .  [2 1]]  =  [6 4]]

# matmul
# [[1 0]    [[1 3]     [[1 3]
#  [0 1]] .  [2 1]]  =  [2 1]]