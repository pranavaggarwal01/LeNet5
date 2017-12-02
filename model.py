import tensorflow as tf
import numpy as np 
from tensorflow.contrib.layers import flatten

flattened_layer_size = 400
FC_layer1_neurons = 120
FC_layer2_neurons = 84
classes_neurons = 10
mu = 0
sigma = 0.001
# image_height = 32
# image_width = 32
#initializing the weights and biases for each layer

#flattened_layer.get_shape().as_list()[1]
FC_layer1 = {'weights' : tf.Variable(tf.random_normal([flattened_layer_size,FC_layer1_neurons])), 
'bais' : tf.Variable(tf.random_normal([FC_layer1_neurons]))}

FC_layer2 = {'weights' : tf.Variable(tf.random_normal([FC_layer1_neurons,FC_layer2_neurons])), 
'bais' : tf.Variable(tf.random_normal([FC_layer2_neurons]))}

output = {'weights' : tf.Variable(tf.random_normal([FC_layer2_neurons,classes_neurons])), 
'bais' : tf.Variable(tf.random_normal([classes_neurons]))}

conv1_w = tf.Variable(tf.truncated_normal(shape = [5,5,1,6],mean = mu, stddev = sigma))
conv1_b = tf.Variable(tf.zeros(6))

conv2_w = tf.Variable(tf.truncated_normal(shape = [5,5,6,16],mean = mu, stddev = sigma))
conv2_b = tf.Variable(tf.zeros(16))

# conv3_w = tf.Variable(tf.truncated_normal(shape = [3,3,10,16],mean = mu, stddev = sigma))
# conv3_b = tf.Variable(tf.zeros(16))

def Model(data):

	
	conv1 = tf.nn.conv2d(data,conv1_w, strides = [1,1,1,1], padding = 'SAME') + conv1_b 
	print('After Conv1 shape:')
	print(conv1.get_shape().as_list())
	# Activation.
	conv1 = tf.nn.relu(conv1)
	# Pooling
	pool_1 = tf.nn.max_pool(conv1,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')

	#Second conv layer	
	conv2 = tf.nn.conv2d(pool_1,conv2_w, strides = [1,1,1,1], padding = 'VALID') + conv2_b 
	print('After Conv2 shape:')
	print(conv2.get_shape().as_list())
	# Activation.
	conv2 = tf.nn.relu(conv2)
	# Pooling
	pool_2 = tf.nn.max_pool(conv2,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')

	# #third conv layer
	# conv3 = tf.nn.conv2d(pool_2,conv3_w, strides = [1,1,1,1], padding = 'VALID') + conv3_b 
	# # Activation.
	# conv3 = tf.nn.relu(conv3)
	# # Pooling
	# pool_3 = tf.nn.max_pool(conv3,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
	# print(pool_3.get_shape())

	flattened_layer = flatten(pool_2)
	print(flattened_layer.get_shape().as_list())

	l1 = tf.add(tf.matmul(flattened_layer,FC_layer1['weights']),FC_layer1['bais'])

	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1,FC_layer2['weights']),FC_layer2['bais'])

	l2 = tf.nn.relu(l2)

	last = tf.add(tf.matmul(l2,output['weights']),output['bais'])

	return last