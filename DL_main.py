import numpy as np 
import glob 
import Deep_learning_operations as DLO
import tensorflow as tf
import sys
from tensorflow.examples.tutorials.mnist import input_data

image_height = 28
image_width = 28
x = tf.placeholder('float', [None, image_height, image_width, 1])
# number_of_train_subjects = 20

if __name__ == "__main__":

	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	#training
	DLO.training(x, mnist)# removed the NaN and 0 xyz's

	#testing
	DLO.testing(x, mnist)


	