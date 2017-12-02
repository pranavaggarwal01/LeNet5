import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import cv2 
import os 
import random
import model as model_arch


learningRate = 0.0001
batch_size = 10
classes_neurons = 10
no_epochs = 30

y = tf.placeholder('float', [None, classes_neurons])

def un_one_hot(labels):
	finalLabel = []
	for i in labels:
		finalLabel.append(list(i).index(1))
	# print (finalLabel)
	return finalLabel

def training(x, mnist):
	prct = model_arch.Model(x)
	saver = tf.train.Saver()
	#defining the cost function. Feel free to experiment by changing weighted to sigmoid or softmax
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prct,labels = y))
	optimise = tf.train.GradientDescentOptimizer(learning_rate = learningRate).minimize(cost)
	# np.random.shuffle(dataset_train)
	losses = []
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for i in range(no_epochs):
			#need to shuffle the dataset
			# np.random.shuffle(dataset_train)
			epoch_loss = 0
			#implementing batchwise training. Please mention the batch_size you need above
			for j in range(int(mnist.train.num_examples/batch_size)):

				batch, gt_batch_binary = mnist.train.next_batch(batch_size)
				# print (batch)
				batch = np.reshape(batch, [batch_size, 28, 28, 1])
				# np.savetxt('sample.txt',batch[0], fmt='%.1f')
				_,c = sess.run([optimise,cost], feed_dict = {x:batch, y:gt_batch_binary})
				epoch_loss += c
				print( 'Epoch: ',i , 'Batch Training Loss: ', c)				

			print('Epoch loss: ' + str(epoch_loss))
			losses.append(epoch_loss)


		# we need to save the model
	
		if (os.path.exists('models/') != True):
			os.mkdir("models/")
		save_path = saver.save(sess, "models/model") 		

		plt.plot(range(no_epochs), losses)
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		# plt.axis([ 0, no_epochs, 0, 5000])
		plt.show()
		# plt.savefig(('train_loss_LearningRate=',learningRate,'btach_size:',batch_size))
		# plt.plot(range(no_epochs), acc_test)
		# plt.xlabel('Epochs')
		# plt.ylabel('Test Accuracy')
		# plt.show()



def testing(x,mnist):
	prct = model_arch.Model(x)
	# saver = tf.train.Saver()
	with tf.Session() as sess:
		##restoring the trained model
		saver = tf.train.import_meta_graph("models/model.meta")
		saver.restore(sess,tf.train.latest_checkpoint("models/"))
		##finding the predicted values
		test_images = np.reshape(mnist.test.images, [mnist.test.num_examples, 28, 28, 1])

		correct = tf.equal(tf.argmax(prct,1), tf.argmax(y,1))
		acc = tf.reduce_mean(tf.cast(correct,'float'))
		labels = mnist.test.labels
		test_accuracy = acc.eval({x:test_images, y:labels})
		print ('Testing Accuracy: ' , test_accuracy)

		predictions = sess.run(prct, feed_dict={x:test_images})
		print ('Confusion Matrix test: ')
		predicted_test = tf.argmax(predictions,1)
		conf = tf.confusion_matrix(labels=un_one_hot(mnist.test.labels),predictions=predicted_test, num_classes=classes_neurons)
		print(conf.eval())

		# # if you need to print the results on the frames the flaf need to be set as 1
		# if video_flag:
		# 	if (os.path.exists("Distraction_results/") != True):
		# 		os.mkdir("Distraction_results/")
		# 	for frame in range(len(dataset_test[:,-1])):
		# 		img = cv2.imread('dataset_frames/' + frame_names[frame].split('/')[0] + '/img/' + frame_names[frame].split('/')[1])
		# 		if (dataset_test[frame,-1] == 0):
		# 			img = cv2.putText(img,'GroundTruth: Not Distracted', (10,20), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255))
		# 		else:
		# 			img = cv2.putText(img,'GroundTruth: Distracted', (10,20), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255))

		# 		if (predicted_test.eval()[frame] == 0):
		# 			img = cv2.putText(img,'Predicted : Not Distracted', (10,100), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255))
		# 		else:
		# 			img = cv2.putText(img,'Predicted : Distracted', (10,100), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255))
		# 		if (os.path.exists("Distraction_results/" + frame_names[frame].split('/')[0]) != True):
		# 			os.mkdir("Distraction_results/" + frame_names[frame].split('/')[0])
		# 		cv2.imwrite('Distraction_results/' + frame_names[frame], img)
	


