#!/usr/bin/env python3

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab
import tensorflow as tf

pylab.show()

def nn(input_x, input_y,test_x,test_y) :



	input_y = input_y.values.reshape((-1, 1))
	test_y = test_y.values.reshape((-1, 1))

	# input_x = input_x[0:10]
	# input_y = input_y[0:10]
	# test_x = test_x[0:10]
	# test_y = test_y[0:10]

	# Training parameters
	learning_rate = 0.03
	num_epochs = 200000
	batch_size = 250
	display_epoch = 100

	# Network Parameters
	n_hidden_1 = int(len(input_x.columns)/0.5) # 1st layer number of neurons
	n_hidden_2 = int(len(input_x.columns)/0.5) # 2nd layer number of neurons
	n_hidden_3 = int(len(input_x.columns)/2) # 3rd layer number of neurons
	n_hidden_4 = int(len(input_x.columns)/4) # 4th layer number of neurons

	num_input = len(input_x.columns)  #Features
	num_classes = 1 #Classes (binary)

	X = tf.placeholder("float", [None, num_input])
	Y = tf.placeholder("float", [None, num_classes])
	keep_prob = tf.placeholder("float") #Dropout

	weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1],stddev=1/np.sqrt(n_hidden_1))),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2],stddev=1/np.sqrt(n_hidden_2))),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3],stddev=1/np.sqrt(n_hidden_3))),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4],stddev=1/np.sqrt(n_hidden_4))),
    'out': tf.Variable(tf.random_normal([n_hidden_4, num_classes],stddev=1/np.sqrt(num_classes)))
	}
	biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])), 
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'out': tf.Variable(tf.random_normal([num_classes]))
	}

	def neural_net(x,keep_prob=0):

	    # Hidden fully connected layer with dropout
	    layer_1 = tf.nn.leaky_relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
	    layer_1 = tf.nn.dropout(layer_1, keep_prob)
	    # Hidden fully connected layer with dropout
	    layer_2 = tf.nn.leaky_relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
	    layer_2 = tf.nn.dropout(layer_2, keep_prob)

	    # Hidden fully connected layer (smaller)
	    layer_3 = tf.nn.leaky_relu(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']))
	   
	    # Hidden fully connected layer (smaller)
	    layer_4 = tf.nn.leaky_relu(tf.add(tf.matmul(layer_3, weights['h4']), biases['b4']))

	    # ouput fully connected layer with sigmoid for probability
	    out_layer = tf.sigmoid(tf.add(tf.matmul(layer_4, weights['out']), biases['out']))


	    return out_layer

	# Construct model
	logits = neural_net(X,keep_prob)

	# Define loss and optimizer
	loss_op = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
	    logits=logits, multi_class_labels=Y))
	optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss_op)

	#Check accuracy (tf.equal, 1 if equal, 0 if not)
	correct_pred = tf.equal(tf.round(logits), tf.round(Y))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	_, recall = tf.metrics.recall(labels=Y,predictions=tf.round(logits))
	_, precision = tf.metrics.precision(labels=Y,predictions=tf.round(logits))


	init_g = tf.global_variables_initializer()
	init_l = tf.local_variables_initializer()

	saver = tf.train.Saver()
	itrr = [] #Training epoch
	train_loss = []
	test_loss = []
	train_acc = []
	test_acc = []
	train_precision = []
	train_recall = []

	with tf.Session() as sess:

	    # Run the initializer
	    sess.run(init_g)
	    sess.run(init_l)
	    saver.restore(sess, "./model/my_test_model.ckpt")


	    n=0
	    for epoch in range(1, num_epochs+1):
	    	sess.run(train_op, feed_dict={X: input_x, Y: input_y,keep_prob: 0.9})
	    	if (n == int(num_epochs/40)):
	    	# if (n == 0):
	    		n = 0
	    		itrr.append(epoch)
	    		test_loss.append(sess.run(loss_op, feed_dict={keep_prob:1.0, X: test_x, Y: test_y}))
	    		train_loss.append(sess.run(loss_op, feed_dict={keep_prob:1.0, X: input_x, Y: input_y}))
	    		test_acc.append(sess.run(accuracy, feed_dict={keep_prob:1.0, X: test_x, Y: test_y}))
	    		train_acc.append(sess.run(accuracy, feed_dict={keep_prob:1.0, X: input_x, Y: input_y}))
	    		train_precision.append(sess.run(precision, feed_dict={keep_prob:1.0, X: input_x, Y: input_y}))
	    		train_recall.append(sess.run(recall, feed_dict={keep_prob:1.0, X: input_x, Y: input_y}))
	    		print("X:")
	    		print (sess.run(logits,feed_dict={keep_prob:1.0, X: input_x[0:20]}))
	    		print("Y:")
	    		print(input_y[0:20])

	    		# print("Train loss: ", sess.run(loss_op, feed_dict={keep_prob:1.0, X: input_x, Y: input_y}))
	    	n=n+1


	
	    print("Optimization Finished!")
	    print("Testing loss: ", sess.run(loss_op, feed_dict={keep_prob:1.0, X: test_x, Y: test_y}))
	    print("Train loss: ", sess.run(loss_op, feed_dict={keep_prob:1.0, X: input_x, Y: input_y}))
	    print("train Accuracy: ", sess.run(accuracy, feed_dict={keep_prob:1.0, X: input_x, Y: input_y}))
	    print("test Accuracy: ", sess.run(accuracy, feed_dict={keep_prob:1.0, X: test_x, Y: test_y}))
	    print("precision: ", sess.run(precision, feed_dict={keep_prob:1.0, X: input_x, Y: input_y}))
	    print("recall: ", sess.run(recall, feed_dict={keep_prob:1.0, X: input_x, Y: input_y}))

	    saver.save(sess, './model/my_test_model.ckpt')

	print("ittr: " + str(itrr) + "  test_loss: " + str(test_loss) )
	plt.plot(itrr,train_loss)
	plt.plot(itrr,test_loss)
	# plt.yscale('log')
	plt.title("train_test_loss")
	plt.savefig( './train_test_loss.pdf')
	plt.clf()

	plt.plot(itrr,train_acc)
	plt.plot(itrr,test_acc)
	# plt.yscale('log')
	plt.title("train_test_acc")
	plt.savefig( './train_test_acc.pdf')
	plt.clf()

	plt.plot(itrr,train_precision)
	plt.plot(itrr,train_recall)
	# plt.yscale('log')
	plt.title("train_test_acc")
	plt.savefig( './train_test_acc.pdf')
	plt.clf()


