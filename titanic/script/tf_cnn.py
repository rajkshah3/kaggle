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

def nn(input_x, input_y) :


	dataOut_tensor = tf.Variable(input_y.as_matrix(), dtype = tf.int32)
	dataIn_tensor = tf.Variable(input_x.as_matrix(), dtype = tf.float32)

	# Parameters
	learning_rate = 0.1
	num_steps = 500
	batch_size = 250
	display_step = 100

	# Network Parameters
	n_hidden_1 = int(len(input_x.columns)/2) # 1st layer number of neurons
	n_hidden_2 = int(len(input_x.columns)/4) # 2nd layer number of neurons
	num_input = len(input_x.columns) # MNIST data input (img shape: 28*28)
	num_classes = 1 # MNIST total classes (0-9 digits)

	X = tf.placeholder("float", [None, num_input])
	Y = tf.placeholder("float", [None, num_classes])

	filter_1_x = 5
	filter_1_y = 5
	num_filters_1 = 3


	filter_1_x = 10
	filter_1_y = 10
	num_filters_1 = 5

	weights = {
	'w_conv1' : tf.Variable(tf,random_normal([filter_1_x,filter1_y,1,num_filters_1])),
	'w_conv2' : tf.Variable(tf,random_normal([filter_2_x,filter2_y,num_filters_1,num_filters_2])),
    'h1': tf.Variable(tf.random_normal([num_filters_2*num_input/8, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
	}
	biases = {
    'b_conv1': tf.Variable(tf.random_normal([num_filters_1])),
    'b_conv2': tf.Variable(tf.random_normal([num_filters_2])),
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
	}

	def neural_net(x):

		conv_layer_1 = tf.nn.relu(tf.add(conv2d(weights['w_conv1']),biases['b_conv1']))

		pool_layer_1 = max_pool_2x2(conv_layer_1,2)

		conv_layer_2 = tf.nn.relu(tf.add(conv2d(weights['w_conv2']),biases['b_conv2']))

		pool_layer_2 = max_pool_2x2(conv_layer_2,2)

		pool_layer_2_flat = tf.reshape(pool_layer_2,[num_filters_2*num_input/8])
	    # Hidden fully connected layer with 256 neurons
	    layer_1 = tf.sigmoid(tf.add(tf.matmul(pool_layer_2_flat, weights['h1']), biases['b1']))
	    # Hidden fully connected layer with 256 neurons
	    layer_2 = tf.sigmoid(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
	    # Output fully connected layer with a neuron for each class
	    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
	    
	    return out_layer

	# Construct model
	logits = neural_net(X)

	# Define loss and optimizer
	loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
	    logits=logits, labels=Y))
	optimizer = tf.train.AdaDeltaOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss_op)

	# Evaluate model (with test logits, for dropout to be disabled)
	#logits is output layer of NN, argmax finds index of most prob
	correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	init = tf.global_variables_initializer()


	with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        
    	batch_x_tensor, batch_y_tensor = tf.train.shuffle_batch( [dataIn_tensor, dataOut_tensor], batch_size = batch_size, enqueue_many=True, capacity = 50000, min_after_dequeue = 10000)


        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      Y: mnist.test.labels}))


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_nxn(x,n):
  return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='SAME')

