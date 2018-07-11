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


	dataOut_tensor = tf.data.Dataset(input_y.as_matrix(), dtype = tf.int32)
	dataIn_tensor = tf.data.Dataset(input_x.as_matrix(), dtype = tf.float32)

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

	weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
	}
	biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
	}

	def neural_net(x):

	    # Hidden fully connected layer with 256 neurons
	    layer_1 = tf.sigmoid(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
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
	optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss_op)

	# Evaluate model (with test logits, for dropout to be disabled)
	#logits is output layer of NN, argmax finds index of most prob
	correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	init = tf.global_variables_initializer()


	with tf.Session() as sess:

	    # Run the initializer
	    sess.run(init)
	    
	    print(dataIn_tensor)
	    batch_x_tensor, batch_y_tensor = tf.train.batch( [dataIn_tensor, dataOut_tensor], batch_size = batch_size, enqueue_many=True, capacity = 50000)

	    print(batch_x_tensor)
	    print(batch_y_tensor)

	    # batched_x = dataIn_tensor.batch(batch_size)
	    # batched_y = dataOut_tensor.batch(batch_size)
	    sess.run(train_op, feed_dict={X: batch_x_tensor, Y: batch_y_tensor})

	    iterator_x = batch_x_tensor.make_one_shot_iterator()
	    iterator_y = batch_y_tensor.make_one_shot_iterator()


	    for step in range(1, num_steps+1):
	        batch_x = iterator_x.get_next()
	        batch_y = iterator_y.get_next()
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


    # Calculate accuracy for test examples

	test_data_y = tf.Variable(input_y.as_matrix(), dtype = tf.int32)
	test_data_x = tf.Variable(input_x.as_matrix(), dtype = tf.float32)

	print("Testing Accuracy: ", sess.run(accuracy, feed_dict={X: test_data_x, Y: test_data_y}))



