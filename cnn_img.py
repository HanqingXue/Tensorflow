from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf 
from sampleutil import *
import random
from feature import *
import numpy as np

sess = tf.InteractiveSession()

def weigh_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

with tf.name_scope('input_layer'): 
	x = tf.placeholder(tf.float32, [None, 784], name = "x_input")
	y_ = tf.placeholder(tf.float32, [None, 10], name = "y_input")
	x_image = tf.reshape(x, [-1, 28, 28, 1] , name = "Image")


W_conv1 = weigh_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weigh_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weigh_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weigh_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)



cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-1).minimize(cross_entropy)
correct_predection = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predection, tf.float32))

train, labels = dataUtil('train.txt')
hot_code = {
	0: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	1: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
}

train_set = []
label_set = []
for item in train:
	item = item + [0] * (784 - len(item))
	train_set.append(np.array(item))

for label in labels:
	label_set.append(np.array(label))



sess = tf.InteractiveSession()	

tf.global_variables_initializer().run()	


train_ids , test_ids, samples = split_samples(0.4)

for i in range(10000):
	batch = [[],[]]
	batch_ids = random.sample(samples, 200)
	for ids in batch_ids:
		batch[0].append(samples[ids]['data'])
		batch[1].append(samples[ids]['label'])
	
	if i % 500 == 0:
		train_accuracy = accuracy.eval(feed_dict = { x: batch[0], y_:batch[1], keep_prob: 1})
		print train_accuracy
		if train_accuracy == 1.0:
			break

	train_step.run(feed_dict = {x: batch[0], y_: batch[1], keep_prob:0.5})
	

test = [[],[]]
for ids in test_ids:
	test[0].append(samples[ids]['data'])
	test[1].append(samples[ids]['label'])

train = [[], []]
for ids in train_ids:
	train[0].append(samples[ids]['data'])
	train[1].append(samples[ids]['label'])


print "Test accuracy %g " % accuracy.eval(feed_dict={
	x: train[0], y_: train[1], keep_prob: 1.0})
