from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf 
from sampleutil import *
import random
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

with tf.name_scope('hidden_layer_one'):
	with tf.name_scope('W_conv1'):
		W_conv1 = weigh_variable([5, 5, 1, 32])
		tf.summary.histogram('hidden_layer_one/W_conv1', W_conv1)
	
	with tf.name_scope('b_conv1'):
		b_conv1 = bias_variable([32])
		tf.summary.histogram('hidden_layer_one/b_conv1', b_conv1)

	with tf.name_scope('h_conv1'):
		h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
		tf.summary.histogram('hidden_layer_one/h_conv1', h_conv1)

	with tf.name_scope('h_pool1'):
		h_pool1 = max_pool_2x2(h_conv1)
		tf.summary.histogram('hidden_layer_one/h_conv1', h_pool1)

with tf.name_scope('hidden_layer_two'):
	with tf.name_scope('W_conv2'):
		W_conv2 = weigh_variable([5, 5, 32, 64])
		tf.summary.histogram('hidden_layer_two/W_conv2', W_conv2)
	
	with tf.name_scope('b_conv2'):
		b_conv2 = bias_variable([64])
		tf.summary.histogram('hidden_layer_two/b_conv2', b_conv2)

	with tf.name_scope('h_conv2'):
		h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
		tf.summary.histogram('hidden_layer_two/h_conv2', h_conv2)
	
	with tf.name_scope('h_pool2'):
		h_pool2 = max_pool_2x2(h_conv2)
		tf.summary.histogram('hidden_layer_two/h_pool2', h_pool2)
	
with tf.name_scope('full_connection_one'):
	with tf.name_scope('W_fc1'):
		W_fc1 = weigh_variable([7 * 7 * 64, 1024])
		tf.summary.histogram('full_connection_one/W_fc1', W_fc1)

	with tf.name_scope('b_fc1'):
		b_fc1 = bias_variable([1024])
		tf.summary.histogram('full_connection_one/b_fc1', b_fc1)

	with tf.name_scope('h_pool2_flat'):
		h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
		tf.summary.histogram('full_connection_one/h_pool2_flat', h_pool2_flat)

	with tf.name_scope('h_fc1'):
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
		tf.summary.histogram('full_connection_one/h_fc1', h_fc1)

with tf.name_scope('dropout'):
	with tf.name_scope('keep_prob'):
		keep_prob = tf.placeholder(tf.float32)
		tf.summary.histogram('dropout/keep_prob', keep_prob)

	with tf.name_scope('h_fc1_drop'):
		tf.summary.histogram('dropout/h_fc1_drop', keep_prob)
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('out_put'):
	with tf.name_scope('W_fc2'):
		W_fc2 = weigh_variable([1024, 10])
		tf.summary.histogram('out_put/W_fc2', W_fc2)
	with tf.name_scope('b_fc2'):
		b_fc2 = bias_variable([10])
		tf.summary.histogram('out_put/b_fc2', b_fc2)

	with tf.name_scope('out_put/y_conv'):
		y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
		tf.summary.histogram('out_put/y_conv', y_conv)


with tf.name_scope('cross_entropy'):
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
	tf.summary.scalar('loss', cross_entropy)
	
with tf.name_scope('train'):
	#train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
	train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(cross_entropy)
correct_predection = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predection, tf.float32))

sess = tf.InteractiveSession()	

tf.global_variables_initializer().run()	

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('logs',sess.graph)

train_ids , test_ids, samples = split_samples(0.8)

for i in range(2000):
	batch = [[],[]]
	batch_ids = random.sample(samples, 100)
	for ids in batch_ids:
		batch[0].append(samples[ids]['data'])
		batch[1].append(samples[ids]['label'])
	
	if i % 500 == 0:
		train_accuracy = accuracy.eval(feed_dict = { x: batch[0], y_:batch[1], keep_prob: 1})
		print train_accuracy
		if train_accuracy > 0.22:
			break

	train_step.run(feed_dict = {x: batch[0], y_: batch[1], keep_prob:0.6})
	

test = [[],[]]
for ids in test_ids:
	test[0].append(samples[ids]['data'])
	test[1].append(samples[ids]['label'])

train = [[], []]
for ids in train_ids:
	train[0].append(samples[ids]['data'])
	train[1].append(samples[ids]['label'])

np.save('imgmatrix.npy' ,train[0])

print "Test accuracy %g " % accuracy.eval(feed_dict={
	x: train[0], y_: train[1], keep_prob: 1.0})

print "Test accuracy %g " % accuracy.eval(feed_dict={
	x: test[0], y_: test[1], keep_prob: 1.0})
