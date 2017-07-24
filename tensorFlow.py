
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn import datasets
from sklearn import cross_validation
from feature import *
# Import data

def singleHiddenNet():
	X , y = dataUtil('train.txt')
	X_train, X_test, y_train, y_test = cross_validation.train_test_split( \
		X, y, test_size=0.2, random_state=0)
#	 Create the model
	inputSize = 21
	outputSize = 2
	x = tf.placeholder(tf.float32, [None, inputSize])
	W = tf.Variable(tf.zeros([inputSize, outputSize]))
	b = tf.Variable(tf.zeros([outputSize]))
	y = tf.matmul(x, W) + b
	y_ = tf.placeholder(tf.float32, [None, outputSize])

# Define loss and optimizer

# The raw formulation of cross-entropy,
#
#   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
#                                 reduction_indices=[1]))
#
# can be numerically unstable.
#
# So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
# outputs of 'y', and then average across the batch.	

	cross_entropy = tf.reduce_mean(
	      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
	train_step = tf.train.GradientDescentOptimizer(0.8).minimize(cross_entropy)	

	sess = tf.InteractiveSession()	

	tf.global_variables_initializer().run()	
	

	# Train
	hot_code = {
		0: [1, 0],
		1: [0, 1],
	}
	start = 0 
	end = 1
	size = 1000
	for _ in range(len(X_train) / size - 1):
		print _
		batch_xs = X_train[(start + _) * size: (end + _) * size]
		batch_ys = y_train[(start + _) * size: (end + _) * size]
		batch_ys = [hot_code[item] for item in batch_ys]
		print batch_ys
		print batch_xs
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})	
    
	# Test trained model
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar('accuracy', accuracy)
	y_test = [hot_code[item] for item in y_test]
	print(sess.run(accuracy, feed_dict={x: X_test,
	                                    y_: y_test}))
	
if __name__ == '__main__':
	
	singleHiddenNet()
