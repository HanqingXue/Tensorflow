#coding=utf-8

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import time

start = time.time()

cluster = tf.train.ClusterSpec({"local": ["172.20.182.232:2222",
                                          "172.20.182.232:2223",
                                          "172.20.182.232:2224",
                                          "172.20.182.232:2225",
                                          "172.20.182.232:2226",
                                          "172.20.182.232:2227",
                                          "172.20.182.232:2228"]})

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

with tf.device("/job:local/task:0"):
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    x_image = tf.reshape(x, [-1, 28, 28, 1])


#Frist layer
with tf.device("/job:local/task:1"):
    W_conv1 = weigh_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

with tf.device("/job:local/task:2"):
    W_conv2 = weigh_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
with tf.device("/job:local/task:3"):
    W_fc1 = weigh_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
with tf.device("/job:local/task:4"):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.device("/job:local/task:5"):
    W_fc2 = weigh_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    
with tf.device("/job:local/task:6"):
    # 优化函数
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    # 优化的策略
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
    correct_predection = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predection, tf.float32))
    init = tf.global_variables_initializer()

with tf.Session("grpc://localhost:2228", config=tf.ConfigProto(log_device_placement=True)) as sess:
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    #sess.run(init)
    steps = 1000
    for i in range(1000):
        batch = mnist.train.next_batch(50)
    
        if i % 200 == 0:
            #train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1})
            #print ('Current Accuracy:')
            #print  (train_accuracy)
            pass
            
        sess.run(train_step, feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob:0.5})
    #print ("Test accuracy %g " % accuracy.eval(feed_dict={ x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

end = time.time()
print ('Run time{0}'.format(end - start))