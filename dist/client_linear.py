#coding=utf-8
import tensorflow as tf


cluster = tf.train.ClusterSpec({"local": ["172.20.182.232:2222",
                                          "172.20.182.232:2223",
                                          "172.20.182.232:2224",
                                          "172.20.182.232:2225",
                                          "172.20.182.232:2226"]})

x = tf.placeholder(tf.float32)
W = tf.Variable(tf.zeros([1]))
b = tf.Variable(tf.zeros([1]))
y_ = tf.placeholder(tf.float32)

with tf.device("/job:local/task:1"):
    y = W * x + b

with tf.device("/job:local/task:0"):
    lost = tf.reduce_mean(tf.square(y_-y))
    
with tf.device("/job:local/task:2"):
    optimizer = tf.train.GradientDescentOptimizer(0.0000001)
    train_step = optimizer.minimize(lost)
    
with tf.Session("grpc://localhost:2222", config=tf.ConfigProto(log_device_placement=True)) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    steps = 1000
    for i in range(steps):
        xs = [i]
        ys = [3 * i]
        feed = { x: xs, y_: ys }
        sess.run(train_step, feed_dict=feed)
        if i % 100 == 0 :
            print("After %d iteration:" % i)
            print("W: %f" % sess.run(W))
            print("b: %f" % sess.run(b))
            print("lost: %f" % sess.run(lost, feed_dict=feed))
