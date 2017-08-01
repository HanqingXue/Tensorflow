import numpy as np
import tensorflow as tf

cluster = tf.train.ClusterSpec({"local": ["172.20.182.232:2222", "172.20.98.222:2222"]})

x = tf.placeholder(tf.float32, 100)

data = []
for  i in xrange(0, 4000000):
	data.append(i)


a = tf.constant(data, shape=[2000, 2000])  
b = tf.constant(data, shape=[2000, 2000])
d = tf.add(a, b)

with tf.device("/job:local/task:1"):
    first_batch = tf.slice(x, [0], [50])
    mean1 = tf.reduce_mean(first_batch)
    c = tf.matmul(a, b)
    print 'ok1'

with tf.device("/job:local/task:0") as task:
    second_batch = tf.slice(x, [50], [-1])
    mean2 = tf.reduce_mean(second_batch)
    mean = (mean1 + mean2) / 2
    print 'ok2'
    d += c

with tf.Session("grpc://localhost:2222", config=tf.ConfigProto(log_device_placement=True)) as sess:
    result = sess.run(d)
    print(result)