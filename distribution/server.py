import tensorflow as tf


cluster = tf.train.ClusterSpec({"local": ["172.20.182.232:2222", "172.20.98.222:2223"]})

x = tf.constant(2)


with tf.device("/job:local/task:1"):
    y2 = x - 66

with tf.device("/job:local/task:0"):
     y1 = x + 300
     y = y1 + y2

with tf.Session("grpc://172.20.182.232:2222") as sess:
     result = sess.run(y)
     print(result)
