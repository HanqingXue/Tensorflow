
'''
import tensorflow as tf
c = tf.constant("Hello, distributed TensorFlow!")
server = tf.train.Server.create_local_server()
print server.target
print '************'
server_one = tf.train.Server.create_local_server()
print server_one.target
sess = tf.Session(server.target)  # Create a session on the server.
sess.run(c)
'''

'''
import tensorflow as tf

x = tf.constant(2)
y1 = x + 300
y2 = x - 66
y = y1 + y2

with tf.Session() as sess:
    result = sess.run(y)
    print(result)
'''

import tensorflow as tf

cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
    
