# Get task number from command line
import sys
task_number = int(sys.argv[1])

import tensorflow as tf

cluster = tf.train.ClusterSpec({"local": ["172.20.182.232:2222", "172.20.182.232:2223", "172.20.182.232:2224"]})
server = tf.train.Server(cluster, job_name="local", task_index=task_number, config=tf.ConfigProto(log_device_placement=True))

print("Starting server #{}".format(task_number))
server.start()
server.join()