
import tensorflow as tf
from tensorflow.python.client import device_lib

initializer = tf.random_uniform_initializer(-0.1, 0.1)

sess = tf.Session()
print(sess.run(initializer))
