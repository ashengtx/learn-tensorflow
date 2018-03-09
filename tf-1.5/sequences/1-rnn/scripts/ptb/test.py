import tensorflow as tf

import reader

class Test(object):

   def __init__(self):

      embedding = tf.get_variable(
          "embedding", [10, 20], dtype=data_type())
      

with tf.name_scope("Train"):
   with tf.variable_scope("Model", reuse=None, initializer=initializer):
      m = PTBModel(is_training=True)

with tf.name_scope("Valid"):
   with tf.variable_scope("Model", reuse=True, initializer=initializer):
      mvalid = PTBModel(is_training=False)
  
