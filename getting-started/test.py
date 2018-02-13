import tensorflow as tf 

a = tf.constant(3.0, dtype=tf.float32)
s = tf.summary.scalar("Training Loss", a)

sess = tf.Session()

# initiate all variable
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(s))

