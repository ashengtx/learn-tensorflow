#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
  
import tensorflow as tf  
from tensorflow.examples.tutorials.mnist import input_data  
from tensorflow.contrib import rnn  
  
data_path = '/home/sheng/data/tensorflow/models/image/mnist/MNIST_data'
mnist=input_data.read_data_sets(data_path,one_hot=True)  
  
training_rate=0.001  
training_iters=100000  
batch_size=128  
display_step=10  
  
n_input=28  
n_steps=28  
n_hidden=128  
n_classes=10  
  
x=tf.placeholder("float",[None,n_steps,n_input])  
y=tf.placeholder("float",[None,n_classes])  
  
weights={'out':tf.Variable(tf.random_normal([n_hidden,n_classes]))}  
biases={'out':tf.Variable(tf.random_normal([n_classes]))}  
  
def RNN(x,weights,biases):  
   x=tf.unstack(x,n_steps,1)  
   lstm_cell=rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)  
   outputs,states=rnn.static_rnn(lstm_cell,x,dtype=tf.float32)  
   return tf.matmul(outputs[-1],weights['out'])+biases['out']  
  
pred=RNN(x,weights,biases)  
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))  
optimizer=tf.train.AdamOptimizer(learning_rate=training_rate).minimize(cost)  
  
correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))  
accuaracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))  
  
init=tf.global_variables_initializer()  
  
with tf.Session() as sess:  
   sess.run(init)  
   step=1  
   while step*batch_size<training_iters:  
      batch_x,batch_y=mnist.train.next_batch(batch_size)  
      batch_x=batch_x.reshape(batch_size,n_steps,n_input)  
      sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})  
      if step%display_step==0:  
         acc=sess.run(accuaracy,feed_dict={x:batch_x,y:batch_y})  
         loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})  
         print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
               "{:.6f}".format(loss) + ", Training Accuracy= " + \
               "{:.5f}".format(acc))  
      step+=1  
