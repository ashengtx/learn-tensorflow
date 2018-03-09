#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
"""
ref: http://blog.csdn.net/fendouaini/article/details/75209180
一个有解读的博文：http://blog.csdn.net/mebiuw/article/details/52705731
"""
  
import tensorflow as tf  
from tensorflow.examples.tutorials.mnist import input_data  
from tensorflow.contrib import rnn  
from rnn_cell_impl import BasicLSTMCell
import numpy as np
  
data_path = '/home/sheng/data/tensorflow/models/image/mnist/MNIST_data'
mnist=input_data.read_data_sets(data_path,one_hot=True)  
  
training_rate=0.001  
training_iters=100000  
batch_size=128  
display_step=10  
  
n_input=28  
n_steps=28  
n_hidden=256  
n_classes=10  
  
x=tf.placeholder("float",[None,n_steps,n_input])  
y=tf.placeholder("float",[None,n_classes])  
  
weights={'out':tf.Variable(tf.random_normal([n_hidden,n_classes]))}  
biases={'out':tf.Variable(tf.random_normal([n_classes]))}  
  
def RNN(x,weights,biases):  
   print(x.shape) # (?, 28, 28)
   x=tf.unstack(x,n_steps,1)  
   # x 这里unstack之后变成28个(?, 28)的tf.Tensor组成的list
   lstm_cell=rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)  
   #lstm_cell=BasicLSTMCell(n_hidden,forget_bias=1.0)  
   outputs,states=rnn.static_rnn(lstm_cell,x,dtype=tf.float32)  
   print(outputs[-1].shape) # (?, 128)
   print(states)
   return tf.matmul(outputs[-1],weights['out'])+biases['out']  
   # (?, 128) * (128, 10) = (?, 10)
  
pred=RNN(x,weights,biases)  
print("pred shape: ", pred.shape)
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
      print(batch_x.shape)
      quit()

      batch_x=batch_x.reshape(batch_size,n_steps,n_input)  
      sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})  
      if step%display_step==0:  
         acc=sess.run(accuaracy,feed_dict={x:batch_x,y:batch_y})  
         loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})  
         print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
               "{:.6f}".format(loss) + ", Training Accuracy= " + \
               "{:.5f}".format(acc))  
      step+=1  
