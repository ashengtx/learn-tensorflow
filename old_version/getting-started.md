# [Getting Started With TensorFlow](https://www.tensorflow.org/get_started/get_started#next_steps)

TensorFlow provides multiple APIs. The lowest level API - TensorFlow Core - provides you with complete programming control. 

The higher level APIs are built on top of TensorFlow Core. These higher level APIs are typically easier to learn and use than TensorFlow Core. 

可见，掌握Tensorflow Core是重点啊！

**Tensors**

```
3 # a rank 0 tensor; a scalar with shape []
[1., 2., 3.] # a rank 1 tensor; a vector with shape [3]
[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]
```

## Tensorflow Core


### Importing Tensorflow

```python
import tensorflow as tf
```

### The Computational Graph

Tersorflow Core programs 就干两件事：

- Building the computational graph.
- Running the computational graph.

What is the computational graph? A computational graph is a series of TensorFlow operations arranged into a graph of nodes. 就是将Tensorflow的操作编排成许多结点组成的有向图的形式

```python
# Building the computational graph.
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)
```

output
```
Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)
```

建立好计算图之后，直接print出来的还是结点，因为计算图还没有运行。

```python
# Running the computational graph.
sess = tf.Session()
print(sess.run([node1, node2]))
```

output:
```
[3.0, 4.0]
```

这样才能输出计算的结果。

**placeholders**

先挖个坑，等运行计算图的时候再把相应的数据丢进去。相当于函数的变量。

```py
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))
```

output:
```
7.5
[ 3.  7.]
```

**Variables**

tf.Variable和tf.constant的区别：constant的值不可变

```py
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b
```

To initialize all the variables in a TensorFlow program, you must explicitly call a special operation as follows:
```py
init = tf.global_variables_initializer()
sess.run(init)
```

调用sess.run之后，所有变量才被初始化。

```py
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))
```

output
```
[ 0.          0.30000001  0.60000002  0.90000004]
```

为了评估模型的好坏，我们需要一个y placeholder来提供desired values，以及一个loss function。

```py
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})) # 23.66
```

这个loss有点大，我们先手动调整一下参数值

```py
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})) # 0.0
```

现在loss是0了。而机器学习要做的事情就是find the correct model parameters automatically，使loss尽量小。

### TensorBoard

如何用Tensorboard可视化计算图？有待学习

参考 https://www.jianshu.com/p/bce3e572bf47

## tf.train API

train linear model
```py
import tensorflow as tf

# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W*x + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x: x_train, y: y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
```

output
```
W: [-0.9999969] b: [ 0.99999082] loss: 5.69997e-11
```

## tf.estimator

tf.estimator is a high-level TensorFlow library. 简化了代码的写法。

主要干三件事：

- running training loops
- running evaluation loops
- managing data sets

tf.estimator定义了许多常见模型，比如，LinearRegressor 。


```py
# NumPy is often used to load, manipulate and preprocess data.
import numpy as np
import tensorflow as tf

# Declare list of features. We only have one numeric feature. There are many
# other types of columns that are more complicated and useful.
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

# An estimator is the front end to invoke training (fitting) and evaluation
# (inference). There are many predefined types like linear regression,
# linear classification, and many neural network classifiers and regressors.
# The following code provides an estimator that does linear regression.
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# TensorFlow provides many helper methods to read and set up data sets.
# Here we use two data sets: one for training and one for evaluation
# We have to tell the function how many batches
# of data (num_epochs) we want and how big each batch should be.
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# We can invoke 1000 training steps by invoking the  method and passing the
# training data set.
estimator.train(input_fn=input_fn, steps=1000)

# Here we evaluate how well our model did.
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)
```

## Summary

- 对tensorflow core有基本的了解
- 对tf.estimator还是一脸懵逼
