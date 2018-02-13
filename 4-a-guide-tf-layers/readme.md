# [A Guide to TF Layers: Building a Convolutional Neural Network](https://www.tensorflow.org/tutorials/layers)

用Tensorflow的layers模块建立一个CNN来识别手写数字，基于[MNIST dataset](http://yann.lecun.com/exdb/mnist/)。

layers提供了一些methods，能够加速全连层，卷积层的创建，添加激活函数，使用dropout regularization。

## Getting Started

Tensorflow程序代码骨架

```py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here

if __name__ == "__main__":
  tf.app.run()
```

## Intro to Convolutional Neural Networks

Convolutional neural networks (CNNs) 是当前图像分类任务的最优架构。CNNs对一张图像的原始像素数据使用了一些filters，来提取高阶特征，然后模型可以用这些特征来对图像进行分类。CNNs包括三个组件：

- Convolutional layers
- Pooling layers
- Dense (fully connected) layers

通常，一个CNN由一系列执行特征提取的卷积模组组成，每个模组由一个卷积层紧接一个池化层构成。最后一个卷积模组连接着一个或多个执行分类的全连层。

CNN的最后一个全连层的每个节点对应模型要预测的每个target class，每个节点都用一个softmax激活函数生成一个0-1之间的值，所有这些softmax值的和为1。这些softmax值可以解释为这张图落到每个类别的概率。

## Building the CNN MNIST Classifier

CNN architecture:

- Convolutional Layer #1: Applies 32 5x5 filters (extracting 5x5-pixel subregions), with ReLU activation function
- Pooling Layer #1: Performs max pooling with a 2x2 filter and stride of 2 (which specifies that pooled regions do not overlap)
- Convolutional Layer #2: Applies 64 5x5 filters, with ReLU activation function
- Pooling Layer #2: Again, performs max pooling with a 2x2 filter and stride of 2
- Dense Layer #1: 1,024 neurons, with dropout regularization rate of 0.4 (probability of 0.4 that any given element will be dropped during training)
- Dense Layer #2 (Logits Layer): 10 neurons, one for each digit target class (0–9).

在```tf.layers```里有三个函数分别创建上述三种layers:

- conv2d()
- max_pooling2d()
- dense()

cnn_mnist.py takes MNIST feature data, labels, and model mode (TRAIN, EVAL, PREDICT) as arguments; configures the CNN; and returns predictions, loss, and a training operation:

```py
def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
```

### Input Layer

在```layers```模块中，为二维图像数据创建卷基层和池化层的方法接受的输入tensor，其shape为```[batch_size, image_width, image_height, channels]```

- batch_size. Size of the subset of examples to use when performing gradient descent during training.
- image_width. Width of the example images.
- image_height. Height of the example images.
- channels. 彩图为3，黑白为1。

MNIST dataset 由28x28 pixel的黑白图构成，因此input layer期望的shape是```[batch_size, 28, 28, 1]```

```py
input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
```

这里-1表示batch size是动态可调的，如果训练的时候features["x"]传入100个sample，batch size就是100。

### Convolutional Layer #1

在第一个卷积层中，使用32个5x5filters，ReLU激活函数

```py
conv1 = tf.layers.conv2d(
    inputs=input_layer,
    filters=32,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)
```

filters=32是人为指定的，为什么设置成32呢???

参数```padding```有两个选项，```valid```（默认值）或者```same```。```padding=same```表示没有在input tensor的边缘补0。在28x28的tensor上做5x5的卷积，会产生24x24的tensor。

conv2d()的输出tensor的shape是```[batch_size, 28, 28, 32]```。

从1通道变成了32通道，因为使用了32个filters。

### Pooling Layer #1

使用2x2的filter执行max pooling，步长为2

```
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
```

宽与高可以设置不同的步长，比如```stride=[3, 6]```。

output shape ```[batch_size, 14, 14, 32]```

### Convolutional Layer #2 and Pooling Layer #2

```py
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)

pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
```

shape of conv2 ```[batch_size, 14, 14, 64]```
shape of pool2 ```[batch_size, 7, 7, 64]```

### Dense Layer

前面的卷积与池化是在为图像提取特征，现在用全连层根据这些特征对图像进行分类。

在连接pool2与dense layer之前，需要先将pool2铺平成```[batch_size, features]```

```py
pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
```

shape of pool2_flat ```[batch_size, 3136]```

然后将其与dense layer相连

```py
dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
```

参数```units=1024```指明全连层有1024个神经元

为了防止过拟合，这里还对dense layer用到dropout regularization

```py
dropout = tf.layers.dropout(
    inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
```

shape of dropout ```[batch_size, 1024]```

### Logits Layer

分几类就创建几个神经元

```py
logits = tf.layers.dense(inputs=dropout, units=10)
```

shape of logits ```[batch_size, 10]```

### Generate Predictions

```py
predictions = {
    "classes": tf.argmax(input=logits, axis=1),
    "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
}
if mode == tf.estimator.ModeKeys.PREDICT:
  return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
```

### Calculate Loss

在training and evaluation阶段，需要定义一个loss function。对于MNIST这样的多分类问题，常常使用cross entropy作为loss。

```py
# Calculate Loss (for both TRAIN and EVAL modes)
onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
loss = tf.losses.softmax_cross_entropy(
  onehot_labels=onehot_labels, logits=logits)
```

从本地load的时候，如果指明了```one_hot=True```，这里就不需要再做one_hot操作，否则会有shape冲突

```py
# Calculate Loss (for both TRAIN and EVAL modes)
#onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
loss = tf.losses.softmax_cross_entropy(
  onehot_labels=labels, logits=logits)
```

### Configure the Training Op

```py
if mode == tf.estimator.ModeKeys.TRAIN:
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
  train_op = optimizer.minimize(
      loss=loss,
      global_step=tf.train.get_global_step())
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
```


### Add evaluation metrics

```py
# Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
```


## Training and Evaluating the CNN MNIST Classifier

上面编好了CNN模型函数，现在我们要train and evaluate it.

### Load Training and Test Data

```py
def main(unused_argv):
  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
```

这个代码从网上load会报错，可能是被墙，也可能是其他原因。解决方法是先从mnist官网下载到本地，再直接从本地load

```py
from tensorflow.examples.tutorials.mnist import input_data

# Import data from local
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
```

### Create the Estimator

```py
# Create the Estimator
mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")
```

### Set Up a Logging Hook

```py
# Set up logging for predictions
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)
```

### Train the Model

```py
# Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)
mnist_classifier.train(
    input_fn=train_input_fn,
    steps=20000,
    hooks=[logging_hook])
```

### Evaluate the Model

```py
# Evaluate the model and print results
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)
eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)
```

这一步报错了，可能是因为计算结果和eval_labels不一致。

计算结果和label如果都是shape(10,)，就不会报错，但是结果accuracy只有 0.24439

后面再分析吧。。。










