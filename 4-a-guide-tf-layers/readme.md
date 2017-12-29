## [A Guide to TF Layers: Building a Convolutional Neural Network](https://www.tensorflow.org/tutorials/layers)

用Tensorflow的layers模块建立一个CNN来识别手写数字，基于[MNIST dataset](http://yann.lecun.com/exdb/mnist/)。

layers提供了一些methods，能够加速全连层，卷积层的创建，添加激活函数，使用dropout regularization。

### Getting Started

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

### Intro to Convolutional Neural Networks

Convolutional neural networks (CNNs) 是当前图像分类任务的最优架构。CNNs对一张图像的原始像素数据使用了一些filters，来提取高阶特征，然后模型可以用这些特征来对图像进行分类。CNNs包括三个组件：

- Convolutional layers
- Pooling layers
- Dense (fully connected) layers
