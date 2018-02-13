# [Vector Representations of Words](https://www.tensorflow.org/tutorials/word2vec)

## Motivation: Why Learn Word Embeddings?


## Scaling up with Noise-Contrastive Training


## The Skip-gram Model


## Building the Graph

-------------------------

## 使用TensorFlow训练word2vec

[代码](https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/word2vec/word2vec_basic.py)

直接运行无法下载默认的```text8```数据集。

手动下载到本地，并指定路径，搞定。

换成中文的数据试试，也没问题。

报了个错，研究完代码应该能弄懂。

```py
Traceback (most recent call last):
  File "word2vec_basic.py", line 217, in <module>
    batch_size, num_skips, skip_window)
  File "word2vec_basic.py", line 126, in generate_batch
    buffer[:] = data[:span]
TypeError: sequence index must be integer, not 'slice'
```

下面学习一下这些代码。

先去看一下RNN吧


