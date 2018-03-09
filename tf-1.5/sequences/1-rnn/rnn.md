# [Recurrent Neural Networks](https://www.tensorflow.org/tutorials/recurrent)

## Introduction

了解rnn和LSTMs，详见[Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
[中文解读](http://www.bijishequ.com/detail/204848)

## Language Modeling 语言建模

这个教程打算用RNN训练一个语言模型，而这个语言模型做的事情就是根据一句话的前几个词预测下一个词。

语言建模是许多其他任务的基础，如语音识别，机器翻译，图像描述（image captioning）

一篇有趣的博客[The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)

这个教程复现了这篇论文[Recurrent Neural Network Regularization, Zaremba et al., 2014](https://arxiv.org/pdf/1409.2329.pdf)

## Data

[PTB dataset](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz)

## The Model

### LSTM

The memory state of the network is initialized with a vector of zeros and gets updated after reading each word.
网络的记忆状态初始化为一个全零向量，读入每个词之后再更新

处理数据是按照每次一个 `batch_size`进行的，每次输入一个某一时刻一个`batch_size`大小的词。

```
t=0  t=1    t=2  t=3     t=4
[The, brown, fox, is,     quick]
[The, red,   fox, jumped, high]

words_in_dataset[0] = [The, The]
words_in_dataset[1] = [brown, red]
words_in_dataset[2] = [fox, fox]
words_in_dataset[3] = [is, jumped]
words_in_dataset[4] = [quick, high]
batch_size = 2, time_steps = 5
```

以这个例子来看，假设每个词的向量大小是300，那么每个时刻输入的就是(2, 300)的矩阵，也就是(batch_size, embed_size)

### Truncated Backpropagation

传统RNN的输出依赖于任意时刻的输入，但是如果序列太长，会导致梯度消失或者梯度爆炸。常见的做法是取固定长度的时刻，然后训练这个有限长度的近似的RNN.

### Inputs

The word IDs will be embedded into a dense representation (see the Vector Representations Tutorial) before feeding to the LSTM. This allows the model to efficiently represent the knowledge about particular words. It is also easy to write:
```
# embedding_matrix is a tensor of shape [vocabulary_size, embedding size]
word_embeddings = tf.nn.embedding_lookup(embedding_matrix, word_ids)
```
The embedding matrix will be initialized randomly and the model will learn to differentiate the meaning of words just by looking at the data.

这里说`embedding_matrix`是随机初始化的，那就有个疑问？在训练的过程中，这个`embedding_matrix`是如何变化的

--------------------------------------

## 先让代码跑起来

先下载[PTB dataset](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz)

解压，直接右键```Extract Here```或者

```
tar xvfz simple-examples.tgz -C $HOME
```

然后

```
cd models/tutorials/rnn/ptb
python ptb_word_lm.py --data_path=$HOME/simple-examples/data/ --model=small
```

正常运行

最后结果

```
Epoch: 13 Train Perplexity: 40.636
Epoch: 13 Valid Perplexity: 119.383
Test Perplexity: 115.250
```

然后研究代码，先看文件```ptb_word_lm.py```。

```py
logging = tf.logging 
```
这个不知道干嘛的，下面的地方也没用到

```py
flags = tf.flags 
flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_integer("num_gpus", 1,
                     "If larger than 1, Grappler AutoParallel optimizer "
                     "will create multiple training replicas with each GPU "
                     "running one replica.")
```
这里通过```flags.DEFINE_string()```函数定义了一些可选参数，通过```python3 ptb_word_lm.py --help```可以显示这些信息

```py
FLAGS = flags.FLAGS
```

这里的```flags.FLAGS```是flags这个模块里```_FlagValues```类的一个实例,在flags这个模块里已经将其实例化```FLAGS = _FlagValues()```

这里，通过```FLAGS.model```这种方式可以读取运行时传入的参数，或者默认值。

```py
class PTBInput(object):
    """The input data."""
    def __init__(self, config, data, name=None):
```
这个类就一个初始化函数，输入数据的时候，初始化一些参数

```py
class PTBModel(object):
    """The PTB model."""
    def __init__(self, is_training, config, input_):

    def _build_rnn_graph(self, inputs, config, is_training):
        """
        两个mode
        if config.rnn_mode == CUDNN:
            _build_rnn_graph_cudnn
        else:
            _build_rnn_graph_lstm
        """
    def _build_rnn_graph_cudnn(self, inputs, config, is_training):
        """Build the inference graph using CUDNN cell."""
    def _get_lstm_cell(self, config, is_training):

    def _build_rnn_graph_lstm(self, inputs, config, is_training):
        """Build the inference graph using canonical LSTM cells."""
    def assign_lr(self, session, lr_value):

    def export_ops(self, name):
        """Exports ops to collections."""
    def import_ops(self):

```
模型的主类，定义了一些构建模型的方法。

```py
class SmallConfig(object):
    """Small config."""
class MediumConfig(object):
    """Medium config."""
class LargeConfig(object):
    """Large config."""
class TestConfig(object):
    """Tiny config, for testing."""
```
配置信息

```py
def run_epoch(session, model, eval_op=None, verbose=False):
    """Runs the model on the given data."""
```

```py
def get_config():
    """Get model config."""
```

```py
def main(_):
```
主程序分析

```py
gpus = [
      x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"
  ]
```

这是为了获得机器里```gpu```的个数。

不过，这里多次调用```list_local_devices()```，会出现报错，```CUDA_ERROR_OUT_OF_MEMORY```
```py
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
```

原因可能是GPU内存被分配之后没有释放，通过```nvidia-smi```可以查看GPU使用情况，果然如此。

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 375.26                 Driver Version: 375.26                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 750     Off  | 0000:01:00.0      On |                  N/A |
| 33%   35C    P0     2W /  38W |    967MiB /   977MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|    0      1609    G   /usr/lib/xorg/Xorg                             182MiB |
|    0      3623    G   compiz                                          73MiB |
|    0      3910    G   fcitx-qimpanel                                   5MiB |
|    0      4650    G   ...el-token=E4A15648C816242EA82DE5760C894BAD    46MiB |
|    0      8475    C   python3                                        309MiB |
|    0      8550    C   python3                                        225MiB |
|    0      8940    C   python3                                         39MiB |
|    0      9201    C   python3                                         81MiB |
+-----------------------------------------------------------------------------+

```

另外，如果要周期性查看，可以使用```watch```

```
$ watch -n 10 nvidia-smi
```

参数```-n 10```表示每10秒刷新一下

用```kill```命令把这几个```python3```进程终止

```
sudo kill -9 8475 8550 8940 9201
```

后面发现，如果在terminal里执行，每次执行完，显存会立即被释放。而如果在sublime text里面执行，则不会被释放，于是会显存溢出。因此，尽量在terminal里面运行。

读入数据
```py
raw_data = reader.ptb_raw_data(FLAGS.data_path)
train_data, valid_data, test_data, _ = raw_data
```

细节去看```reader.py```文件

这里用了两个```with tf.Graph().as_default():```，不知道为什么这样写，一个不行么???

根据均匀分布生成随机数。
```py
initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
```

根据均匀分布生成指定shape的随机数，第一个参数是shape。
```py
initializer=tf.random_uniform(
            [params_size_t], -config.init_scale, config.init_scale)
```

```py
with tf.name_scope("Train"): # 不太懂，只能理解下面是属于“Train”的scope???
  train_input = PTBInput(config=config, data=train_data, name="TrainInput")
  # 读入train_data，并根据config配置batch_size，num_steps等参数
  with tf.variable_scope("Model", reuse=None, initializer=initializer):
    m = PTBModel(is_training=True, config=config, input_=train_input)
  tf.summary.scalar("Training Loss", m.cost)
  tf.summary.scalar("Learning Rate", m.lr)
```

这个```tf.summary.scalar```好像是和Tensorboard可视化有关，等研究Tensorboard的时候再看。???

```py
for name, model in models.items():
    model.export_ops(name)
```

进入```export_ops```函数

```py
def export_ops(self, name):
    """Exports ops to collections."""
    ...
    for name, op in ops.items():
        tf.add_to_collection(name, op) # 见tf_basic
    ...
    util.export_state_tuples(self._initial_state, self._initial_state_name)
    util.export_state_tuples(self._final_state, self._final_state_name)
```

```py
def export_state_tuples(state_tuples, name):
  for state_tuple in state_tuples:
    tf.add_to_collection(name, state_tuple.c)
    tf.add_to_collection(name, state_tuple.h)
```

这里不懂的是```self._initial_state```和```self._final_state```是一个什么样的结构，不过可以知道的是，这里都是把它们里的东西export到collection

在```_build_rnn_graph_cudnn```里
```py
self._initial_state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)
```
在```_build_rnn_graph_lstm```里
```py
self._initial_state = cell.zero_state(config.batch_size, data_type())
```

先往下看

```py
metagraph = tf.train.export_meta_graph()
```

Returns ```MetaGraphDef``` proto. Optionally writes it to filename.

这个函数将graph,saver,collection导出到```MetaGraphDef```协议缓存，为了晚些时间或者在后面的地方导入，to restart training, run inference, or be a subgraph。

也就是下面的部分，前面有export，这里有import
```py
with tf.Graph().as_default():
    tf.train.import_meta_graph(metagraph)
    for model in models.values():
        model.import_ops()
```

我猜tensorflow应该有维护一个collection，这里`export_ops`的时候，通过`tf.add_to_collection(name, op)`将`ops`先导出到词collection，然后`import_ops`的时候再通过`tf.get_collection_ref(name)`取回来。

```py
sv = tf.train.Supervisor(logdir=FLAGS.save_path)
config_proto = tf.ConfigProto(allow_soft_placement=soft_placement)
with sv.managed_session(config=config_proto) as session:
```

这里用到了一个`tf.train.Supervisor`，The Supervisor is a small wrapper around a `Coordinator`, a `Saver`, and a `SessionManager` that takes care of common needs of TensorFlow training programs.

用法

```py
with tf.Graph().as_default():
  ...add operations to the graph...
  # Create a Supervisor that will checkpoint the model in '/tmp/mydir'.
  sv = Supervisor(logdir='/tmp/mydir')
  # Get a TensorFlow session managed by the supervisor.
  with sv.managed_session(FLAGS.master) as sess:
    # Use the session to train the graph.
    while not sv.should_stop():
      sess.run(<my_train_op>)
```

模型有小、中、大三个配置，
max_epoch分别是4、6、14
max_max_epoch分别是13、39、55，
learning_rate初始值都是1.0
lr_decay分别是0.5、0.8、1/1.15(0.87)
num_steps 分别是20,35,35

num_steps可以理解为用前`num_steps`个词预测下一个词


在训练过程中，learning_rate随着epoch衰减（原理???忘了）
```py
for i in range(config.max_max_epoch):
   lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
   print(lr_decay)
```

小模型
```
1.0
1.0
1.0
1.0
0.5
0.25
0.125
0.0625
0.03125
0.015625
0.0078125
0.00390625
0.001953125
```

中
```
1.0
1.0
1.0
1.0
1.0
1.0
0.5
0.25
0.125
0.0625
0.03125
0.015625
0.0078125
0.00390625
0.001953125
0.0009765625
0.00048828125
0.000244140625
...
```

大
```
1.0
1.0
1.0
1.0
1.0
1.0
1.0
1.0
1.0
1.0
1.0
1.0
1.0
1.0
0.5
0.25
0.125
0.0625
0.03125
0.015625
0.0078125
0.00390625
0.001953125
0.0009765625
0.00048828125
0.000244140625
0.0001220703125
...
```

接下来就是具体每一个epoch的训练，一个epoch就是过一遍所有的训练数据，一个epoch之后返回一个困惑度（perplexity），这个值等于 $ e^{-loss} $

```py
train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                     verbose=True)
```

在看`run_epoch`之前，需要先把`m(PTBModel)`给弄懂，这是核心部分。

```py
class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, input_):
    self._is_training = is_training
    self._input = input_
    self._rnn_params = None
    self._cell = None
    self.batch_size = input_.batch_size
    self.num_steps = input_.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size
```

`PTBModel`初始化的时候输入一个`input_`，这个在train,valid,test阶段分别是training_input,valid_input,test_input

而这个`input`是`PTBInput`返回的结果
```py
class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)
```

-----------------------------------

## 总结

这个基于PTB语料库的Language model，其实就做了一件事情，根据前几个词，预测下一个词。




