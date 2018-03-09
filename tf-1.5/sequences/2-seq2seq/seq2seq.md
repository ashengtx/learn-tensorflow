# Neural Machine Translation(seq2seq) Tutorial

[seq2seq论文](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)

## Introduction

seq2seq应用：machine translation, speech recognition, and text summarization. 

## Basic

### Background on Neural Machine Translation

 traditional phrase-based translation systems performed their task by breaking up source sentences into multiple chunks and then translated them phrase-by-phrase. This led to disfluency in the translation outputs and was not quite like how we, humans, translate. 
 传统的翻译系统是基于短语的，翻译得不流利。而我们人类，是读完一整句，理解之后再翻译的。Neural Machine Translation (NMT)就是模拟这种方式。

 神经翻译模型的encoder-decoder architecture：

 sentence -> "encoder" -> vector -> "decoder" -> target language

The RNN models, differ in terms of: 
(a) directionality – unidirectional or bidirectional; 
(b) depth – single- or multi-layer; and 
(c) type – often either a vanilla RNN, a Long Short-term Memory (LSTM), or a gated recurrent unit (GRU). 

### Trainning - How to build our first NMT system

- encoder_inputs [max_encoder_time, batch_size]: source input words.
- decoder_inputs [max_decoder_time, batch_size]: target input words.
- decoder_outputs [max_decoder_time, batch_size]: target output words, these are 

encoder只有输入，没有输出

#### Embedding

vocabulary只保留词频大于min_count的词，其他低频词被转成一个`<unknown>`token统一对待。

如果没有特殊处理，embedding是一个随机矩阵 [vocab_size, embed_size]，也可以用`word2vec`作为输入。

```py
# Embedding
embedding_encoder = variable_scope.get_variable(
    "embedding_encoder", [src_vocab_size, embedding_size], ...)
# Look up embedding:
#   encoder_inputs: [max_time, batch_size]
#   encoder_emb_inp: [max_time, batch_size, embedding_size]
encoder_emb_inp = embedding_ops.embedding_lookup(
    embedding_encoder, encoder_inputs)
```

#### Encoder

These two RNNs, in principle, can share the same weights; however, in practice, we often use two different RNN parameters (such models do a better job when fitting large training datasets).

encoder和decoder理论上可以共享权重矩阵，不过，在大的数据集上使用不同的矩阵效果更好

```py
# Build RNN cell
encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

# Run Dynamic RNN
#   encoder_outputs: [max_time, batch_size, num_units]
#   encoder_state: [batch_size, num_units]
encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
    encoder_cell, encoder_emb_inp,
    sequence_length=source_sequence_length, time_major=True)
```

没给句子的长度不一，为了节省计算，这里通过`source_sequence_length`告诉使用`dynamic_rnn`句子的具体长度。`time_major=True`指定我们的输入是`time major`.

>
Time_major决定了inputs Tensor前两个dim表示的含义 
time_major=False时[batch_size, sequence_length, embedding_size]
time_major=True时[sequence_length, batch_size, embedding_size]

#### Decoder

`encoder`的初始状态是0，而`decoder`的初始状态为`encoder`的最后一个`hidden state`。

```py
# Build RNN cell
decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

# Helper
helper = tf.contrib.seq2seq.TrainingHelper(
    decoder_emb_inp, decoder_lengths, time_major=True)
# Decoder
decoder = tf.contrib.seq2seq.BasicDecoder(
    decoder_cell, helper, encoder_state,
    output_layer=projection_layer)
# Dynamic decoding
outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, ...)
logits = outputs.rnn_output
```

这里把`helper`和`decoder`分开，使得我们可以复用代码。比如，可以用`GreedyEmbeddingHelper`代替`TrainingHelper`来做greedy decoding。

最后，`projection_layer`是将`decoder`的输出映射成`target vocab size`

#### Loss

>
Important note: It's worth pointing out that we divide the loss by batch_size, so our hyperparameters are "invariant" to batch_size. Some people divide the loss by (batch_size * num_time_steps), which plays down the errors made on short sentences. More subtly, our hyperparameters (applied to the former way) can't be used for the latter way. For example, if both approaches use SGD with a learning of 1.0, the latter approach effectively uses a much smaller learning rate of 1 / num_time_steps.

???没领悟

#### Gradient computation & optimization

```py
# Calculate and clip gradients
params = tf.trainable_variables() # weights and bias
gradients = tf.gradients(train_loss, params) # 原始梯度
clipped_gradients, _ = tf.clip_by_global_norm(
    gradients, max_gradient_norm) # 处理之后的梯度
```

One of the important steps in training RNNs is gradient clipping. 
??? 啥是`gradient clipping`

>
clip_gradient 的引入是为了处理gradient explosion的问题。当在一次迭代中权重的更新过于迅猛的话，很容易导致loss divergence。clip_gradient 的直观作用就是让权重的更新限制在一个合适的范围。
具体的细节是，
１．在solver中先设置一个clip_gradient
２．在前向传播与反向传播之后，我们会得到每个权重的梯度diff，这时不像通常那样直接使用这些梯度进行权重更新，而是先求所有权重梯度的平方和sumsq_diff，如果sumsq_diff > clip_gradient，则求缩放因子scale_factor = clip_gradient / sumsq_diff。这个scale_factor在(0,1)之间。如果权重梯度的平方和sumsq_diff越大，那缩放因子将越小。
３．最后将所有的权重梯度乘以这个缩放因子，这时得到的梯度才是最后的梯度信息。
这样就保证了在一次迭代更新中，所有权重的梯度的平方和在一个设定范围以内，这个范围就是clip_gradient.
这个参数多用于LSTM中.

作者：Gein Chen
链接：https://www.zhihu.com/question/29873016/answer/77647103
来源：知乎

```py
# Optimization
optimizer = tf.train.AdamOptimizer(learning_rate)
update_step = optimizer.apply_gradients(
    zip(clipped_gradients, params)) # 用clip之后的梯度去更新参数
```

### Hangs-on - Let's train an NMT model

download data for training NMT model:
```
nmt/scripts/download_iwslt15.sh /tmp/nmt_data
```

start training
```
mkdir /tmp/nmt_model
python -m nmt.nmt \
    --src=vi --tgt=en \
    --vocab_prefix=/tmp/nmt_data/vocab  \
    --train_prefix=/tmp/nmt_data/train \
    --dev_prefix=/tmp/nmt_data/tst2012  \
    --test_prefix=/tmp/nmt_data/tst2013 \
    --out_dir=/tmp/nmt_model \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --num_units=128 \
    --dropout=0.2 \
    --metrics=bleu
```

### Inference - How to generate translations

Decoding methods include greedy, sampling, and beam-search decoding. Here, we will discuss the greedy decoding strategy.

```
# Helper
helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
    embedding_decoder,
    tf.fill([batch_size], tgt_sos_id), tgt_eos_id)

# Decoder
decoder = tf.contrib.seq2seq.BasicDecoder(
    decoder_cell, helper, encoder_state,
    output_layer=projection_layer)
# Dynamic decoding
outputs, _ = tf.contrib.seq2seq.dynamic_decode(
    decoder, maximum_iterations=maximum_iterations)
translations = outputs.sample_id
```

`decoder`在inference阶段和training阶段的区别就是把上一个时刻的预测词作为下一个时刻的输入。

由于不知道target sequence的长度，使用`maximum_iterations`限制翻译长度。

一种启发式方法是翻译直到source sentence长度的两倍

```
maximum_iterations = tf.round(tf.reduce_max(source_sequence_length) * 2)
```

模型训练好之后，执行翻译
```
cat > /tmp/my_infer_file.vi
# (copy and paste some sentences from /tmp/nmt_data/tst2013.vi)

python -m nmt.nmt \
    --out_dir=/tmp/nmt_model \
    --inference_input_file=/tmp/my_infer_file.vi \
    --inference_output_file=/tmp/nmt_model/output_infer

cat /tmp/nmt_model/output_infer # To view the inference as output
```

在训练中也可以翻译测试，只要存在training checkpoint

## Intermediate

attention mechanism, which was first introduced by [Bahdanau et al., 2015](https://arxiv.org/pdf/1508.04025.pdf), then later refined by [Luong et al., 2015](https://arxiv.org/pdf/1409.0473.pdf) and others.

The key idea of the attention mechanism is to establish direct short-cut connections between the target and the source by paying "attention" to relevant source content as we translate. 

attention mechanism的关键在于建立target和source之间的直接short-cut connections，通过关注source content的相关部分，而不是一整句。

Remember that in the vanilla seq2seq model, we pass the last source state from the encoder to the decoder when starting the decoding process. This works well for short and medium-length sentences; however, for long sentences, the single fixed-size hidden state becomes an information bottleneck.

单纯的seq2seq模型中，在decoding process阶段，我们只将encoder的last state传给decoder，这对于较短或者中等长度的句子还ok，但是对于长句，单一定长的hidden state成为一个信息瓶颈。（???这是为啥）

Instead of discarding all of the hidden states computed in the source RNN, the attention mechanism provides an approach that allows the decoder to peek at them (treating them as a dynamic memory of the source information). By doing so, the attention mechanism improves the translation of longer sentences. 

attention mechanism保留了计算的所有hidden state作为一个动态记忆，让decoder在工作的时候peek at them，这改善了长句的翻译。

### Background on the Attention Mechanism

the attention computation happens at every decoder time step.  It consists of the following stages:

1. 为每个source hidden state计算一个attention weights
2. 把所有的source hidden state乘以相应权重再求和得到一个context vector
3. 再把这个context vector和target hidden state合并成一个attention vector
4. 这个attention vector可以用来预测当前时刻的词，然后再作为下一个时刻的input

### Attention Wrapper API

Instead of having readable & writable memory, the attention mechanism presented in this tutorial is a read-only memory. 

这个教程呈现的attention mechanism是只读记忆，将


