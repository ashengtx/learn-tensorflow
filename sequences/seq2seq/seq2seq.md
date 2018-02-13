# Neural Machine Translation Tutorial

## Introduction

## Basic

### Background on Neural Machine Translation

 traditional phrase-based translation systems performed their task by breaking up source sentences into multiple chunks and then translated them phrase-by-phrase. This led to disfluency in the translation outputs and was not quite like how we, humans, translate. 
 传统的翻译系统是基于短语的，翻译得不流利。而我们人类，是读完一整句，理解之后再翻译的。Neural Machine Translation (NMT)就是模拟这种方式。

 神经翻译模型的encoder-decoder architecture：

 sentence -> "encoder" -> vector -> "decoder" -> target language

The RNN models, however, differ in terms of: 
(a) directionality – unidirectional or bidirectional; 
(b) depth – single- or multi-layer; and 
(c) type – often either a vanilla RNN, a Long Short-term Memory (LSTM), or a gated recurrent unit (GRU). 
