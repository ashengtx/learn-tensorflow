
## tf.variable_scope

```py
with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v = tf.get_variable("v", [1])
        assert v.name == "foo/bar/v:0"
```

每个tensorflow节点都有个name，这里的```tf.variable_scope("foo")```就是为这个结点指定名字的scope，或者作用域/命名空间。

```py
def foo():
  with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
    v = tf.get_variable("v", [1])
  return v

v1 = foo()  # Creates v.
v2 = foo()  # Gets the same, existing v.
assert v1 == v2
```
or
```py
with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
    v1 = tf.get_variable("v", [1])
    return v
with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
    v2 = tf.get_variable("v", [1])
    return v
assert v1 == v2
```

这里```reuse=tf.AUTO_REUSE```使得```foo```这个name scope可以被reuse，如果没有这个选项，```v2 = foo()```就会报错

```py
with tf.variable_scope("foo"):
    v1 = tf.get_variable("v", [1])
with tf.variable_scope("foo", reuse=True):
    v2 = tf.get_variable("v", [1])
with tf.variable_scope("foo", reuse=True):
    v3 = tf.get_variable("v", [1])
assert v1 == v2
assert v1 == v3
```

参数```reuse=True```的用法，只需要在第二个以后声明就行。

```py
with tf.variable_scope("foo") as scope:
    v1 = tf.get_variable("v", [1])
    scope.reuse_variables()
    v2 = tf.get_variable("v", [1])
    v3 = tf.get_variable("v", [1])
assert v1 == v2
assert v1 == v3
```

这个```scope.reuse_variables()```比```reuse=True```更方便，只需要在第一个变量声明一次

```py
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
    v1 = tf.get_variable("v", [1])
    #  Raises ValueError("... v already exists ...").
```

像这种非reuse scope在声明同名变量的时候就会报错

```py
with tf.variable_scope("foo", reuse=True):
    v = tf.get_variable("v", [1])
    #  Raises ValueError("... v does not exists ...").
```

而在reuse scope，如果这个name的变量之前没有创建过，就会报错。因为它没有东西拿来reuse。

## tf.get_variable

tf.get_variable()通常和tf.variable_scope()一起使用，可用来创建带有名字的变量

tf.get_variable(name, shape, initializer): 通过所给的名字创建并返回一个变量.

```py
test = tf.get_variable("test", [2, 3], dtype="float32")
init = tf.global_variables_initializer() # 初始化之后才有值
sess.run(init)
sess.run(test)

array([[-0.6606623 ,  0.812075  ,  0.7179302 ],
       [ 0.36955142, -0.4984063 , -0.36186665]], dtype=float32)
```

## tf.add_to_collection()

```py
add_to_collection(
    name,
    value
)
```

在```tf.Graph```下，有一个collection，这个函数就是把value加到collection里面名为name的list。

```py
get_collection(
    key,
    scope=None
)
```

而```get_collection```可以将名为key的list从collection里取出来。

## rnn.BasicLSTMCell 内部实现

```py
n_hidden = 128
# 输入x的shape (?, 28, 28) (batch_size, time_steps, input_size)
x=tf.unstack(x,n_steps,1)  
# 由于RNN是按时刻feed数据的，因此要把x reshape
# x在这里unstack之后变成28个(?, 28)的tf.Tensor组成的list
lstm_cell=rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)  
# 这里n_hidden是LSTMCell输出的维度
outputs,states=rnn.static_rnn(lstm_cell,x,dtype=tf.float32)  
# outputs shape是 (?, 128) (batch_size, n_hidden)
# states是一个tuple (state_c, state_h), 两个状态的 shape 都是(?, 128)
```

为什么维度28的input x 进去之后出来的output是128呢，怎么实现的？

看`BasicLSTMCell`里面的call()函数
```py
def call(self, inputs, state):
    """LSTM cell with layer normalization and recurrent dropout."""
    c, h = state
    args = array_ops.concat([inputs, h], 1)
    # 这里的inputs就是x，shape为(?, 28), h的shape为(?, 128)
    # concat之后args的shape为(?, 156)
    concat = self._linear(args)
    # 最关键的信息隐含在_linear()这个函数里
    dtype = args.dtype

    i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)
    if self._layer_norm:
      i = self._norm(i, "input", dtype=dtype)
      j = self._norm(j, "transform", dtype=dtype)
      f = self._norm(f, "forget", dtype=dtype)
      o = self._norm(o, "output", dtype=dtype)
    # 因此这里四个i, j, f, 0 都是(?, 128) 
    g = self._activation(j)
    if (not isinstance(self._keep_prob, float)) or self._keep_prob < 1:
      g = nn_ops.dropout(g, self._keep_prob, seed=self._seed)

    new_c = (
        c * math_ops.sigmoid(f + self._forget_bias) + math_ops.sigmoid(i) * g)
    if self._layer_norm:
      new_c = self._norm(new_c, "state", dtype=dtype)
    new_h = self._activation(new_c) * math_ops.sigmoid(o)

    new_state = rnn_cell_impl.LSTMStateTuple(new_c, new_h)
    # 这里的new_c, new_h也都是(?, 128) 
    return new_h, new_state
```

```py
def _linear(self, args):
    out_size = 4 * self._num_units # 这个_num_units就是隐层大小n_hidden 128，
    proj_size = args.get_shape()[-1] # 这个就是[inputs, h]的最后一个维度156
    dtype = args.dtype
    weights = vs.get_variable("kernel", [proj_size, out_size], dtype=dtype)
    # 这里weights的初始化是动态的，随proj_size而变
    out = math_ops.matmul(args, weights)
    # 因此，不管inputs的大小是28还是128还是188，也不管隐层大小n_hidden是128还是888
    # weights都可以随之调整，这里的out的维度都是(?, 4*128) (?, 4 * self._num_units)
    if not self._layer_norm:
      bias = vs.get_variable("bias", [out_size], dtype=dtype)
      out = nn_ops.bias_add(out, bias)
    return out
```

源码地址： https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/rnn/python/ops/rnn_cell.py

## time major in rnn

Time_major决定了inputs Tensor前两个dim表示的含义 
time_major=False时[batch_size, sequence_length, embedding_size]
time_major=True时[sequence_length, batch_size, embedding_size]

作者：大白象
链接：https://www.zhihu.com/question/66550207/answer/270182707
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
