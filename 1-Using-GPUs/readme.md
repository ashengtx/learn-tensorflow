## [Using GPUs](Using GPUs)

### Supported devices 设备支持

一台机子上有多个计算设备，Tensorflow支持CPU和GPU。 它们用字符串表示，如：

- "/cpu:0": The CPU of your machine.
- "/device:GPU:0": The GPU of your machine, if you have one.
- "/device:GPU:1": The second GPU of your machine, etc.

如果某个操作（比如矩阵乘法```matmul```）同时有CPU和GPU实现，则优先选择GPU执行.

### Logging Device Placement 

如果想知道每个operation和tensor被分配给哪个设备，可以在创建session的时候配置参数```log_device_placement=True```。

```py
# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))
```

output:
```
[[ 22.  28.]
 [ 49.  64.]]
Device mapping:
/job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: GeForce GTX 750, 
pci bus id: 0000:01:00.0, compute capability: 5.0
MatMul: (MatMul): /job:localhost/replica:0/task:0/device:GPU:0
b: (Const): /job:localhost/replica:0/task:0/device:GPU:0
a: (Const): /job:localhost/replica:0/task:0/device:GPU:0
```

### Manual device placement

可以通过```tf.device```创建一个device context，将具体的operation和tensor手动指派给特定的设备。

```py
# Creates a graph.
with tf.device('/cpu:0'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))
```

这里，```a```和```b```两个结点就会被分配给```cpu:0```。而```MatMul```没有被手动分配，根据优先规则，自动分配给了```gpu:0```

output:
```
[[ 22.  28.]
 [ 49.  64.]]
Device mapping:
/job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: GeForce GTX 750, 
pci bus id: 0000:01:00.0, compute capability: 5.0
MatMul: (MatMul): /job:localhost/replica:0/task:0/device:GPU:0
b: (Const): /job:localhost/replica:0/task:0/device:CPU:0
a: (Const): /job:localhost/replica:0/task:0/device:CPU:0
```

### Allowing GPU memory growth

默认情况下，Tensorflow 会使用几乎全部的显存(GPU memory）。这么做能够减少显存碎片，以提高显存的使用效率。

不过有时候，我们只想分配一部分显存，等有必要的时候再增加显存的量。Tensorflow提供了两种控制方法。

- 方法一 ```allow_growth```

这种方法一开始只分配非常少的显存，随着Session的运行，需要更多显存的时候，再扩展所需的显存区域。（为了减少碎片，并不释放显存）

```py
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config, ...)
```

- 方法二 ```per_process_gpu_memory_fraction```

分配一定比例的显存(for each GPU)

```py
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config, ...)
```

### Using a single GPU on a multi-GPU system

在一个多GPU系统上，默认ID号最小的会被选择。想使用特定的GPU需要手动分配

```py
# Creates a graph.
with tf.device('/device:GPU:2'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))
```

而如果分配的这个GPU不存在，会抛出```InvalidArgumentError```异常。

通过设置```allow_soft_placement = True```，在找不到指定设备的时候，Tensorflow会自动选择一个存在的设备。


```py
# Creates a graph.
with tf.device('/device:GPU:2'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
# Creates a session with allow_soft_placement and log_device_placement set
# to True.
sess = tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=True))
# Runs the op.
print(sess.run(c))
```

### Using multiple GPUs

一脸懵逼

```py
# Creates a graph.
c = []
for d in ['/device:GPU:2', '/device:GPU:3']:
  with tf.device(d):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    c.append(tf.matmul(a, b))
with tf.device('/cpu:0'):
  sum = tf.add_n(c)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(sum))
```

output
```
Device mapping:
/job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: Tesla K20m, pci bus
id: 0000:02:00.0
/job:localhost/replica:0/task:0/device:GPU:1 -> device: 1, name: Tesla K20m, pci bus
id: 0000:03:00.0
/job:localhost/replica:0/task:0/device:GPU:2 -> device: 2, name: Tesla K20m, pci bus
id: 0000:83:00.0
/job:localhost/replica:0/task:0/device:GPU:3 -> device: 3, name: Tesla K20m, pci bus
id: 0000:84:00.0
Const_3: /job:localhost/replica:0/task:0/device:GPU:3
Const_2: /job:localhost/replica:0/task:0/device:GPU:3
MatMul_1: /job:localhost/replica:0/task:0/device:GPU:3
Const_1: /job:localhost/replica:0/task:0/device:GPU:2
Const: /job:localhost/replica:0/task:0/device:GPU:2
MatMul: /job:localhost/replica:0/task:0/device:GPU:2
AddN: /job:localhost/replica:0/task:0/cpu:0
[[  44.   56.]
 [  98.  128.]]
```
