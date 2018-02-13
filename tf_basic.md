
## tf.variable_scope

```py
with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v = tf.get_variable("v", [1])
        assert v.name == "foo/bar/v:0"
```

每个tensorflow节点都有个name，这里的```tf.variable_scope("foo")```就是为这个结点指定名字的scope吧。

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

