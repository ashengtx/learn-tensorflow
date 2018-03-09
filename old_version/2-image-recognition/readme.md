## [Image Recognition](https://www.tensorflow.org/tutorials/image_recognition)

人类可以轻易区分狮子和老虎，读懂标志，识别人脸，但是，计算机却难以完成这些任务。因为人脑很擅长理解图像。

近些年，机器学习领域在这些困难问题上，取得了巨大的进步。人们发现一种叫做深度卷积神经网络（deep convolutional neural network）的模型，能够在困难的视觉识别任务上取得很好的性能，在一些领域已经赶上甚至超过人类的水平。

研究者通过在[ImageNet](http://www.image-net.org/)数据集上验证他们的工作，显示了计算机视觉稳定进步的过程。许多模型相继出现并取得state-of-the-art结果：[QuocNet](https://static.googleusercontent.com/media/research.google.com/en//archive/unsupervised_icml2012.pdf), [AlexNet](https://www.cs.toronto.edu/~fritz/absps/imagenet.pdf), [Inception (GoogLeNet)](https://arxiv.org/abs/1409.4842), [BN-Inception-v2](https://arxiv.org/abs/1502.03167)。研究者纷纷发表论文描述了这些模型，然而它们还是难以复现。因此，google大佬发布自己在图像识别上最新模型的代码来造福大家，[Inception-v3, 2015](https://arxiv.org/abs/1512.00567)。

Inception-v3模型被训练用于ImageNet Large Visual Recognition Challenge，这是计算机视觉的标准任务，模型需要将所有图片分成[1000类](http://image-net.org/challenges/LSVRC/2014/browse-synsets)，like "Zebra", "Dalmatian", and "Dishwasher"（斑马，斑点犬，洗碗机）。

评估模型的指标是"top-5 error rate"，即一张图可以猜五次，最后统计错误率。on the 2012 validation data set，AlexNet achieved 15.3% top-5 error rate; Inception (GoogLeNet) achieved 6.67%; BN-Inception-v2 achieved 4.9%; Inception-v3 reaches 3.46%。人类的水平大概在5%左右。

### Usage with Python API

直接运行```classify_image.py```，数据载不下来，手动下载没什么问题，原因以后再找。

```
python classify_image.py --model_dir=/home/user/model_path
```

output
```
giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca (score = 0.89107)
indri, indris, Indri indri, Indri brevicaudatus (score = 0.00779)
lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens (score = 0.00296)
custard apple (score = 0.00147)
earthstar (score = 0.00117)
```

用```--image_file```可以指定要识别的图片

这里是用一个已经训练好的模型来识别一张图片，模型的训练过程呢？源码呢？？

以后再说。
