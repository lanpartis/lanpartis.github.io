---
layout: post
title:  Attention and Memory in Deep Learning
category: Deep Learning
tags: [Deep Learning, Attention, Neural Networks]
---



Attention : the ability to focus on one thing and ignore others.

Memory : attention throught past informations.

## implicit attention: 神经网络自动会学到的注意力


通过计算神经网络输出y关于输入x的雅可比行列式,可以知道y主要由x的哪些部分决定,这便是神经网络隐含的注意力机制.


## Explicit Attention: 使用专门的结构帮助神经网络处理注意力问题.

computational efficiency

scalable (fixed sized glimpse for any size image)

interpretability


### hard attention

fixed size attention windows moved around the image, trained with rl techniques.

### soft attention

attention doesn't need to be hard. soft attention is differentiable and can train end-to-end with backprop.

eaiser than RL, expensive to compute.

### Associative Attention

不是通过输出的位置来改变注意力,而是通过一个key vector k在所有输入数据x_i中,根据相似度函数S计算的相似度来得到注意力

$$ w_i = \frac{exp\space S(k,x_i)}{\sum_j exp \space S(k,x_j)} $$

S可以是需要训练的,也可使用固定函数如点乘或余弦相似度等.

### Differentiable Visual Attention

类似于hard attention,只是每个window是一个可训练的filter,可以利用反向传播来调节filter的参数.
![20201130145717](https://raw.githubusercontent.com/lanpartis/DocsPics/master/images_for_docs/20201130145717.png)


## introspective Attention

之前的Attention都是对于外部输入的,而对于内部信息同样可以有注意力机制.

## self-attention: Transformer

[The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)