---
layout: post
title: RECURRENT EXPERIENCE REPLAY IN DISTRIBUTED REINFORCEMENT LEARNIN
category: Reinforcement Learning
tags: [Reinforcement Learning]
---

## 概要

### 问题
### 现状
### 新方法
## 算法

## 实现细节

We propose a new agent, the Recurrent Replay Distributed DQN (R2D2), and use it to study the interplay between recurrent state, experience replay, and distributed training. R2D2 is most similar to Ape-X, built upon prioritized distributed replay and n-step double Q-learning (with n = 5), generating experience by a large number of actors (typically 256) and learning from batches of replayed experience by a single learner. Like Ape-X, we use the dueling network architecture of Wang et al. (2016), but provide an LSTM layer after the convolutional stack, similarly to Gruslys et al. (2018).

Instead of regular (s, a, r, s 0 ) transition tuples, we store ﬁxed-length (m = 80) sequences of (s, a, r) in replay, with adjacent sequences overlapping each other by 40 time steps, and never crossing episode boundaries.

### value rescaling

使用了一个可逆函数来rescale value


### 超参数

![截屏2020-09-03 上午11.16.21](https://raw.githubusercontent.com/lanpartis/DocsPics/master/images_for_docs/%E6%88%AA%E5%B1%8F2020-09-03%20%E4%B8%8A%E5%8D%8811.16.21.png)

## 分析
## 总结
## 核心逻辑代码