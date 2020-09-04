---
layout: post
title:  "Ape-X: Distributed Prioritized Experience Replay"
category: Reinforcement Learning
tags: [Reinforcement Learning, 10-minutes paper]
---
论文链接[DISTRIBUTED PRIORITIZED EXPERIENCE REPLAY](https://arxiv.org/pdf/1803.00933.pdf)

## 概要
本文提出一种分布式强化学习的方法，将强化学习的经验收集和训练拆分开来，并行的进行收集和训练任务，以生产者/消费者的形式通信。此外，通过增加经验收集任务的线程数量提高了经验收集的效率。

## 算法

![Distributed Training illustration](https://raw.githubusercontent.com/lanpartis/DocsPics/master/images_for_docs/%E6%88%AA%E5%B1%8F2020-09-02%20%E4%B8%8B%E5%8D%885.28.44.png)

Actor定期从Learning处同步policy参数，并在环境上采样数据存入经验池。
Learner负责从经验池中采样经验来优化policy。
Learner和Actor完全并行执行。各自的算法如下：

![Algorithms](https://raw.githubusercontent.com/lanpartis/DocsPics/master/images_for_docs/%E6%88%AA%E5%B1%8F2020-09-02%20%E4%B8%8B%E5%8D%885.28.53.png)

Ape-X除去并行化之外，相比于普通的Prioritized Experience Replay，还有两个优点。

1. 在收集数据阶段Actor便会算出优先值。
2. actor可以执行不同的策略

此外Ape-X的视线中使用到了的一下tricks
1. n-step estimation
2. double Q-learning 
3. dueling architecture
4. Prioritized experience replay
