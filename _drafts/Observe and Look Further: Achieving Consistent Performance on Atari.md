---
layout: post
title: Observe and Look Further, Achieving Consistent Performance on Atar
category: Reinforcement Learning
tags: [Reinforcement Learning, 10-minutes paper]
---

## 概要

### 问题

作者提出了强化学习三个重要问题
1. 处理不同的reward分布
2. 考虑长期回报
3. 高效探索

### 新方法

针对reward分布的问题，作者提出了一个新的Bellman算子来处理不同分布的reward。

针对第二个问题作者提出了temporal consistency loss，使得我们可以使用更大的折扣率$\gamma = 0.999$.

对于最后一个问题，作者通过在算法中通过结合human demonstrations来引导agent朝reward更大的方向探索。

## 算法


## 分析


## 总结