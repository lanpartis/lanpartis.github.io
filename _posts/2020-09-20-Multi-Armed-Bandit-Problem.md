---
layout: post
title: Multi-Armed Bandit Problem 多臂老虎机问题简介
category: Reinforcement Learning
tags: [Reinforcement Learning]
---
## 问题

多臂老虎机问题是强化学习领域一个经典的探索/利用困境。它从赌场的老虎机游戏抽象而来。在这个问题中，每一个单臂老虎机在都会根据某个概率分布产生奖励。玩家的目标是在有限次的游戏过程中，获取最大的总奖励。玩家面临的最大问题便是如何权衡探索与利用：是探索未知机器的奖励分布还是利用已知的奖励分布中收益最高的机器来获取奖励。过于偏好探索会导致花费更多机会在最优机器以外的低收益机器上，过于偏好利用则可能导致错误估计最优机器。

## 探索算法

* $\epsilon$-greedy

一个最简单的想法便是按照一个固定的比例去决定探索和利用，这也是强化学习中非常常用的探索策略。
探索时在所有老虎机中随机选择一个来玩，利用时选择当前收益估计值最大的老虎机来玩。

这种方法的缺点在于探索时所有的机器获得的探索概率相同。

* Boltzmann探索

这种方式利用了当前各个老虎机的收益估计值，按照估计值的大小，以不同的概率(如通过softmax函数将所有估计值转换成概率)来选择使用的老虎机。

## 估计算法

* 期望估计

最简单的估计方式,计算每台老虎机的奖励期望作为收益的估计。

* Upper Confidence Bound(UCB)

运用概率统计中的Chernoff-Hoeffding Bound定理,估计一个有较高置信度的期望区间，然后用这个区间的上界作为对收益的估计。

Chernoff-Hoeffding Bound：

![截屏2020-09-17 下午6.01.39](https://raw.githubusercontent.com/lanpartis/DocsPics/master/images_for_docs/%E6%88%AA%E5%B1%8F2020-09-17%20%E4%B8%8B%E5%8D%886.01.39.png)

例如,$\delta=\sqrt {2\ln T/n}$(T表示玩各个老虎机的次数,n表示玩某一个老虎机的次数)时，这个老虎机的收益期望落在区间$[\overline{X}-\sqrt {2\ln T/n},\overline{X}+\sqrt {2\ln T/n}]$的概率为
$1-2e^{-2n\delta^2}=1-\frac{2}{T^4}$。

当T=4时，n=2时，这个老虎机的收益估计有99.2%的概率落在区间$[\overline{X}-1.18, \overline{X}+1.18]$中，我们对这个老虎机的收益估计取其上界$\overline{X}+1.18$.

* sliding-window UCB

对于奖励分布固定的老虎机，UCB是一个足够优秀的算法了，但如果这个老虎机的背后有人根据情况动态的调整奖励分布，那就需要一个可以适应这种变化的新算法了。sliding-window UCB便是其中一种。它提供了一个长度为N的window用于保存之前的历史，在计算收益期望时只考虑window中的历史信息。

## 应用

探索算法和估计算法可以任意组合，例如在Agent57算法中便结合$\epsilon$-greedy和sliding-window UCB算法。
