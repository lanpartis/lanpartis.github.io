---
layout: post
title: "Retrace: Safe and Efficient Off-Policy Reinforcement Learning"
category:  Reinforcement Learning
tags: [ Reinforcement Learning, 10-minutes paper]
---
论文链接 [Safe and Efficient Off-Policy Reinforcement Learning](https://arxiv.org/abs/1606.02647)
## 概要

权衡Monte Carlo回报和Q函数的bootstrap估计一直依赖是个强化学习的一个问题，$TD(\lambda)$则是一种比较经典的解决办法。本文讨论$TD(\lambda)$的优化版本$Q(\lambda)$和 $TB(\lambda)$的优劣，并提出了一种新的算法Retrace。目前Retrace算法已经被用于NGU和Agent57中。

### 问题

在强化学习中,使用Monte Carlo回报来优化值函数会面临较大的方差，而使用bootstrap的方式则会因为价值函数的估计误差而导致较大的偏差，这两个方式之间的权衡必然影响强化学习算法的学习效率。

### 现状

$TD(\lambda)$算法通过复合不同长度的真实回报轨迹和不同时刻的价值函数估计，通过超参数$\lambda$进行了两种方式的平滑。并且，在其上也衍生了几种应用到off-policy情况的新算法。

1. $TD(\lambda)$ with Importance Sampling。这种方式可以保证收敛，但由于Importance sampling引入了极大的方差，导致算法的收敛过程不稳定。

2. $Q(\lambda)$:忽略Importance Sampling, 直接应用到off-policy中。这种方式在behavior policy和target policy接近时才可以保证收敛，因此这种方式不太安全。

3. $TB(\lambda)$通过树回溯方式，将target policy的动作概率乘到这个动作后续状态的所有TD上。这种方式可以保证收敛，但对于behavior和target policy在接近时对回报做了过多的压缩，降低了效率。

### 新方法

作者提出了通用算子$\mathcal{R}$:

$$ \mathcal{R} Q(x,a):=Q(x,a)+\mathbb{E}_\mu[\sum_{t\ge0}\gamma^t(\prod_{s=1}^t c_s)(r_t+\gamma\mathbb{E}_\pi Q(x_{t+1},·)-Q(x_t,a_t))] $$

从而将$TB(\lambda)$,$Q(\lambda)$等算法的不同点总结为$c_s$的不同。并提出了新的$c_s$设计。

1. Importance Sampling:

$$ c_s = \lambda\frac{\pi(a_s|x_s)}{\mu(a_s|x_s)} $$

2. $Q(\lambda)$

$$ c_s = \lambda $$

3. $TB(\lambda)$

$$ c_s = \lambda\pi(a_s|x_s) $$

4. Retrace:

$$ c_s = \lambda min(1,\frac{\pi(a_s|x_s)}{\mu(a_s|x_s)}) $$

由于Retrace使用了在1处截断的Importance Sampling，方差得到了降低。同时，因为
$min(1,\frac{\pi(a_s|x_s)}{\mu(a_s|x_s)}) \ge \pi(a_s|x_s)$
所以Retrace对回报的压缩幅度比$TB(\lambda)$更弱，(尤其是在两个policy接近时)提高了回报的利用效率。
