---
layout: post
title: "Categorical DQN:A Distributional Perspective on Reinforcement Learning"
category: Reinforcement Learning
tags: [Reinforcement Learning, 10-minutes paper]
---

论文链接<[A Distributional Perspective on Reinforcement Learning](https://arxiv.org/pdf/1707.06887.pdf)>

## 概要

通常情况下，强化学习中的值函数是对状态价值或状态动作价值的期望进行估计，而本文提出了对其分布进行估计的算法，将贝尔曼方程拓展到了基于价值分布的形式。在实际运用中也获得了很好的效果。

## 算法

### Distributional Bellman Equation

标准的贝尔曼方程对价值的期望进行迭代，而本论文中则按照以下公式对价值的分布进行迭代。

$$ Z(x, a) =^D R(x, a) +\gamma Z(X', A') $$

Z(x, a)是价值分布，这个分布受到回报R,下一个动作状态对$(X',A')$以及它关于$X',A'$的回报分布$Z(X',A')$这三个随机变量决定。

### Distributional Bellman Operator

根据分布形式的贝尔曼公式,作者给出了策略评估时的算子

$$ {\it{T}}^\pi Z(x,a) :=^D R(x,a)+\gamma P^\pi Z(x,a)  $$

$$ P^\pi Z(x,a):=^D Z(X',A') \quad X'\sim P（·|x,a),A' \sim \pi(·|X') $$

### 价值分布的表示

本论文中使用了离散分布来近似价值分布。这个离散分布由$V_{MIN}$到$V_{MAX}$之间N个均匀分布的点的概率组成。数学上可以表示为:

$$ Z_\theta(x,a)=z_i \quad w.p.\quad p_i(x,a):=\frac{e^{\theta_i(x,a)}}{\sum_je^{\theta_j(x,a)}}$$

$$ z_i \in \{z_i = V_{MIN}+i\Delta z:0 \le i \lt N\},\Delta z := \frac{V_{MAX}-V_{MIN}}{N-1}$$

$z_i$表示了某一个离散的价值，$p_i(x,a)$表示了该价值的概率。


### Projected Bellman Update

对于离散的价值分布，通过算子${\it{T}}^\pi$更新得到的分布${\it{Τ}}^\pi Z_\theta$几乎不会和$Z_\theta$的support有任何交集。因此，作者提出了Projected update的方法，把${\it{T}}^\pi Z(x,a)$的support上的概率分配到相邻的,$Z_\theta$的support上。

$$ (\Phi \hat{\it{T}}Z_\theta(x,a))_i=\Sigma^{N-1}_{j=0}[{1-\frac{|[\hat{\it{T}} z_j]^{V_{MAX}}_{V_{MIN}} - z_i|}{\Delta z}}]^1_0 p_j(x',\pi(x'))$$

$[·]^a_b$表示把它的参数限制在$[a,b]$之间.
$\hat{\it{T}} z_j:=r+\gamma z_j$.

这个式子看起来比较复杂，但表达的意思只是把Z经过算子$\hat{\it{T}}$计算后得到的一组新的support $\hat{\it{T}} Z$上每个support对应的概率，分配给与它相邻（即距离小于$\Delta z$)的$Z_\theta$的support上，分配的多少由他们的距离远近决定。

从算法上能更清晰的看出每个新support的概率是如何分配的。

![20200819160058](https://raw.githubusercontent.com/lanpartis/DocsPics/master/images_for_docs/20200819160058.png)

对于新得到的一个support$\hat{\it{T}} z_j$,它左右的邻近support就是$z_l$和$z_u$，而$\hat{\it{T}} z_j$处的概率$p_j(x_{t+1},a*)$则按照$u-b_j$与$b_j-l$这一比例分配给了$z_l$和$z_u$。

最后，在得到更新后的价值分布后，就可以通过最小化新旧分布的交叉熵来更新参数$\theta$了。需要注意的是，在优化时，新分布应当看作由固定参数$\hat\theta$得到的固定分布$\Phi \hat{\it{T}}Z_{\hat\theta}(x,a))$。

## 核心功能代码
代码实现上相对DQN的区别共有3处。
1. 在获取Q值时需要在用各个atom上的概率计算。
2. 策略迭代时迭代的是分布而不是一个值
3. loss不再是Q与Q_target的MSE而是两个Q分布的交叉熵

Q值计算：
```python
def _dist_to_q(self, dist):
    return (
        dist
        * torch.linspace(
            self.V_MIN, self.V_MAX, self.atom_num, device=self.model.device
        )
    ).sum(-1)
```

Projected Bellman Update：
```python

def _project_distribution(self, next_dist, returns, gammas, done):
    batch_size = next_dist.size(0)
    delta_z = float(self.V_MAX - self.V_MIN) / (self.atom_num - 1)
    support = torch.linspace(
        self.V_MIN, self.V_MAX, self.atom_num, device=self.model.device
    )

    next_dist = next_dist.detach()

    returns = returns.unsqueeze(1).expand_as(next_dist)
    gammas = gammas.unsqueeze(1).expand_as(next_dist)
    support = support.unsqueeze(0).expand_as(next_dist)
    done = to_torch_as(done, returns).unsqueeze(1).expand_as(next_dist)

    new_support = returns + (1 - done) * gammas * support
    new_support = new_support.clamp(min=self.V_MIN, max=self.V_MAX - 1e-4)
    b = (new_support - self.V_MIN) / delta_z
    l = b.floor().long()
    u = l + 1

    offset = (
        torch.linspace(
            0,
            (batch_size - 1) * self.atom_num,
            batch_size,
            device=self.model.device,
        )
        .long()
        .unsqueeze(1)
        .expand(batch_size, self.atom_num)
    )

    proj_dist = torch.zeros(next_dist.size(), device=self.model.device)
    proj_dist.view(-1).index_add_(
        0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
    )
    proj_dist.view(-1).index_add_(
        0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
    )
    return proj_dist
```

loss计算：
```python
loss = -(new_dist * q_dist.log()).sum(1) #axis0是batch纬度
```