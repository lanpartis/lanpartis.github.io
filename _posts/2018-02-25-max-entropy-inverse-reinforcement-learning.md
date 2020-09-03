---
layout: post
title: Maximum Entropy Inverse Reinforcement Learning
category: Deep Learning
tags: [Inverse Reinforcement Learning]
---

The goal of Inverse Reinforcement Learning(IRL) is to infer reward function from expert demonstrations.

#### Given
1. state &action space

2. Rool-outs from $ \pi^* $

#### Goal
1. recover reward function

2. use reward to get policy

## Maximum Entropy Inverse Reinforcement Learning

**Notation:**

* trajectory :　

$$ \tau = \{s_1,a_1,...,s_t,a_t,...,s_T\} $$

* learned reward :

$$ R_{\psi}(\tau) = \sum_{t} r_\psi(s_t,a_t) $$ 

($ \psi $ are learned parameters.)

* expert demonstrations :　

$$ \{\tau_i\}\sim \pi^* $$

**MaxEnt formulation:**

$$ p(\tau)=\frac 1 Z \exp(R_\psi(\tau)) $$

$$ Z = \int \exp(R_\psi(\tau))d\tau $$

To infer the reward function, we maximize the log likelihood of our set of demonstrations with respect to the parameters of our reward function.

$$\max_\psi\sum_{\tau\in \mathcal{D}}\log p_{r_\psi}(\tau) $$

**Maximum Entropy IRL Optimization**

$$\mathcal{L}(\psi) =  \sum_{\tau\in \mathcal{D}}\log p_{r_\psi}(\tau) $$

$$ = \sum_{\tau\in \mathcal{D}} \log \frac 1 Z \exp(R_\psi(\tau)) $$

$$ = \sum_{\tau\in \mathcal{D}} R_\psi(\tau) - M \log Z $$

$$ = \sum_{\tau\in \mathcal{D}} R_\psi(\tau) - M \log \sum_\tau \exp(R_\psi(\tau)) $$

We will use gradient descent to optimize $ \mathcal{L} $

$$ \nabla_\psi\mathcal{L}(\psi) = \sum_{\tau\in\mathcal{D}} \frac{\mathcal{d}R_\psi(\tau)}{\mathcal{d}\psi} - M\underbrace{\frac1{\sum_\tau\exp(R_\psi(\tau))}\sum_\tau \exp(R_\psi(\tau))\frac {\mathcal{d}R_\psi(\tau)}{\mathcal{d}\psi} }$$

The part with under brace can be written as　
$ \sum_\tau p(\tau|\psi)\frac{\mathcal{d}R_\psi(\tau)}{\mathcal{d}\psi} $

Further more, the sum of trajectory is also the sum of state, thus equals　
$ \sum_s p(s|\psi)\frac{\mathcal{d}R_\psi(s)}{\mathcal{d}\psi} $


So now the problem becomes calculating　
$ p(s|\psi) $

let $ \mu_t(s) $ be the prob of visiting s at step t. It can be calculated using DP:

$$ \mu_1(s) = p(s1 = s) $$ 

where $ p(s1) $ is the initial distribution.

$$ \mu_{t+1}(s') = \sum_a\sum_s\mu_t(s)\Pi(a|s)p(s'|s,a) $$

So

$$ p(s|\psi) = \frac 1 T \sum_t\mu_t(s)$$

**MaxEnt IRL Algorithm**

0. Initialize $ \psi $, gather demonstrations $ \mathcal{D} $
1. Solve for optimal policy $ \pi(a \vert s) $ w.r.t. reward $ r_\psi $
2. Solve for state visitation frequencies $ p(s \vert \psi) $
3. Compute gradient $ \nabla_\psi\mathcal{L} = - \frac 1 {\vert \mathcal{D} \vert} \sum_{\tau_d \in \mathcal{D}} \frac {\mathcal{d}r_\psi}{\mathcal{d}\psi}(\tau_\mathcal{d})- \sum_s p(s \vert \psi)\frac {\mathcal{d}r_\psi}{\mathcal{d}\psi}(s)$
4. Update $ \psi $ with one gradient step using $ \nabla_\psi\mathcal{L} $
5. goto 2

## Discuss

The algorithm is only suitable for low dimension state space, action space and state transaction probability is needed.
