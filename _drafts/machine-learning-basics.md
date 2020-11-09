---
layout: post
title: "模型评估与选择"
category: Machine Learning
tags: [Machine Learning]
---

# CHAP 2 经验误差与过拟合

分类问题：

错误率(error rate)

    若m个样本中有a个样本分类错误，那么错误率.
    E=a/m

精度(accuracy)

    A=1-E

误差(error)

    学习器的预测结果与样本的实际值之间的差异.

    在训练集(training set)上的误差成为训练误差(training error)或经验误差(empirical error).
    在新样本上的误差称为泛化误差(generalization error).

欠拟合(unerfitting)

    训练误差很大.
    通常是由于学习器的能力不足造成的

过拟合(overfitting)

    在训练误差小但泛化误差大.
    有多重因素可能导致过拟合,最常见的情况是由于学习器的学习能力过于强大,把训练样本所包含的不太一般的特性都学到了(最极端的例子是学习器背下了所有的训练样本,而非学到任何可以泛化的共性).

