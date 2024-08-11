+++
title = "DeepAG: Attack Graph Construction and Threats Prediction With Bi-Directional Deep Learning"
description = "DeepAG：利用双向深度学习构建攻击图和预测威胁"
tags = [
    "paper",
    "Attack Graph",
]
date = "2024-04-02"
categories = [
    "paper",
    "Attack Graph",
]
menu = "main"
+++
## 关键挑战

1. 如何同时检测攻击并根据日志定位攻击点；

2. 如何克服单向模型偏差带来的学习不足的挑战；

3. 如何对**非线性依赖关系进行建模并构建攻击图**来帮助用户掌握入侵者的策略。

### 解决方法
为了应对上述挑战，我们提出了 DeepAG，一种在线方法，能够同时检测 APT 序列，并分别利用日志语义向量和索引定位序列中的攻击阶段，并根据上述日志索引构建攻击图。 当检测到攻击序列时，DeepAG可以定位序列的异常点。 首先，我们提取日志的词汇和语义信息[16]并将其矢量化，以减少日志信息的丢失。 特别地，我们使用的日志序列是由多个连续的日志语句组成的，这可以帮助发现异常的用户行为序列并显示异常点。 为了检测攻击，我们利用变压器模型 [23]，它并行处理高维语义向量并有助于减少运行时间。 此外，我们提出了双向模型来学习日志索引序列之间的位置关系。 基于前向和后向 LSTM，它可以生成多个序列，为可靠的预测提供更多信息，这与传统的 BiLSTM [17] 对同一序列进行单时间步预测不同。 此外，我们还介绍了 OOV 文字处理器和在线更新的机制，分别克服了检测和预测模型学习不足的问题。 最后，DeepAG构建了攻击图，对非线性依赖关系进行建模，直观地展示攻击阶段，帮助用户掌握对手的策略。 我们在四种不同系统日志的开源数据集上评估 DeepAG：HDFS、OpenStack、PageRank 和 BGL 日志。  DeepAG可以高效地实现实时攻击检测，与基线相比，时间成本降低了3倍以上。

## 3.系统模型

虽然多步攻击具有高度的隐蔽性和复杂性，但我们仍然可以通过分析系统日志来发现它们。

受此启发，我们从系统日志中学习攻击阶段，并提取日志模板。

![image-20240715165902109](./images/DeepAG/image-20240715165902109.png)

图2示出了DeepAG的高级概述，其被分为五个部分：文本表示、训练阶段、检测阶段、预测阶段和图构造。

首先，为了表示日志，我们提取了日志模板，然后将它们分别转换为索引和向量。

在训练阶段，我们对日志进行矢量化，得到几个对数向量序列，并将这些序列输入到转换器中进行训练，以进行APT序列检测。

此外，我们向双向模型提供最近指数序列的历史，并将历史之后的下一个指数作为输出。

检测和预测阶段的目的是对APT序列进行判断，通过双向模型得到预测的攻击阶段及其概率分布。

在构建攻击图的基础上，DeepAG提供了一种与条件概率相关的图生成方法。

### 3.1 准备工作

1. 双向预测。在每一步的预测中，由于可能的分叉(即并发)或学习不足，可能会产生不同概率的多个结果。为了提高预测的可靠性，我们提出了包含两个LSTM的双向模型，从两个方向对事件进行验证。它们分别由顺序执行中包含日志序列的正向日志和反向日志序列的反向日志来训练。最后，对于每一个新的条目，在从前向LSTM获得对下一个时间步长t的预测后，我们进一步基于前向LSTM进行h步预测，得到几个长度为h的序列，然后将这些序列反转并输入到后向LSTM，得到对时间步长t的向后预测。最后，我们从两个方向综合对时间步长t的预测，做出可靠和全面的预测。

Bi-directional prediction. In the prediction of every step, there may be multiple results with different probabilities caused by possible forks (i.e., concurrency) or insufficient learning. In order to enhance the reliability of every prediction, we propose the bi-directional model including two LSTMs to validate events from both directions. They are respectively trained by forward logs that contain log sequences in the sequential execution and backward logs of reversed log sequences. In the end, for every new entry, after getting the predictions for next time step t from forward LSTM, we further make h-steps predictions based on forward LSTM and obtain several sequences with length of h. Then we reverse these sequences and input them to backward LSTM to get the backward predictions for time step t. Finally, we integrate the predictions of time step t from two directions to make reliable and comprehensive predictions.

2. 并发事件。日志序列之间的顺序为不同攻击之间的关系提供了重要信息，并且可能存在具有交互关系的日志消息。并发事件是几个不同的线程或并发运行的任务。
3. 攻击图。攻击图是一个直观的框架，可以为分析人员展示可能利用的漏洞和攻击路径。它可以对隔离执行、并发引起的分叉、系统日志之间的循环结构等非线性依赖关系进行建模。

### 3.2 文本表示法

首先将日志转换为索引和向量。

对于日志索引，有些日志中包含一个“标识符字段”，这是一种`数字`形式，代表某个事件。例如，HDFS日志和OpenStack日志可以分别按块ID和实例ID分组到不同的会话中。
![image-20240715171406301](./images/DeepAG/image-20240715171406301.png)

### 3.5 预测阶段

DeepAG能够利用双向模型进行在线预测。我们首先向前向LSM提供新的对数索引序列。在得到预测的日志索引和概率分布后，我们将这些索引视为多个分支的起点。对于它们中的每一个，然后我们对h步进行进一步的预测，并将每个序列存储为原始序列的分支。然后对每个分支进行反转，输入到向后的LSM中，计算预测指标的概率。我们认为只有一个模型预测的指数的概率为0。在获得从两个LSTM(即，前向和后向LSTM)整合的预测索引集合后，DeepAG验证实际标签是否出现在集合中。如果不是，这个预测将被认为是错误的。此外，DeepAG还可以通过分析员的报告调整新模式，以应对错误的预测。