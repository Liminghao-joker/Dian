# LSTM | 基础循环神经网络

## 预备知识 RNN

**Why RNN?**

在传统深度网络（如DNN和CNN）的假设是：**所有输入（和输出）是相互独立的**。

但在很多机器学习的任务中，数据以**序列形式**呈现，前后元素高度相关，如自然语言、时间序列（股票预测）、视频处理等。

基于此，RNN的优势体现在：

- 处理变长序列的输入

- 捕捉序列的时间依赖与上下文信息

- 参数共享

	所有的输入共享参数矩阵和偏置。

**RNN的问题——长程依赖**

当序列很长时，在反向传播过程中，梯度需要从序列末尾一直传回开头。梯度是连乘的，如果梯度值通常小于1，连乘后会变得极其微小（vanishing gradients），导致开头的参数几乎得不到更新；反之，如果大于1，则会变得巨大（梯度爆炸）。可归纳为：**隐状态本身传递的是一种短期记忆（Short-Term Memory）**。

为了解决长距离记忆传递和梯度消失/爆炸的问题，我们引入了LSTM（Long Short-term Memory）。

## LSTM

使用两种不同的路径：长期记忆、短期记忆。

- cell-state long-term memory

- hidden-state short-term memory

### Activation function

**Sigmoid**

$\sigma(z)=\frac{1}{1+e^{-z}}\in(0,1)$ 

**tanh**

$\tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}\in(-1,1)$ 

### Step 1 三个门

#### Forget Gate

决定从上一时刻的细胞状态$C_{t-1}$中丢弃哪些信息，也就是要记住长期记忆。 

$f_t=σ(W_f⋅[h_{t−1},x_t]+b_f)$

#### Input Gate

决定哪些新信息被存储到记忆细胞中。

**输入门控信号**

$i_t=\sigma(W_i\cdot[h_{t-1},x_t]+b_i)$

**候选记忆值**

$\tilde{C}_t=\tanh(W_C\cdot[h_{t-1},x_t]+b_C)$

输入门控信号 $i_t$ 控制候选记忆$\tilde{C}$的哪些部分被添加到记忆细胞。

#### Output Gate

决定从记忆细胞中输出什么信息。

$o_t=\sigma(W_o\cdot[h_{t-1},x_t]+b_o)$

它控制着当前记忆细胞状态$C_t$的哪些部分被输出到隐藏状态$h_t$。

### Step 2 更新记忆细胞状态

$C_t=f_t\odot C_{t-1}+i_t\odot\tilde{C}_t$

> - ⊙ 表示 Hadamard积（逐元素相乘）
>
> - $f_t⊙C_{t−1}$：遗忘门控制旧记忆的保留程度。
> - $i_t⊙\tilde{C}_t$：输入门控制新记忆的添加程度。

### Step 3 计算隐藏状态

隐藏状态 $h_t$是短期记忆，用于**当前时间步的输出**和**传递到下一个时间步**，公式更新如下：

$h_t=o_t\odot\tanh(C_t)$



