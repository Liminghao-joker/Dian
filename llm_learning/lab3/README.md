# 基于LSTM的文本生成

## 题目要求

使用IMDB数据集，用LSTM训练一个文本生成器，要求可以生成简单文本。

## 参数设置

**训练超参数**

| 参数            | 值                 | 说明                                           |
| --------------- | ------------------ | ---------------------------------------------- |
| `batch_size`    | `128`              | 每批次样本数                                   |
| `max_epochs`    | `10`               | 总训练轮数                                     |
| `learning_rate` | `0.001`            | Adam 优化器学习率                              |
| 优化器          | `Adam`             | `torch.optim.Adam`                             |
| 损失函数        | `CrossEntropyLoss` | 交叉熵损失函数                                 |
| 随机种子        | `42`               | 固定 `torch` 和 `numpy` 随机种子以保证可复现性 |
|                 |                    |                                                |

## 结果

### 平均损失

`Epoch [4/4], Avg Loss: 1.2309`

### 温度

`temperature`控制采样的随机性。“温度”越低生成的内容越保守，反之则越多样。

*加粗的语句为模型生成内容。*

**temperature is 1.0**
i love this movie because **he way together a character has the which give aw**

**temperature is 0.8**
i love this movie **and his is a machines were to say you will have b**

**temperature is 1.2**
this movie is terrible because **made who (oh, but the savvy, susanching each adel**



## 学习知识

### 参数设置

通过导入`argparse`的形式来进行设置。

## Reference

Example1：[Text Generation with LSTM in PyTorch - MachineLearningMastery.com](https://machinelearningmastery.com/text-generation-with-lstm-in-pytorch/)

 Example2：[PyTorch LSTM: Text Generation Tutorial](https://closeheat.com/blog/pytorch-lstm-text-generation-tutorial)

[推荐文章：彻底理解embedding-CSDN](https://blog.csdn.net/github_37382319/article/details/106939006?ops_request_misc=%7B%22request%5Fid%22%3A%22166705042916800184184497%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=166705042916800184184497&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-106939006-null-null.142^v62^pc_search_tree,201^v3^control_1,213^v1^control&utm_term=embedding&spm=1018.2226.3001.4187)