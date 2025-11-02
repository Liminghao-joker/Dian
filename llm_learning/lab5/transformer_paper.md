# Transformer论文逐段精读

## Position Encoding

$PE_{(pos,2i)} = sin(pos/10000^{2i/d_{model}})$

$PE_{(pos,2i+1)} = cos(pos/10000^{2i/d_{model}})$

其中$pos$为位置索引，$i$为维度索引。

该设计的优势：任意固定偏移$k$，$PE_{pos+k}$可表示为$PE_{pos}$的线性组合，便于模型学习相对位置。

