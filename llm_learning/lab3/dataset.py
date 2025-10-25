# 将IMDB数据集处理为适合LSTM模型输入的形式
import os
import re
import numpy as np
import torch
import glob
from torch.utils.data import Dataset, DataLoader
from collections import Counter

def load_imdb_data(data_dir):
    """
    从 aclImdb/train/pos 和 aclImdb/train/neg 加载所有 .txt 文件
    返回一个大字符串（仅训练集）
    """
    texts = []
    for label in ['pos', 'neg']:
        dir_path = os.path.join(data_dir, label, "*.txt")
        for file_path in glob.glob(dir_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read().strip())

        raw_text = "\n".join(texts).lower() # 转小写
        raw_text = re.sub(r'<.*?>', '', raw_text) # 去除HTML标签
        return raw_text

class IMDBDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.seq_length = args.seq_length

        raw_text = load_imdb_data(args.data_dir)
        self.chars = list(raw_text) #? 转为字符列表

        # 获取唯一字符
        self.uniq_chars = sorted(set(self.chars))
        self.vocab_size = len(self.uniq_chars)

        #* 构建映射
        #? 构建字符到索引的映射
        #? 为什么要这样做，具体如何体现？
        self.char_to_idx = {char: idx for idx, char in enumerate(self.uniq_chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.uniq_chars)}

        #? 将全文本转换为索引列表
        self.data = [self.char_to_idx[char] for char in self.chars]

    def __getitem__(self, idx):
        """
        x: input sequence
        y: target sequence (输入序列的下一个字符)
        """
        x = torch.tensor(self.data[idx:idx + self.seq_length], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + self.seq_length + 1], dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.data) - self.seq_length



# if __name__ == "__main__":
#     # test load_imdb_data function
#     data_dir = "E:\\university\\Dian\\llm_learning\\lab1\\data\\train"
#     all_text = load_imdb_data(data_dir)
#     print(f"Loaded {len(all_text)} characters from IMDB training data.")
#     print(all_text[:500])  # 打印前500个字符以检查内容
# test_dataset.py
