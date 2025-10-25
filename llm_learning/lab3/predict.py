# predict.py
import argparse
import torch
import numpy as np
import random
from model import LSTM as Model
from dataset import IMDBDataset as Dataset
from train import train

def predict(model, dataset, text="the movie was", length=50, temperature=1.0, ctx_len=None):
    device = torch.cuda.is_available() and torch.device('cuda') or torch.device('cpu')
    model.to(device)
    print(f"predict device: {device}")
    model.load_state_dict(torch.load('lstm_imdb_generator.pth'))
    model.eval()
    device = next(model.parameters()).device
    print(f"Generating text on device: {device}")
    
    # 将 text 转为索引
    chars = list(text.lower())
    idxs = []
    for ch in chars:
        if ch in dataset.char_to_idx:
            idxs.append(dataset.char_to_idx[ch])
        else:
            idxs.append(random.choice(list(dataset.char_to_idx.values())))  # 随机选择一个已知字符的索引来替代未知字符

    with torch.no_grad():
        ctx_len = ctx_len if ctx_len is not None else dataset.seq_length
        for _ in range(length):
            # 取最后 ctx_len 个字符作为输入
            # ctx_len应小于等于当前已有的idxs长度
            if ctx_len > len(idxs):
                input_seq = idxs
            else:
                input_seq = idxs[-ctx_len:]
            x = torch.tensor([input_seq], dtype=torch.long).to(device)
            logits, _ = model(x)
            # 取最后一个时间步的 logits
            logits = logits[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=0)
            next_idx = torch.multinomial(probs, 1).item()
            idxs.append(next_idx)
    
    generated = ''.join(dataset.idx_to_char[i] for i in idxs)
    return generated



if __name__ == "__main__":
    dataset = Dataset(argparse.Namespace(data_dir='E:\\university\\Dian\\llm_learning\\lab1\\data\\train', seq_length=100))
    model = Model(dataset, argparse.Namespace(hidden_size=256, embedding_size=128, num_layers=2))
    print(predict(model, dataset, text="This movie is terrible because", length=50, temperature=1.2, ctx_len=5))