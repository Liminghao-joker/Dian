import os
import torch
import torch.nn as nn
import numpy as np
import re
import string
from torch.utils.data import DataLoader, Dataset
from itertools import chain
from collections import Counter

main_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(main_dir, "data")

def read_imdb_data(data_dir: str = data_dir) -> tuple[list[str], list[int]]:
    """
    Read IMDB dataset from the specified directory.
    Args:
        data_dir (str): Path to the IMDB dataset directory.
    Returns:
        reviews (list): List of review texts.
        labels (list): List of corresponding labels (1 for positive, 0 for negative).
    """
    reviews = []
    labels = []

    pos_dir = os.path.join(data_dir, "train", "pos")
    neg_dir = os.path.join(data_dir, "train", "neg")

    for filename in os.listdir(pos_dir):
        with open(os.path.join(pos_dir, filename), "r", encoding="utf-8") as f:
            reviews.append(f.read())
            labels.append(1)  # positive label

    for filename in os.listdir(neg_dir):
        with open(os.path.join(neg_dir, filename), "r", encoding="utf-8") as f:
            reviews.append(f.read())
            labels.append(0)  # negative label

    return reviews, labels

def preprocess_text(text: str) -> list:
    """
    Preprocess the input text by removing HTML tags, converting to lowercase,
    removing punctuation, and tokenizing into words.
    """
    # remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.lower()
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()

reviews, labels = read_imdb_data()

def _filter_and_sort_words(word_list: list, min_count: int = 1) -> list:
    """
    返回出现次数不少于 min_count 的单词，并按出现次数从高到低排序；
    若频次相同，则按字典序升序。
    """
    counter = Counter(word_list)
    items = ((word, cnt) for word, cnt in counter.items() if cnt >= min_count)
    sorted_items = sorted(items, key=lambda x: (-x[1], x[0]))
    return [word for word, _ in sorted_items]

def create_vocab(word_list: list, min_count: int = 1) -> dict:
    filtered_words = _filter_and_sort_words(word_list, min_count = min_count)
    vocab = dict((word, idx+1) for idx, word in enumerate(filtered_words))
    vocab['<unk>'] = 0  # unknown token
    return vocab

# 将原电影评论以词表序号替代，不在词表的单词统一替换为<unk>序号0
def text_to_sequence(text: list, vocab: dict) -> list:
    return [vocab.get(word, 0) for word in preprocess_text(text)]

class MiniIMDBProcessor():
    def __init__(self, data_dirreviews: list, labels: list):


def test():

def main():
    # test