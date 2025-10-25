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
np.random.seed(42)

class MiniIMDBProcessor():
    def __init__(self, data_dir: str = data_dir, min_count: int = 1, select_num: int = 0):
        self.data_dir = data_dir
        self.min_count = min_count
        self.reviews, self.labels = self.read_imdb_data(select_num=select_num)
        self.vocab = self.create_vocab()

    def read_imdb_data(self, select_num: int = 0) -> tuple[list[str], list[int]]:
        reviews = []
        labels = []

        pos_dir = os.path.join(self.data_dir, "train", "pos")
        neg_dir = os.path.join(self.data_dir, "train", "neg")

        for filename in os.listdir(pos_dir):
            with open(os.path.join(pos_dir, filename), "r", encoding="utf-8") as f:
                reviews.append(f.read())
                labels.append(1)  # positive label

        for filename in os.listdir(neg_dir):
            with open(os.path.join(neg_dir, filename), "r", encoding="utf-8") as f:
                reviews.append(f.read())
                labels.append(0)  # negative label
        
        if select_num > 0:
            # # select positive and negative samples randomly
            # selected_reviews = [reviews[i] for i in np.random.choice(len(reviews), size=select_num, replace=False)]
            # # Because we use the random seed, the labels will correspond correctly
            # selected_labels = [labels[i] for i in np.random.choice(len(labels), size=select_num, replace=False)]
            # return selected_reviews, selected_labels
            indices = np.random.choice(len(reviews), size=select_num, replace=False)
            selected_reviews = [reviews[i] for i in indices]
            selected_labels = [labels[i] for i in indices]
            return selected_reviews, selected_labels
        else:
            return reviews, labels

    def preprocess_text(self, text: str) -> list:
        text = re.sub(r'<[^>]+>', ' ', text)
        text = text.lower()
        text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.split()

    def _filter_and_sort_words(self, word_list: list) -> list:
        counter = Counter(word_list)
        items = ((word, cnt) for word, cnt in counter.items() if cnt >= self.min_count)
        sorted_items = sorted(items, key=lambda x: (-x[1], x[0]))
        return [word for word, _ in sorted_items]

    def create_vocab(self) -> dict:
        word_list = list(chain.from_iterable(self.preprocess_text(review) for review in self.reviews))
        filtered_words = self._filter_and_sort_words(word_list)
        vocab = dict((word, idx + 1) for idx, word in enumerate(filtered_words))
        vocab['<unk>'] = 0  # unknown token
        return vocab

    def text_to_sequence(self, text: list) -> list:
        return [self.vocab.get(word, 0) for word in self.preprocess_text(text)]

def main():
    """
    create MiniIMDB vocabulary and process sample text
    """
    processor = MiniIMDBProcessor(data_dir=data_dir, min_count=10, select_num=2000)
    print(f"已加载{len(processor.reviews)}条评论，词表大小为{len(processor.vocab)}。")
    # simple test
    sample_text = "I love this movie, it's fantastic and thrilling!"
    print("Sample text:", sample_text)
    print("Preprocessed words:", processor.preprocess_text(sample_text))
    print("Word indices:", processor.text_to_sequence(sample_text))
    # print top 100 words in vocab
    print("\nTop 100 words in vocabulary:")
    for word, idx in list(processor.vocab.items())[:100]:
        print(f"{word}: {idx}")

if __name__ == "__main__":
    main()



