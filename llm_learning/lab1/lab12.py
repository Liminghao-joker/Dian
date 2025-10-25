# lab12.py
# 使用 Dataset 和 Dataloader 包装IMDB数据集
from lab11 import MiniIMDBProcessor
import torch
from torch.utils.data import Dataset, DataLoader

class IMDBDataset(Dataset):
    def __init__(self, processor: MiniIMDBProcessor):
        self.processor = processor
        self.texts = processor.reviews
        self.labels = processor.labels
        # preprocess and convert texts to input IDs
        self.sequences = [processor.text_to_sequence(text) for text in self.texts]

    def __len__(self):
        """
        return length of the dataset
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        dataset[idx] 
        """
        return {
            'text': self.texts[idx],
            'input_ids': torch.tensor(self.sequences[idx], dtype=torch.long),
            'labels': int(self.labels[idx]),
        }

#? 为什么必须要将变长序列 padding 到同一长度？
def imdb_collate_batch(batch):
    """
    Collate function to pad sequences to the same length
    """
    texts = [x['text'] for x in batch]
    labels = torch.tensor([x['labels'] for x in batch], dtype=torch.long) # (B,)
    seqs = [x['input_ids'] for x in batch] # (L_i, )

    # Pad sequences to the same length
    max_len = max(s.size(0) for s in seqs) if seqs else 0
    pad_id = 0  # Padding token ID
    input_ids = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_id)  # (B, L_max)
    return {
        'texts': texts,
        'input_ids': input_ids,
        'labels': labels
    }

if __name__ == "__main__":
    processor = MiniIMDBProcessor(select_num=1000) # only use 1000 samples
    dataset = IMDBDataset(processor)
    # test the dataset
    print("Dataset length:", len(dataset))
    sample = dataset[0]
    print("\nSample input IDs:", sample['input_ids'].tolist())
    print("\nSample label:", sample['labels'])

    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, collate_fn=imdb_collate_batch)
    for batch in dataloader:
        input_ids = batch['input_ids']
        labels = batch['labels']
        print("\nInput IDs shape:", input_ids.shape)
        print("\nLabels shape:", labels.shape)
        break
