# model.py
from torch import nn
# epoch：10-20
# batch_size: 128
# seq_len: 200
# hidden_size: 256
# num_layers: 3
class LSTM(nn.Module):
    def __init__(self, dataset, args):
        super(LSTM, self).__init__()
        self.hidden_size = args.hidden_size if hasattr(args, 'hidden_size') else 256
        self.embedding_size = args.embedding_size if hasattr(args, 'embedding_size') else 128
        self.num_layers = args.num_layers if hasattr(args, 'num_layers') else 3

        n_vocab = len(dataset.uniq_chars)
        self.embedding = nn.Embedding(n_vocab, self.embedding_size)
        self.lstm = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=0.2, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, n_vocab)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden
