import argparse
import torch
import numpy as np
import random
from torch import nn, optim
from torch.utils.data import DataLoader
from model import LSTM as Model
from dataset import IMDBDataset as Dataset

def train(dataset, model, args):
    print('Loading IMDB data...')
    print(f"Model vocab size: {len(dataset.uniq_chars)}")
    device = torch.cuda.is_available() and torch.device('cuda') or torch.device('cpu')
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    model.to(device)
    print(f"Training on device: {device}")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Total sequences: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print("Start training...")
    for epoch in range(args.max_epochs):
        model.train()

        total_loss = 0.0
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (batch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{args.max_epochs}], Step [{batch+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{args.max_epochs}], Avg Loss: {avg_loss:.4f}')

def main():
    #? 为什么采取这种设置参数的方式
    parser = argparse.ArgumentParser(description="Train LSTM on IMDB dataset")
    parser.add_argument('--data_dir', type=str, default='E:\\university\\Dian\\llm_learning\\lab1\\data\\train')
    parser.add_argument('--seq_length', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_epochs', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.002)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)

    args = parser.parse_args()
    # random seed
    torch.manual_seed(42)
    np.random.seed(42)

    dataset = Dataset(args)
    model = Model(dataset, args)
    train(dataset, model, args)

    # save the model
    torch.save(model.state_dict(), "lstm_imdb_generator.pth")
    print("Model saved as 'lstm_imdb_generator.pth'")

if __name__ == "__main__":
    main()