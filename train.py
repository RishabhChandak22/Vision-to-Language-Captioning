# train.py

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import CaptionDataset
from model import EncoderCNN, DecoderRNN
from utils import save_model, get_loader, vocab

# Hyperparameters
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 1

def train():
    # 1. Load data
    train_loader, vocab_size = get_loader(batch_size=BATCH_SIZE)

    # 2. Initialize model
    encoder = EncoderCNN(EMBED_SIZE).to(device)
    decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, vocab_size, NUM_LAYERS).to(device)

    # 3. Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = optim.Adam(params, lr=LEARNING_RATE)

    # 4. Training loop
    for epoch in range(EPOCHS):
        for i, (images, captions) in enumerate(train_loader):
            images, captions = images.to(device), captions.to(device)
            
            features = encoder(images)
            outputs = decoder(features, captions)

            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

        save_model(encoder, decoder, epoch)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train()
