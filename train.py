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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train()
