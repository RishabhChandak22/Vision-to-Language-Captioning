import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from dataset import get_loader
from vocab import DummyVocab
from model import EncoderCNN, DecoderRNN
from PIL import Image

EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 1
BATCH_SIZE = 2
NUM_EPOCHS = 3
LEARNING_RATE = 3e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_paths = [f"test_images/img{i}.jpg" for i in range(2)]
captions = [['a', 'man', 'riding', 'bike'], ['a', 'man', 'on', 'street']]
vocab = DummyVocab()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_loader = get_loader(image_paths, captions, vocab, transform, batch_size=BATCH_SIZE)

encoder = EncoderCNN(EMBED_SIZE).to(DEVICE)
decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, len(vocab), NUM_LAYERS).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
optimizer = optim.Adam(params, lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    for idx, (images, captions) in enumerate(train_loader):
        images = images.to(DEVICE)
        captions = captions.to(DEVICE)

        features = encoder(images)
        outputs = decoder(features, captions)

        loss = criterion(outputs.view(-1, len(vocab)), captions[:, 1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 1 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

print("Training complete.")
