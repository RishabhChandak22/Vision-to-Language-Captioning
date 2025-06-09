# 🖼️ Image Captioning with CNN-RNN (Vision to Language)

This project implements an end-to-end deep learning model that generates natural language captions for images. A convolutional neural network (CNN) encodes image features, and a recurrent neural network (RNN) decodes them into grammatically correct captions.

## 🚀 Objective
To train a model that can generate accurate and descriptive captions for input images by combining computer vision and natural language processing.

## 📦 Architecture
- **CNN Encoder**: Pretrained ResNet extracts image feature vectors
- **RNN Decoder**: An LSTM generates word sequences conditioned on the encoded features
- **Dataset**: Trained on MSCOCO or Flickr8k image-caption pairs

## 🛠️ Features
- Transfer learning with pretrained CNNs
- Tokenized and padded text data handling
- Custom PyTorch `Dataset` and `DataLoader` classes
- Caption generation with greedy decoding
- Optional BLEU score evaluation

## 🧪 Files
- `train.py` – training loop for the captioning model
- `model.py` – encoder and decoder definitions
- `dataset.py` – dataset preprocessing and loading
- `utils.py` – helper functions for vocabulary, image transforms, etc.
- `inference.py` – generate captions for test images
