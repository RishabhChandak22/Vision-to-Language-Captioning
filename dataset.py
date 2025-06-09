# dataset.py

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class CaptionDataset(Dataset):
    def __init__(self, image_paths, captions, vocab, transform=None):
        self.image_paths = image_paths
        self.captions = captions
        self.vocab = vocab
        self.transform = transform

        self.vocab_size = len(vocab)
        self.max_length = max(len(caption) for caption in captions)
        
    def __len__(self):
        return len(self.image_paths)

    