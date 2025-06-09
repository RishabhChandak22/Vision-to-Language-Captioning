

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

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        caption = self.captions[index]
        caption = [self.vocab.stoi['<start>']] + [self.vocab.stoi[token] for token in caption] + [self.vocab.stoi['<end>']]
        caption = torch.Tensor(caption).long()
        return image, caption


def get_loader(image_paths, captions, vocab, transform, batch_size):
    dataset = CaptionDataset(image_paths, captions, vocab, transform)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    return data_loader
