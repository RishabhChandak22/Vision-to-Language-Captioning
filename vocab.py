class DummyVocab:
    def __init__(self):
        self.special_tokens = ['<pad>', '<start>', '<end>', '<unk>']
        self.words = ['a', 'man', 'riding', 'bike', 'on', 'street']
        self.itos = self.special_tokens + self.words
        self.stoi = {word: idx for idx, word in enumerate(self.itos)}

        self.pad_idx = self.stoi['<pad>']
        self.start_idx = self.stoi['<start>']
        self.end_idx = self.stoi['<end>']
        self.unk_idx = self.stoi['<unk>']

    def __len__(self):
        return len(self.itos)
