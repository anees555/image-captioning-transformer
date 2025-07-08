import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch

# -------------------------
# Build Vocabulary
# -------------------------
def build_vocab_from_captions(csv_file, min_freq=1):
    """
    Builds a simple word2idx vocab from captions CSV file.
    """
    df = pd.read_csv(csv_file)
    word_freq = {}
    for caption in df['caption']:
        for word in caption.lower().split():
            word_freq[word] = word_freq.get(word, 0) + 1

    # Start with special tokens
    word2idx = {'<sos>': 0, '<eos>': 1, '<pad>': 2, '<unk>': 3}
    idx = 4
    for word, freq in word_freq.items():
        if freq >= min_freq:
            word2idx[word] = idx
            idx += 1

    return {'word2idx': word2idx}


# -------------------------
# Tokenizer class
# -------------------------
class MyTokenizer:
    def __init__(self, vocab, max_len=30):
        self.word2idx = vocab['word2idx']
        self.max_len = max_len

    def __call__(self, text):
        tokens = text.lower().split()
        ids = [self.word2idx.get(w, self.word2idx['<unk>']) for w in tokens]
        ids = ids[:self.max_len-2]  # reserve space for <sos> <eos>
        ids = [self.word2idx['<sos>']] + ids + [self.word2idx['<eos>']]
        while len(ids) < self.max_len:
            ids.append(self.word2idx['<pad>'])
        return torch.tensor(ids)


# -------------------------
# Image transform
# -------------------------
my_transforms = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225])
])


# -------------------------
# Dataset class
# -------------------------
class ImageCaptionDataset(Dataset):
    def __init__(self, csv_file, img_folder, tokenizer, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_folder = img_folder
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_folder, row['image'])
        caption = row['caption']

        # Load image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Tokenize caption
        caption_tensor = self.tokenizer(caption)

        return image, caption_tensor


# -------------------------
# Testing block
# -------------------------
if __name__ == "__main__":
    # Build vocab from your captions file
    vocab = build_vocab_from_captions('data/Captions/captions.txt')

    # Instantiate tokenizer
    my_tokenizer = MyTokenizer(vocab)

    # Create dataset
    dataset = ImageCaptionDataset(
        csv_file='data/Captions/captions.txt',
        img_folder='data/Images',
        tokenizer=my_tokenizer,
        transform=my_transforms
    )



# DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Test batch
    for imgs, caps in loader:
        print(imgs.shape)  # [32, 3, 224, 224]
        print(caps.shape)  # [32, 30]
        break
