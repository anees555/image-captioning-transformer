import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset

class ImageCaptionDataset(Dataset):
    def __init__(self, csv_file, img_folder, tokenizer, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_folder = img_folder
        self.transform = transform
        self.tokenizer = tokenizer  # e.g., to convert captions to token IDs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_folder, row['image'])
        caption = row['caption']
        
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Tokenize the caption (to tensor of token ids)
        caption_tensor = self.tokenizer(caption)
        
        return image, caption_tensor

dataset = ImageCaptionDataset(
    csv_file='data/Captions/caption.txt', 
    img_folder='data/Images',
    tokenizer=my_tokenizer,
    transform=my_transforms
)
