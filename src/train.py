import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from models import ImageCaptioningModel
from dataset import ImageCaptionDataset, my_transforms, build_vocab_from_captions, MyTokenizer

def train_model(
    csv_path='../data/Captions/captions.txt',
    image_folder='../data/Images',
    epochs=10,
    batch_size=32,
    lr=1e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    checkpoint_path='checkpoints/best_model.pt'
):
    # Step 1: Build vocab and tokenizer
    vocab = build_vocab_from_captions(csv_path)
    tokenizer = MyTokenizer(vocab)
    
    # Step 2: Dataset and Dataloader
    dataset = ImageCaptionDataset(
        csv_file=csv_path,
        img_folder=image_folder,
        tokenizer=tokenizer,
        transform=my_transforms
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Step 3: Model
    model = ImageCaptioningModel(
        vocab_size=len(vocab['word2idx']),
        embed_dim=256,
        num_heads=8,
        num_layers=3,
        max_len=30
    ).to(device)

    # Step 4: Loss and Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['word2idx']['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Step 5: Training Loop
    best_loss = float('inf')
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, captions in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, captions = images.to(device), captions.to(device)

            optimizer.zero_grad()
            outputs = model(images, captions[:, :-1])  # input sequence (except last token)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), captions[:, 1:].reshape(-1))  # target (except first token)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ Saved best model with loss {best_loss:.4f} at {checkpoint_path}")

    print("Training complete ✅")

# Optional: run directly if needed
if __name__ == "__main__":
    train_model()
