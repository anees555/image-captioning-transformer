import torch
from PIL import Image
import argparse
import pickle
from src.models import ImageCaptioningModel
from src.dataset import my_transforms


def generate_caption(image_path, model, vocab, device, max_length=30):
    """Generate caption for an image"""
    # Use the same transforms from dataset.py
    image = Image.open(image_path).convert("RGB")
    image = my_transforms(image).unsqueeze(0).to(device)
    
    model.eval()
    
    # Get word2idx and idx2word mappings
    word2idx = vocab['word2idx']
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    # Start with <sos> token
    caption_ids = [word2idx['<sos>']]
    
    with torch.no_grad():
        for _ in range(max_length):
            # Convert current caption to tensor
            current_caption = torch.tensor(caption_ids).unsqueeze(0).to(device)
            
            # Get model output
            outputs = model(image, current_caption)
            
            # Get the last predicted token
            next_token_logits = outputs[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()
            
            # Add to caption
            caption_ids.append(next_token_id)
            
            # Stop if we hit <eos> token
            if next_token_id == word2idx.get('<eos>', -1):
                break
    
    # Convert IDs to words (skip <sos> and <eos>)
    words = []
    for idx in caption_ids[1:]:  # Skip <sos>
        word = idx2word.get(idx, '<unk>')
        if word == '<eos>':
            break
        if word not in ['<pad>', '<sos>']:
            words.append(word)
    
    return ' '.join(words)


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load vocab
    with open(args.vocab_path, "rb") as f:
        vocab = pickle.load(f)

    # Define model
    model = ImageCaptioningModel(
        vocab_size=len(vocab['word2idx']),
        embed_dim=256,
        num_heads=8,
        num_layers=3,
        max_len=30
    ).to(device)

    # Load checkpoint
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))

    # Generate caption
    caption = generate_caption(args.image_path, model, vocab, device)
    print(f"🖼️  Caption: {caption}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to the test image")
    parser.add_argument("--vocab_path", type=str, default="vocab.pkl", help="Path to vocab.pkl")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/best_model.pt", help="Path to trained model checkpoint")

    args = parser.parse_args()
    main(args)
