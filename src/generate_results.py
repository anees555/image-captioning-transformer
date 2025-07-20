#!/usr/bin/env python3
"""
Script to generate results.json using your trained model for evaluation.
"""

import torch
import pickle
import json
import os
from models import ImageCaptioningModel
from create_evaluation_files import create_results_from_model

def main():
    """Generate results.json using the trained model"""
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_path = "../data/vocab.pkl"
    model_path = "../checkpoints/best_model.pt"
    image_folder = "../data/Images"
    references_path = "../references.json"
    results_path = "../results.json"
    
    print(f"🔧 Using device: {device}")
    
    # Check if required files exist
    if not os.path.exists(vocab_path):
        print(f" Vocabulary file not found: {vocab_path}")
        print("Please make sure you have saved the vocabulary during training.")
        return
    
    if not os.path.exists(model_path):
        print(f" Model checkpoint not found: {model_path}")
        print("Please make sure you have trained and saved the model.")
        return
        
    if not os.path.exists(references_path):
        print(f" References file not found: {references_path}")
        print("Please run create_evaluation_files.py first to create references.json")
        return
    
    # Load vocabulary
    print(" Loading vocabulary...")
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    
    print(f"✅ Vocabulary loaded with {len(vocab['word2idx'])} words")
    
    # Load model
    print(" Loading model...")
    model = ImageCaptioningModel(
        vocab_size=len(vocab['word2idx']),
        embed_dim=256,
        num_heads=8,
        num_layers=3,
        max_len=30
    ).to(device)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(" Model loaded successfully")
    
    # Generate results
    print(" Starting caption generation...")
    results_path = create_results_from_model(
        model=model,
        vocab=vocab,
        device=device,
        image_folder=image_folder,
        references_path=references_path,
        output_path=results_path
    )
    
    print(f" Results saved to: {results_path}")
    print(" You can now run evaluation.py to get the metrics!")

if __name__ == "__main__":
    main()
