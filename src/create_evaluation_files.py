import pandas as pd
import json
import os
from collections import defaultdict

def create_coco_format_files(csv_path, output_dir="."):
    """
    Create COCO format JSON files for evaluation from your captions CSV.
    
    Args:
        csv_path: Path to your captions.txt file
        output_dir: Directory to save the JSON files
    """
    # Read the captions CSV
    df = pd.read_csv(csv_path)
    
    # Create references (ground truth) in COCO format
    references = {
        "info": {"description": "Image Captioning Dataset"},
        "licenses": [],
        "images": [],
        "annotations": []
    }
    
    # Group captions by image
    image_captions = defaultdict(list)
    for _, row in df.iterrows():
        image_captions[row['image']].append(row['caption'])
    
    # Create image entries and annotations
    image_id = 0
    annotation_id = 0
    
    for image_name, captions in image_captions.items():
        # Add image info
        references["images"].append({
            "id": image_id,
            "file_name": image_name,
            "width": 224,  # Default size, adjust if needed
            "height": 224
        })
        
        # Add all captions for this image
        for caption in captions:
            references["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "caption": caption
            })
            annotation_id += 1
        
        image_id += 1
    
    # Save references file
    references_path = os.path.join(output_dir, "references.json")
    with open(references_path, 'w') as f:
        json.dump(references, f, indent=2)
    
    print(f"✅ Created references.json with {len(references['images'])} images and {len(references['annotations'])} captions")
    
    # Create empty results file template
    results_template = []
    
    # Add one example result for each image (you'll need to replace these with actual model predictions)
    for i, (image_name, captions) in enumerate(image_captions.items()):
        results_template.append({
            "image_id": i,
            "caption": "a placeholder caption that needs to be replaced with model output"
        })
    
    results_path = os.path.join(output_dir, "results_template.json")
    with open(results_path, 'w') as f:
        json.dump(results_template, f, indent=2)
    
    print(f"✅ Created results_template.json with {len(results_template)} entries")
    print("📝 You need to replace the placeholder captions with your model's actual predictions")
    
    return references_path, results_path

def create_results_from_model(model, vocab, device, image_folder, references_path, output_path="results.json"):
    """
    Generate results.json by running your model on all images in the references.
    
    Args:
        model: Your trained ImageCaptioningModel
        vocab: Vocabulary dictionary
        device: torch device
        image_folder: Path to images folder
        references_path: Path to references.json
        output_path: Where to save results.json
    """
    import torch
    from PIL import Image
    from dataset import my_transforms
    
    # Load references to get image list
    with open(references_path, 'r') as f:
        references = json.load(f)
    
    results = []
    model.eval()
    
    print("🔄 Generating captions for all images...")
    
    for img_info in references["images"]:
        image_id = img_info["id"]
        image_name = img_info["file_name"]
        image_path = os.path.join(image_folder, image_name)
        
        try:
            # Generate caption using your model
            caption = generate_caption_for_image(image_path, model, vocab, device)
            
            results.append({
                "image_id": image_id,
                "caption": caption
            })
            
            if len(results) % 100 == 0:
                print(f"Processed {len(results)} images...")
                
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            results.append({
                "image_id": image_id,
                "caption": "error generating caption"
            })
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Created {output_path} with {len(results)} generated captions")
    return output_path

def generate_caption_for_image(image_path, model, vocab, device, max_length=30):
    """Generate caption for a single image"""
    import torch
    from PIL import Image
    from dataset import my_transforms
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image = my_transforms(image).unsqueeze(0).to(device)
    
    # Get word mappings
    word2idx = vocab['word2idx']
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    # Start with <sos> token
    caption_ids = [word2idx['<sos>']]
    
    with torch.no_grad():
        for _ in range(max_length):
            current_caption = torch.tensor(caption_ids).unsqueeze(0).to(device)
            outputs = model(image, current_caption)
            next_token_logits = outputs[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()
            caption_ids.append(next_token_id)
            
            if next_token_id == word2idx.get('<eos>', -1):
                break
    
    # Convert IDs to words
    words = []
    for idx in caption_ids[1:]:  # Skip <sos>
        word = idx2word.get(idx, '<unk>')
        if word == '<eos>':
            break
        if word not in ['<pad>', '<sos>']:
            words.append(word)
    
    return ' '.join(words)

if __name__ == "__main__":
    # Create the COCO format files
    csv_path = "../data/Captions/captions.txt"
    create_coco_format_files(csv_path, output_dir="..")
    
    print("\n🎯 Next steps:")
    print("1. The references.json file is ready to use")
    print("2. Use create_results_from_model() to generate results.json with your trained model")
    print("3. Then run evaluation.py to get BLEU, CIDEr, METEOR scores")
