#!/usr/bin/env python3
"""
Direct CIDEr evaluation without using COCO API to avoid format issues.
"""

import json
import os
from collections import defaultdict

def load_data(reference_path, results_path):
    """Load and format the data for CIDEr evaluation."""
    
    # Load references
    with open(reference_path, 'r') as f:
        references = json.load(f)
    
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Group reference captions by image_id
    gts = defaultdict(list)
    for ann in references['annotations']:
        img_id = ann['image_id']
        caption = ann['caption'].strip()
        gts[img_id].append(caption)  # Store as string, not dict
    
    # Format results
    res = {}
    for result in results:
        img_id = result['image_id']
        caption = result['caption'].strip()
        res[img_id] = [caption]  # Store as list of strings
    
    return gts, res

def direct_cider_evaluation(reference_path, results_path):
    """
    Direct CIDEr evaluation using the CIDEr scorer.
    """
    try:
        # Load data
        print(" Loading references and results...")
        gts, res = load_data(reference_path, results_path)
        
        print(f" Processing {len(res)} images...")
        print(f" Average references per image: {sum(len(refs) for refs in gts.values()) / len(gts):.1f}")
        
        # Import CIDEr scorer
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '../pycocoevalcap'))
        
        from pycocoevalcap.cider.cider import Cider
        
        # Calculate CIDEr score
        print(" Calculating CIDEr score...")
        cider_scorer = Cider()
        cider_score, cider_scores = cider_scorer.compute_score(gts, res)
        
        # Print results
        print("="*50)
        print(" CIDER EVALUATION RESULTS")
        print("="*50)
        print(f"CIDEr Score: {cider_score:.4f}")
        print("="*50)
        
        # Show statistics
        print(f"\n Detailed Statistics:")
        print(f"Total images evaluated: {len(res)}")
        print(f"Total reference captions: {sum(len(refs) for refs in gts.values())}")
        print(f"Average CIDEr per image: {cider_score:.4f}")
        print(f"Min CIDEr score: {min(cider_scores):.4f}")
        print(f"Max CIDEr score: {max(cider_scores):.4f}")
        print(f"Std CIDEr score: {(sum((s - cider_score)**2 for s in cider_scores) / len(cider_scores))**0.5:.4f}")
        
        return cider_score, cider_scores
        
    except Exception as e:
        print(f" Error during CIDEr evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def show_examples(reference_path, results_path, cider_scores, num_examples=3):
    """Show best and worst examples."""
    try:
        # Load raw data
        with open(reference_path, 'r') as f:
            references = json.load(f)
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Create mappings
        result_map = {r['image_id']: r['caption'] for r in results}
        image_map = {img['id']: img['file_name'] for img in references['images']}
        
        # Group annotations by image_id
        image_annotations = defaultdict(list)
        for ann in references['annotations']:
            image_annotations[ann['image_id']].append(ann['caption'])
        
        # Get sorted examples
        imgIds = list(result_map.keys())
        score_pairs = [(img_id, cider_scores[i]) for i, img_id in enumerate(imgIds)]
        score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Show best examples
        print(f"\n TOP {num_examples} BEST PREDICTIONS:")
        print("="*80)
        for i in range(min(num_examples, len(score_pairs))):
            img_id, score = score_pairs[i]
            img_name = image_map.get(img_id, f"image_{img_id}")
            
            print(f"\n {img_name} (CIDEr: {score:.4f})")
            print(f" Predicted: {result_map[img_id]}")
            print(" References:")
            for gt in image_annotations[img_id][:3]:
                print(f"   • {gt}")
        
        # Show worst examples  
        print(f"\n WORST {num_examples} PREDICTIONS:")
        print("="*80)
        for i in range(min(num_examples, len(score_pairs))):
            img_id, score = score_pairs[-(i+1)]
            img_name = image_map.get(img_id, f"image_{img_id}")
            
            print(f"\n {img_name} (CIDEr: {score:.4f})")
            print(f" Predicted: {result_map[img_id]}")
            print(" References:")
            for gt in image_annotations[img_id][:3]:
                print(f"   • {gt}")
                
    except Exception as e:
        print(f"Error showing examples: {e}")

if __name__ == "__main__":
    ref_json = "../references.json"
    res_json = "../results.json"
    
    # Check files exist
    if not os.path.exists(ref_json):
        print(f" Reference file not found: {ref_json}")
        exit(1)
    
    if not os.path.exists(res_json):
        print(f" Results file not found: {res_json}")
        exit(1)
    
    # Run evaluation
    cider_score, cider_scores = direct_cider_evaluation(ref_json, res_json)
    
    if cider_score is not None:
        # Show examples
        show_examples(ref_json, res_json, cider_scores, num_examples=3)
        print(f"\n CIDEr evaluation completed! Final score: {cider_score:.4f}")
    else:
        print("\n CIDEr evaluation failed!")
