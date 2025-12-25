#!/usr/bin/env python3
"""
Script to create AIME24_L2 dataset by filtering questions with accuracy between 12.5% and 70%
"""

import json
import os
from pathlib import Path
from datasets import load_dataset, Dataset
from tqdm import tqdm

def load_trajectory_files(trajectory_dir):
    """Load all trajectory files and extract accuracy information"""
    trajectory_dir = Path(trajectory_dir)
    trajectory_files = sorted(trajectory_dir.glob("traj_single-turn_q*.json"))
    
    question_accuracies = []
    
    for traj_file in tqdm(trajectory_files, desc="Loading trajectory files"):
        with open(traj_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        question_idx = data.get('question_idx', -1)
        accuracy = data.get('accuracy', 0.0)
        question = data.get('question', '')
        ground_truth = data.get('ground_truth', '')
        
        # Handle both percentage (87.5) and decimal (0.0) formats
        if accuracy > 1.0:  # It's already in percentage
            accuracy_percent = accuracy
            accuracy_decimal = accuracy / 100.0
        else:  # It's in decimal, convert to percentage
            accuracy_percent = accuracy * 100.0
            accuracy_decimal = accuracy
        
        question_accuracies.append({
            'question_idx': question_idx,
            'accuracy': accuracy_decimal,
            'accuracy_percent': accuracy_percent,
            'question': question,
            'ground_truth': ground_truth
        })
    
    return question_accuracies

def filter_questions_by_accuracy(question_accuracies, min_acc=12.5, max_acc=70.0):
    """Filter questions with accuracy between min_acc and max_acc (inclusive)"""
    filtered = []
    for q in question_accuracies:
        acc_percent = q['accuracy_percent']
        if min_acc <= acc_percent <= max_acc:
            filtered.append(q)
    return filtered

def create_aime24_l2_dataset(trajectory_dir, output_path, min_acc=12.5, max_acc=70.0):
    """Main function to create AIME24_L2 dataset"""
    
    print("=" * 80)
    print("Creating AIME24_L2 Dataset")
    print("=" * 80)
    print(f"Filtering criteria: Accuracy between {min_acc}% and {max_acc}%")
    print()
    
    # Step 1: Load trajectory files and extract accuracies
    print("Step 1: Loading trajectory files...")
    question_accuracies = load_trajectory_files(trajectory_dir)
    print(f"Loaded {len(question_accuracies)} questions from trajectory files")
    print()
    
    # Step 2: Filter questions by accuracy
    print(f"Step 2: Filtering questions with accuracy between {min_acc}% and {max_acc}%...")
    filtered_questions = filter_questions_by_accuracy(question_accuracies, min_acc, max_acc)
    print(f"Found {len(filtered_questions)} questions matching the criteria")
    print()
    
    # Display filtered questions
    print("Filtered Questions:")
    print("-" * 80)
    for q in filtered_questions:
        print(f"Question {q['question_idx']}: Accuracy = {q['accuracy_percent']:.2f}%")
    print("-" * 80)
    print()
    
    # Step 3: Load original AIME24 dataset
    print("Step 3: Loading original AIME24 dataset...")
    aime24_dataset = load_dataset("math-ai/aime24", split="test")
    print(f"Loaded {len(aime24_dataset)} questions from AIME24 dataset")
    print()
    
    # Step 4: Extract filtered questions from original dataset
    print("Step 4: Extracting filtered questions from original dataset...")
    filtered_indices = [q['question_idx'] for q in filtered_questions]
    filtered_data = []
    
    for idx in filtered_indices:
        if 0 <= idx < len(aime24_dataset):
            item = aime24_dataset[idx]
            # Add accuracy information to the item
            item_dict = dict(item)
            # Find the corresponding accuracy
            acc_info = next((q for q in filtered_questions if q['question_idx'] == idx), None)
            if acc_info:
                item_dict['accuracy'] = acc_info['accuracy']
                item_dict['accuracy_percent'] = acc_info['accuracy_percent']
            filtered_data.append(item_dict)
        else:
            print(f"Warning: Question index {idx} is out of range (dataset size: {len(aime24_dataset)})")
    
    print(f"Extracted {len(filtered_data)} questions")
    print()
    
    # Step 5: Create new dataset
    print("Step 5: Creating AIME24_L2 dataset...")
    if len(filtered_data) > 0:
        aime24_l2_dataset = Dataset.from_list(filtered_data)
        print(f"Created dataset with {len(aime24_l2_dataset)} questions")
        print()
        
        # Step 6: Save dataset
        print(f"Step 6: Saving dataset to {output_path}...")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON file
        aime24_l2_dataset.to_json(output_path, indent=2)
        print(f"Dataset saved to {output_path}")
        print()
        
        # Also save as Hugging Face dataset format (optional)
        hf_output_path = output_path.parent / (output_path.stem + "_hf")
        aime24_l2_dataset.save_to_disk(str(hf_output_path))
        print(f"Dataset also saved in Hugging Face format to {hf_output_path}")
        print()
    else:
        print("Warning: No questions found matching the criteria. Dataset not created.")
        print()
    
    # Print summary statistics
    print("=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    accuracies = [q['accuracy_percent'] for q in filtered_questions]
    print(f"Total questions in AIME24_L2: {len(filtered_data)}")
    print(f"Accuracy range: {min(accuracies):.2f}% - {max(accuracies):.2f}%")
    print(f"Average accuracy: {sum(accuracies) / len(accuracies):.2f}%")
    print("=" * 80)
    
    return aime24_l2_dataset

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create AIME24_L2 dataset from trajectory files")
    parser.add_argument(
        "--trajectory-dir",
        type=str,
        default="/home/li003968/infinite_think/Policy-As-GenVerifier/InfReasoner/output/trajectories_gpt-4o_20251219_015827",
        help="Directory containing trajectory JSON files"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/home/li003968/infinite_think/Policy-As-GenVerifier/InfReasoner/AIME24_L2.json",
        help="Output path for the new dataset"
    )
    parser.add_argument(
        "--min-acc",
        type=float,
        default=12.5,
        help="Minimum accuracy percentage (default: 12.5)"
    )
    parser.add_argument(
        "--max-acc",
        type=float,
        default=70.0,
        help="Maximum accuracy percentage (default: 70.0)"
    )
    
    args = parser.parse_args()
    
    create_aime24_l2_dataset(
        trajectory_dir=args.trajectory_dir,
        output_path=args.output_path,
        min_acc=args.min_acc,
        max_acc=args.max_acc
    )
