#!/usr/bin/env python3
"""
Merge LoRA weights with the base model

After fine-tuning with QLoRA, this script merges the LoRA adapters
back into the base model for easier deployment.
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import torch
from pathlib import Path

# Add paths
sys.path.insert(0, '/workspace/alpamayo')
sys.path.insert(0, '/workspace/alpamayo/src')


def merge_lora_weights(
    base_model_path: str,
    lora_path: str,
    output_path: str,
    safe_serialization: bool = True,
):
    """Merge LoRA weights with base model"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    print("=" * 60)
    print("Merging LoRA Weights")
    print("=" * 60)
    
    print(f"\nBase model: {base_model_path}")
    print(f"LoRA adapter: {lora_path}")
    print(f"Output path: {output_path}")
    
    # Load base model
    print("\nLoading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Load on CPU first
    )
    
    # Load LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    # Merge weights
    print("Merging weights (this may take a while)...")
    model = model.merge_and_unload()
    
    # Save merged model
    print(f"\nSaving merged model to: {output_path}")
    model.save_pretrained(
        output_path,
        safe_serialization=safe_serialization,
    )
    
    # Save tokenizer
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)
    
    print("\n" + "=" * 60)
    print("Merge Complete!")
    print("=" * 60)
    print(f"\nMerged model saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights with base model")
    parser.add_argument(
        "--base_model",
        type=str,
        default="nvidia/Alpamayo-1.5-10B",
        help="Base model path or HuggingFace model ID"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="/workspace/alpamayo/outputs/lora_checkpoint/final",
        help="Path to LoRA adapter"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/workspace/alpamayo/outputs/merged_model",
        help="Output path for merged model"
    )
    parser.add_argument(
        "--safe_serialization",
        type=bool,
        default=True,
        help="Use safe serialization (safetensors)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Merge weights
    merge_lora_weights(
        base_model_path=args.base_model,
        lora_path=args.lora_path,
        output_path=args.output_path,
        safe_serialization=args.safe_serialization,
    )


if __name__ == "__main__":
    main()
