#!/usr/bin/env python3
"""
Accelerate-based QLoRA Fine-tuning for Alpamayo 1.5
Optimized for 7GB VRAM

This script uses HuggingFace Accelerate for distributed training support
and bitsandbytes for 4-bit quantization.
"""

import os
import sys
import gc
import json
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Add paths
sys.path.insert(0, '/workspace/alpamayo')
sys.path.insert(0, '/workspace/alpamayo/src')


@dataclass
class TrainingConfig:
    """Configuration for QLoRA training"""
    model_name: str = "nvidia/Alpamayo-1.5-10B"
    output_dir: str = "/workspace/alpamayo/outputs/lora_checkpoint"
    
    # LoRA settings (optimized for 7GB VRAM)
    lora_rank: int = 16  # Lower rank = less memory
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    
    # Training settings
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 1
    max_steps: int = 100
    warmup_steps: int = 10
    max_grad_norm: float = 0.3
    weight_decay: float = 0.01
    
    # Memory optimization
    use_gradient_checkpointing: bool = True
    use_4bit_quantization: bool = True
    use_paged_adamw: bool = True
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 50
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]


class MemoryMonitor:
    """Monitor GPU memory usage"""
    @staticmethod
    def get_memory_stats() -> Dict[str, float]:
        if not torch.cuda.is_available():
            return {}
        
        stats = {
            "allocated_gb": torch.cuda.memory_allocated(0) / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved(0) / 1024**3,
            "max_allocated_gb": torch.cuda.max_memory_allocated(0) / 1024**3,
        }
        return stats
    
    @staticmethod
    def print_memory(stage: str = ""):
        stats = MemoryMonitor.get_memory_stats()
        if stats:
            print(f"\n[Memory {stage}]")
            for key, value in stats.items():
                print(f"  {key}: {value:.2f} GB")


def setup_quantization():
    """Setup bitsandbytes quantization"""
    import bitsandbytes as bnb
    
    quantization_config = bnb.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    return quantization_config


def load_model_and_tokenizer(config: TrainingConfig):
    """Load model with QLoRA and tokenizer"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("=" * 60)
    print("Loading Model with QLoRA")
    print("=" * 60)
    
    # Quantization config
    bnb_config = setup_quantization()
    
    # Load model
    print(f"\nLoading: {config.model_name}")
    print(f"Quantization: 4-bit NF4")
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    MemoryMonitor.print_memory("After Loading")
    
    return model, tokenizer


def prepare_lora_model(model, config: TrainingConfig):
    """Apply LoRA to the model"""
    from peft import LoraConfig, get_peft_model, TaskType
    
    print("\n" + "=" * 60)
    print("Configuring LoRA")
    print("=" * 60)
    
    print(f"  Rank (r): {config.lora_rank}")
    print(f"  Alpha: {config.lora_alpha}")
    print(f"  Target modules: {config.target_modules}")
    print(f"  Dropout: {config.lora_dropout}")
    
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    
    print(f"\nTrainable parameters:")
    model.print_trainable_parameters()
    
    MemoryMonitor.print_memory("After LoRA Setup")
    
    return model


def create_sample_dataset(tokenizer, num_samples: int = 100):
    """Create a sample dataset for training demonstration"""
    
    class SampleDataset(Dataset):
        def __init__(self, size, tokenizer):
            self.size = size
            self.tokenizer = tokenizer
            
            # Sample prompts related to autonomous driving
            self.prompts = [
                "Describe the road conditions and traffic in this scene.",
                "What obstacles can be detected in the forward camera view?",
                "Predict the trajectory of the ego vehicle.",
                "Analyze the driving scenario and plan the next action.",
                "What are the key objects of interest in this frame?",
            ]
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            # Select a prompt
            prompt = self.prompts[idx % len(self.prompts)]
            
            # Tokenize
            encoding = self.tokenizer(
                prompt,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": encoding["input_ids"].squeeze(0),
            }
    
    return SampleDataset(num_samples, tokenizer)


def training_step(model, batch, config: TrainingConfig):
    """Perform a single training step"""
    # Move batch to device
    batch = {k: v.to(model.device) for k, v in batch.items()}
    
    # Forward pass
    outputs = model(**batch)
    loss = outputs.loss / config.gradient_accumulation_steps
    
    # Backward pass
    loss.backward()
    
    return loss.item() * config.gradient_accumulation_steps


def train(
    model,
    train_dataset,
    config: TrainingConfig,
):
    """Main training loop"""
    from torch.optim import AdamW
    
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    print(f"\nTraining config:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Max steps: {config.max_steps}")
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    # Optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    # Learning rate scheduler
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.max_steps,
        eta_min=config.learning_rate * 0.1,
    )
    
    # Training loop
    model.train()
    global_step = 0
    epoch = 0
    
    pbar = tqdm(total=config.max_steps, desc="Training")
    
    while global_step < config.max_steps:
        epoch += 1
        print(f"\nEpoch {epoch}/{config.num_epochs}")
        
        for batch in train_loader:
            # Forward pass
            loss = training_step(model, batch, config)
            
            # Gradient accumulation
            if (global_step + 1) % config.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.max_grad_norm
                )
                
                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                pbar.update(1)
                
                # Logging
                if global_step % config.logging_steps == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    mem_stats = MemoryMonitor.get_memory_stats()
                    print(f"\n  Step {global_step}: loss={loss:.4f}, lr={current_lr:.2e}")
                    print(f"  Memory: {mem_stats.get('allocated_gb', 0):.2f} GB allocated")
                
                # Save checkpoint
                if global_step % config.save_steps == 0:
                    checkpoint_path = Path(config.output_dir) / f"checkpoint-{global_step}"
                    checkpoint_path.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(checkpoint_path)
                    print(f"  Checkpoint saved: {checkpoint_path}")
                
                if global_step >= config.max_steps:
                    break
            
            # Check for OOM
            if torch.cuda.is_available() and torch.cuda.memory_allocated(0) > 6 * 1024**3:
                print("\nWARNING: High memory usage detected!")
    
    pbar.close()
    
    # Save final model
    print("\nSaving final model...")
    final_path = Path(config.output_dir) / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_path)
    
    # Save training info
    training_info = {
        "config": {
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "target_modules": config.target_modules,
            "final_step": global_step,
        },
        "total_steps": global_step,
        "epochs": epoch,
    }
    
    with open(final_path / "training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    return model


def main():
    """Main entry point"""
    print("=" * 60)
    print("Alpamayo 1.5 QLoRA Fine-tuning")
    print("Accelerate-based Training")
    print("=" * 60)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("ERROR: CUDA not available")
        return
    
    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Configuration
    config = TrainingConfig(
        model_name="nvidia/Alpamayo-1.5-10B",
        output_dir="/workspace/alpamayo/outputs/lora_checkpoint",
        lora_rank=16,  # Low rank for 7GB VRAM
        lora_alpha=32,
        batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=50,  # Start with few steps
        learning_rate=2e-4,
    )
    
    try:
        # Step 1: Load model
        model, tokenizer = load_model_and_tokenizer(config)
        
        # Step 2: Apply LoRA
        model = prepare_lora_model(model, config)
        
        # Step 3: Create dataset
        print("\n" + "=" * 60)
        print("Creating Training Dataset")
        print("=" * 60)
        train_dataset = create_sample_dataset(tokenizer, num_samples=100)
        print(f"Dataset size: {len(train_dataset)}")
        
        # Step 4: Train
        train(model, train_dataset, config)
        
    except torch.cuda.OutOfMemoryError as e:
        print("\n" + "!" * 60)
        print("OUT OF GPU MEMORY!")
        print("!" * 60)
        print("\nSuggestions for 7GB VRAM:")
        print("  1. Reduce LoRA rank: set lora_rank=8")
        print("  2. Reduce sequence length")
        print("  3. Use CPU offloading for vision encoder")
        print("  4. Freeze more layers")
        raise
    
    print("\n\nNext steps:")
    print("  1. Merge LoRA weights: python scripts/merge_lora.py")
    print("  2. Run inference with merged model")


if __name__ == "__main__":
    main()
