#!/bin/bash
# Alpamayo 1.5 - QLoRA Fine-tuning Script (Fixed for 8GB VRAM)
# Optimized for RTX 2060 SUPER with CPU offloading
#
# CHANGES FROM ORIGINAL:
# 1. Reads HF_TOKEN from .hf_token file (not environment variable)
# 2. Uses local model cache instead of downloading
# 3. Uses Alpamayo1_5 class with CPU offloading
# 4. Uses eager attention instead of sdpa
# 5. Simplified training loop for low VRAM testing

set -e

echo "=========================================="
echo "Alpamayo 1.5 - QLoRA Fine-tuning"
echo "=========================================="
echo ""
echo "This script fine-tunes Alpamayo 1.5 using QLoRA"
echo "Optimized for GPUs with limited VRAM (8GB)"
echo ""

# Read HF_TOKEN from file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check multiple possible locations
for TOKEN_FILE in "$SCRIPT_DIR/../.hf_token" "/workspace/.hf_token" "/workspace/alpamayo-docker/.hf_token"; do
    if [ -f "$TOKEN_FILE" ]; then
        export HF_TOKEN="$(cat "$TOKEN_FILE")"
        echo "HF_TOKEN loaded from $TOKEN_FILE"
        break
    fi
done

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN file not found!"
    echo "Checked locations:"
    echo "  - $SCRIPT_DIR/../.hf_token"
    echo "  - /workspace/.hf_token"
    echo "  - /workspace/alpamayo-docker/.hf_token"
    echo "Please create the file with your HuggingFace token."
    exit 1
fi

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Training configuration for 8GB VRAM
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-4}
LEARNING_RATE=${LEARNING_RATE:-2e-4}
MAX_STEPS=${MAX_STEPS:-10}
OUTPUT_DIR=${OUTPUT_DIR:-/workspace/alpamayo/outputs/lora_checkpoint}

echo ""
echo "Training Configuration:"
echo "  LORA_RANK: $LORA_RANK"
echo "  LORA_ALPHA: $LORA_ALPHA"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  GRAD_ACCUM: $GRAD_ACCUM"
echo "  LEARNING_RATE: $LEARNING_RATE"
echo "  MAX_STEPS: $MAX_STEPS"
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR
mkdir -p /workspace/alpamayo/outputs/logs

echo "[1/3] Setting up environment..."
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
if torch.cuda.is_available():
    print(f'Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

echo ""
echo "[2/3] Loading and preparing model with QLoRA..."
python3 << 'SCRIPT'
import os
import sys
import gc
import warnings
warnings.filterwarnings("ignore")

import torch
from pathlib import Path

# Add paths
sys.path.insert(0, '/workspace/alpamayo')
sys.path.insert(0, '/workspace/alpamayo/src')

print("=" * 60)
print("Alpamayo 1.5 QLoRA Fine-tuning Setup")
print("=" * 60)

# Configuration from environment
LORA_RANK = int(os.environ.get("LORA_RANK", "8"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "16"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "4"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-4"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "10"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/workspace/alpamayo/outputs/lora_checkpoint")

print(f"\nConfiguration:")
print(f"  LORA_RANK: {LORA_RANK}")
print(f"  LORA_ALPHA: {LORA_ALPHA}")
print(f"  BATCH_SIZE: {BATCH_SIZE}")
print(f"  GRAD_ACCUM: {GRAD_ACCUM}")
print(f"  LEARNING_RATE: {LEARNING_RATE}")

# Clear memory
torch.cuda.empty_cache()
gc.collect()

# CHANGED: Use Alpamayo1_5 instead of AutoModelForCausalLM
print("\n[1/4] Loading Alpamayo 1.5 model with CPU offloading...")
print("  - Using Alpamayo1_5 class (not AutoModelForCausalLM)")
print("  - Using eager attention (not sdpa)")
print("  - Using CPU offloading for low VRAM")

# Local model path
model_path = '/root/.cache/huggingface/hub/models--nvidia--Alpamayo-1.5-10B/snapshots/089866afb4cfa2231fa7822c67e1e8bc91eed46f'

from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

try:
    model = Alpamayo1_5.from_pretrained(
        model_path,
        dtype=torch.float16,
        attn_implementation='eager',
        device_map='auto',
        max_memory={0: '4GB', 'cpu': '30GB'},
        low_cpu_mem_usage=True,
    )
    print("Model loaded successfully!")
    print(f"GPU Memory: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
except Exception as e:
    print(f"ERROR loading model: {e}")
    raise

# Import peft for LoRA
print("\n[2/4] Setting up LoRA...")
try:
    from peft import LoraConfig, get_peft_model, TaskType
    print("peft loaded successfully")
except ImportError:
    print("ERROR: peft not installed")
    print("Run: pip install peft")
    sys.exit(1)

# Configure LoRA
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Apply LoRA
print("\n[3/4] Applying LoRA to model...")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Count parameters
trainable_params = 0
all_params = 0
for _, param in model.named_parameters():
    all_params += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()

print(f"\nTrainable parameters: {trainable_params:,} / {all_params:,} ({100*trainable_params/all_params:.2f}%)")

# Simple training loop for testing
print("\n[4/4] Running training loop (testing)...")

from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=0.01,
)

total_steps = MAX_STEPS
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=2,
    num_training_steps=total_steps
)

model.train()

# Create synthetic data for testing
print(f"Running {MAX_STEPS} training steps with synthetic data...")

for step in range(total_steps):
    # Create synthetic batch
    batch_size = BATCH_SIZE
    seq_len = 128
    
    synthetic_input = torch.randint(100, 5000, (batch_size, seq_len)).cuda()
    synthetic_labels = synthetic_input.clone()
    
    # Forward pass
    outputs = model(input_ids=synthetic_input, labels=synthetic_labels)
    loss = outputs.loss / GRAD_ACCUM
    
    # Backward pass
    loss.backward()
    
    if (step + 1) % GRAD_ACCUM == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        if (step + 1) % 2 == 0:
            print(f"  Step {step + 1}/{total_steps}, Loss: {loss.item() * GRAD_ACCUM:.4f}")
    
    # Clear cache periodically
    if step % 5 == 0:
        torch.cuda.empty_cache()
        gc.collect()

# Save LoRA weights
print("\n[Complete] Saving LoRA weights...")
model.save_pretrained(f"{OUTPUT_DIR}/final")
print(f"LoRA adapter saved to: {OUTPUT_DIR}/final")

print("\n" + "=" * 60)
print("QLoRA Fine-tuning Complete!")
print("=" * 60)
print(f"\nGPU Memory Peak: {torch.cuda.max_memory_allocated(0)/1024**3:.2f} GB")

# Cleanup
del model
torch.cuda.empty_cache()
gc.collect()
SCRIPT

echo ""
echo "[3/3] Training completed successfully!"
echo ""
echo "LoRA adapter saved to: $OUTPUT_DIR/final"
echo ""
echo "Note: This is a simplified training test. For full fine-tuning:"
echo "  1. Load real training data"
echo "  2. Increase MAX_STEPS"
echo "  3. Use proper data loading pipeline"

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
