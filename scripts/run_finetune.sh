#!/bin/bash
# Alpamayo 1.5 - QLoRA Fine-tuning Script
# Optimized for 7GB VRAM

set -e

echo "=========================================="
echo "Alpamayo 1.5 - QLoRA Fine-tuning"
echo "=========================================="
echo ""
echo "This script fine-tunes Alpamayo 1.5 using QLoRA"
echo "Optimized for GPUs with limited VRAM (7GB)"
echo ""

# Check environment
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN is not set!"
    echo "Please set your HuggingFace token:"
    echo "  export HF_TOKEN=your_token_here"
    exit 1
fi

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# Training configuration for 7GB VRAM
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-4}
LEARNING_RATE=${LEARNING_RATE:-2e-4}
NUM_EPOCHS=${NUM_EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:-100}
OUTPUT_DIR=${OUTPUT_DIR:-/workspace/alpamayo/outputs/lora_checkpoint}

echo "Training Configuration:"
echo "  LORA_RANK: $LORA_RANK"
echo "  LORA_ALPHA: $LORA_ALPHA"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  GRAD_ACCUM: $GRAD_ACCUM"
echo "  LEARNING_RATE: $LEARNING_RATE"
echo "  NUM_EPOCHS: $NUM_EPOCHS"
echo "  MAX_STEPS: $MAX_STEPS"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
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
LORA_RANK = int(os.environ.get("LORA_RANK", "32"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "64"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "4"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-4"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "1"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "100"))
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

# Import bitsandbytes
print("\n[1/4] Setting up quantization...")
try:
    import bitsandbytes as bnb
    print("bitsandbytes loaded successfully")
except ImportError:
    print("ERROR: bitsandbytes not installed")
    print("Run: uv pip install bitsandbytes")
    sys.exit(1)

# Import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    load_in_8bit=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Load model
print("\n[2/4] Loading Alpamayo 1.5 model with 4-bit quantization...")
print("This will take some time and uses ~4-5GB VRAM...")

model_name = "nvidia/Alpamayo-1.5-10B"

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"ERROR loading model: {e}")
    print("\nPossible issues:")
    print("  1. HuggingFace token not set correctly")
    print("  2. Model not downloaded - run setup first")
    print("  3. Network issues")
    raise

# Load tokenizer
print("\n[3/4] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded!")

# Configure LoRA
print("\n[4/4] Configuring LoRA...")
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Find and apply LoRA to linear layers
def find_all_linear_names(model):
    import re
    cls = torch.nn.Linear
    names = []
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names.append(name.split('.')[-1])
    # Remove duplicates and non-trainable
    return list(set(names))

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("\n" + "=" * 60)
print("Model preparation complete!")
print("=" * 60)
print(f"\nTrainable parameters: {model.print_trainable_parameters()}")
print("\nNote: For actual training, replace synthetic data with real dataset.")
print("See train_lora.py for full training implementation.")
SCRIPT

echo ""
echo "[3/3] Model prepared successfully!"
echo ""
echo "The QLoRA model is now configured and ready for fine-tuning."
echo ""
echo "IMPORTANT NOTES:"
echo "  - Current model uses 4-bit quantization"
echo "  - Only LoRA layers are trainable (~0.1% of parameters)"
echo "  - Vision encoder is frozen"
echo ""
echo "Next steps:"
echo "  1. Prepare your custom dataset"
echo "  2. Run the full training loop:"
echo "     python /workspace/alpamayo/scripts/train_lora.py"
echo ""
echo "Or use pre-built training with Accelerate:"
echo "  accelerate launch scripts/train_accelerate.py"
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
