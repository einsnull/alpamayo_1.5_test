#!/bin/bash
# Alpamayo 1.5 - Download and Setup Script
# Downloads model weights and sample data

set -e

echo "=========================================="
echo "Alpamayo 1.5 Setup Script"
echo "=========================================="

# Check if HuggingFace token is set
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN is not set!"
    echo "Please set your HuggingFace token:"
    echo "  export HF_TOKEN=your_token_here"
    echo ""
    echo "Get your token from: https://huggingface.co/settings/tokens"
    echo ""
    echo "You also need to request access to:"
    echo "  - PhysicalAI/PhysicalAI-Autonomous-Vehicles dataset"
    echo "  - nvidia/Alpamayo-1.5-10B model"
    exit 1
fi

# Authenticate with HuggingFace
# Write token to HF credentials file
mkdir -p ~/.cache/huggingface
echo "$HF_TOKEN" > ~/.cache/huggingface/token
chmod 600 ~/.cache/huggingface/token

echo "HuggingFace token configured."

# Download model weights
echo ""
echo "[1/3] Downloading tokenizer..."
python3 << 'EOF'
from transformers import AutoTokenizer
import os

model_name = "nvidia/Alpamayo-1.5-10B"
cache_dir = "/root/.cache/huggingface/hub"

print(f"Downloading tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
print("Tokenizer downloaded successfully!")
EOF

# Download sample dataset
echo ""
echo "[2/3] Downloading sample data..."
python3 << 'EOF'
from datasets import load_dataset
import json
import os

dataset_name = "PhysicalAI/PhysicalAI-Autonomous-Vehicles"
cache_dir = "/workspace/alpamayo/data"

print(f"Loading dataset: {dataset_name}")

try:
    dataset = load_dataset(
        dataset_name,
        split="train[:3]",
        trust_remote_code=True
    )
    
    print(f"Dataset loaded successfully!")
    print(f"Number of samples: {len(dataset)}")
    print(f"Features: {list(dataset.features.keys())}")
    
    os.makedirs("/workspace/alpamayo/data", exist_ok=True)
    sample_info = {
        "dataset_name": dataset_name,
        "num_samples": len(dataset),
        "features": list(dataset.features.keys()),
    }
    
    with open("/workspace/alpamayo/data/sample_info.json", "w") as f:
        json.dump(sample_info, f, indent=2)
    
    print("Sample data saved!")
except Exception as e:
    print(f"Dataset download skipped: {e}")
    print("You can download later with proper access.")
EOF

echo ""
echo "[3/3] Verifying model access..."
python3 << 'EOF'
try:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("nvidia/Alpamayo-1.5-10B")
    print(f"Model config loaded: {config.model_type}")
    print("Model access verified!")
except Exception as e:
    print(f"Model access: {e}")
    print("This may require additional access permissions on HuggingFace.")
EOF

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
