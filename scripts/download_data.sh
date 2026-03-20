#!/bin/bash
# Alpamayo 1.5 - Download Sample Data Script
# Downloads a small subset of the PhysicalAI dataset for testing

set -e

echo "=========================================="
echo "Alpamayo 1.5 - Download Sample Data"
echo "=========================================="

# Check HuggingFace token
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN is not set!"
    echo "Please set your HuggingFace token first:"
    echo "  export HF_TOKEN=your_token_here"
    exit 1
fi

# Create data directory
mkdir -p /workspace/alpamayo/data/samples

echo ""
echo "[1/4] Authenticated with HuggingFace"

# Download sample dataset using Python
python3 << 'EOF'
import os
import json
import warnings
warnings.filterwarnings("ignore")

from datasets import load_dataset
from pathlib import Path

print("\n[2/4] Loading PhysicalAI dataset sample...")

# Dataset configuration
DATASET_NAME = "PhysicalAI/PhysicalAI-Autonomous-Vehicles"
DATA_DIR = "/workspace/alpamayo/data/samples"

# Number of samples to download
NUM_SAMPLES = 10  # Small sample for testing

print(f"Dataset: {DATASET_NAME}")
print(f"Number of samples: {NUM_SAMPLES}")

try:
    # Load small sample from training set
    dataset = load_dataset(
        DATASET_NAME,
        split=f"train[:{NUM_SAMPLES}]",
        trust_remote_code=True,
    )
    
    print(f"Dataset loaded successfully!")
    print(f"Number of samples: {len(dataset)}")
    
    # Show features
    print(f"\nFeatures available:")
    for feature_name in dataset.features:
        print(f"  - {feature_name}")
    
    # Save sample metadata
    sample_info = {
        "dataset_name": DATASET_NAME,
        "num_samples": len(dataset),
        "features": list(dataset.features.keys()),
        "split": f"train[:{NUM_SAMPLES}]",
    }
    
    # Get sample clip IDs
    if "clip_id" in dataset.column_names:
        sample_info["clip_ids"] = [dataset[i]["clip_id"] for i in range(min(5, len(dataset)))]
    
    output_path = Path(DATA_DIR) / "sample_info.json"
    with open(output_path, "w") as f:
        json.dump(sample_info, f, indent=2)
    
    print(f"\nSample info saved to: {output_path}")
    
    # Save a few sample entries for inspection
    print("\n[3/4] Saving sample entries...")
    
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        sample_path = Path(DATA_DIR) / f"sample_{i}.json"
        
        # Convert to JSON-safe format
        sample_dict = {}
        for key, value in sample.items():
            try:
                if hasattr(value, 'tolist'):
                    sample_dict[key] = value.tolist()
                elif hasattr(value, '__dict__'):
                    sample_dict[key] = str(value)
                else:
                    sample_dict[key] = value
            except:
                sample_dict[key] = str(value)
        
        with open(sample_path, "w") as f:
            json.dump(sample_dict, f, indent=2, default=str)
        
        print(f"  Sample {i} saved: {sample_path}")
    
    # Save dataset for training
    print("\n[4/4] Preparing dataset for training...")
    dataset.save_to_disk(f"{DATA_DIR}/dataset")
    print(f"Dataset saved to: {DATA_DIR}/dataset")
    
    # Calculate approximate size
    approx_size_mb = NUM_SAMPLES * 50  # Rough estimate: ~50MB per sample with images
    print(f"\nApproximate data size: ~{approx_size_mb} MB")
    
    print("\n" + "=" * 60)
    print("Sample Data Download Complete!")
    print("=" * 60)
    print("\nYou can now:")
    print("  1. Inspect samples in: /workspace/alpamayo/data/samples/")
    print("  2. Use the data for fine-tuning")
    print("  3. Test inference with sample clips")
    
except Exception as e:
    print(f"\nERROR: {e}")
    print("\nPossible issues:")
    print("  1. Dataset access not granted")
    print("     Request access: https://huggingface.co/datasets/PhysicalAI/PhysicalAI-Autonomous-Vehicles")
    print("  2. Network issues")
    print("  3. Token authentication failed")
    raise

EOF

echo ""
echo "=========================================="
echo "Download Complete!"
echo "=========================================="
