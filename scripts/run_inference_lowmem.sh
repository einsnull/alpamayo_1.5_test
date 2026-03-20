#!/bin/bash
# Alpamayo 1.5 - Low Memory Inference Script (Optimized for 7GB VRAM)
# Uses model CPU offloading to run on limited VRAM

set -e

# Read HF_TOKEN from file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOKEN_FILE="${SCRIPT_DIR}/../.hf_token"

if [ ! -f "$TOKEN_FILE" ]; then
    echo "ERROR: HF_TOKEN file not found at $TOKEN_FILE"
    echo "Please create the file with your HuggingFace token."
    exit 1
fi

export HF_TOKEN="$(cat "$TOKEN_FILE")"

echo "=========================================="
echo "Alpamayo 1.5 - Low Memory Inference"
echo "=========================================="

# Memory optimization settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 << 'SCRIPT'
#!/usr/bin/env python3
"""Optimized inference script for Alpamayo 1.5 on low VRAM (7GB)

Optimizations applied:
1. Use eager attention
2. Reduce model to float16
3. CPU offloading for large layers
4. Minimal batch size (1 sample)
"""

import os
import sys
import gc
import numpy as np
import torch
from pathlib import Path

# Add project to path
sys.path.insert(0, '/workspace/alpamayo')
sys.path.insert(0, '/workspace/alpamayo/src')

print("=" * 60)
print("Alpamayo 1.5 - Low Memory Inference (7GB VRAM)")
print("=" * 60)

# Configuration optimized for 7GB VRAM (RTX 2060 SUPER)
CONFIG = {
    "model_name": "nvidia/Alpamayo-1.5-10B",
    "dtype": torch.float16,  # float16 instead of bf16 for less memory
    "attn_implementation": "eager",  # eager attention
    "device_map": "auto",  # Automatic device mapping with CPU offload
    "max_memory": {0: "4GB", "cpu": "30GB"},  # Limit GPU to 4GB, use CPU RAM
    "low_cpu_mem_usage": True,
    "num_traj_samples": 1,  # Single trajectory for low memory
}

print("\nConfiguration:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")

# Check GPU
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This script requires GPU.")
    
print(f"\nGPU: {torch.cuda.get_device_name(0)}")
print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Import Alpamayo modules
try:
    from alpamayo1_5 import helper
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    print("Alpamayo modules loaded successfully!")
except ImportError as e:
    print(f"ERROR: Could not import Alpamayo modules: {e}")
    raise

# Test clip ID
clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
t0_us = 5_100_000

print(f"\n[1/5] Loading dataset for clip_id: {clip_id}...")
data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
print("Dataset loaded.")

# Create messages for VLM
messages = helper.create_message(data["image_frames"].flatten(0, 1))
print(f"Created messages from {len(data['image_frames'])} image frames")

# Load model with CPU offloading
print(f"\n[2/5] Loading model: {CONFIG['model_name']}")
print("  - Using eager attention")
print("  - Device map: auto with CPU offload")
print("  - Dtype: float16")
print("  - GPU limit: 4GB (rest on CPU)")

# Clear GPU cache
torch.cuda.empty_cache()
gc.collect()

model = Alpamayo1_5.from_pretrained(
    CONFIG["model_name"],
    dtype=CONFIG["dtype"],
    attn_implementation=CONFIG["attn_implementation"],
    device_map=CONFIG["device_map"],
    max_memory=CONFIG["max_memory"],
    low_cpu_mem_usage=CONFIG["low_cpu_mem_usage"],
)

print(f"Model loaded. GPU Memory: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

print("\n[3/5] Applying memory optimization...")

# Get processor
processor = helper.get_processor(model.tokenizer)

# Prepare inputs
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    continue_final_message=True,
    return_dict=True,
    return_tensors="pt",
)

model_inputs = {
    "tokenized_data": inputs,
    "ego_history_xyz": data["ego_history_xyz"],
    "ego_history_rot": data["ego_history_rot"],
}

model_inputs = helper.to_device(model_inputs, "cuda")

# Run inference
print("\n[4/5] Running inference...")
print("  - num_traj_samples: 1")
print("  - Note: Slow due to CPU-GPU transfers")

torch.cuda.manual_seed_all(42)

try:
    with torch.autocast("cuda", dtype=torch.float16):
        with torch.no_grad():
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=CONFIG["num_traj_samples"],
                max_generation_length=256,
                return_extra=True,
            )
    
    print("\n[5/5] Results:")
    print("-" * 40)
    
    # Print Chain-of-Causation reasoning
    if "cot" in extra:
        print("\nChain-of-Causation:")
        print(extra["cot"][0][:500] + "..." if len(extra["cot"][0]) > 500 else extra["cot"][0])
    
    # Calculate minADE
    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = diff.min()
    
    print(f"\nminADE: {min_ade:.4f} meters")
    
    # Save results
    output_dir = Path("/workspace/alpamayo/outputs")
    output_dir.mkdir(exist_ok=True)
    
    results = {
        "clip_id": clip_id,
        "min_ade": float(min_ade),
        "num_traj_samples": CONFIG["num_traj_samples"],
        "gpu_memory_gb": torch.cuda.memory_allocated(0)/1024**3,
    }
    
    import json
    with open(output_dir / "inference_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_dir / 'inference_results.json'}")
    print("\nInference completed successfully!")
    
except torch.cuda.OutOfMemoryError:
    print("\n" + "!" * 60)
    print("ERROR: Out of GPU memory!")
    print("!" * 60)
    raise

finally:
    # Cleanup
    del model
    del model_inputs
    torch.cuda.empty_cache()
    gc.collect()

print("\n" + "=" * 60)
print("Inference finished!")
print("=" * 60)
SCRIPT

echo ""
echo "Done!"
