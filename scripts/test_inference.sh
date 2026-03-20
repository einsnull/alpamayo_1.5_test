#!/bin/bash
# Alpamayo 1.5 - Test Script
# Tests the complete inference pipeline with CPU offloading

set -e

echo "=========================================="
echo "Alpamayo 1.5 - Test Script"
echo "=========================================="
echo ""

# Read HF_TOKEN from file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOKEN_FILE="${SCRIPT_DIR}/../.hf_token"

if [ ! -f "$TOKEN_FILE" ]; then
    echo "ERROR: HF_TOKEN file not found at $TOKEN_FILE"
    echo "Please create the file with your HuggingFace token."
    exit 1
fi

export HF_TOKEN="$(cat "$TOKEN_FILE")"
echo "HF_TOKEN loaded from .hf_token file"
echo ""

# Memory optimization settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /workspace/alpamayo

python3 << 'PYEOF'
import sys
sys.path.insert(0, '/workspace/alpamayo')
import torch
import gc
import numpy as np
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()
gc.collect()

print("=" * 60)
print("Alpamayo 1.5 - Test Inference")
print("=" * 60)
print("")

# Check GPU
if not torch.cuda.is_available():
    print("ERROR: CUDA is not available!")
    sys.exit(1)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print("")

# Test 1: Model Loading
print("[Test 1/5] Model Loading with CPU Offloading...")
model_path = '/root/.cache/huggingface/hub/models--nvidia--Alpamayo-1.5-10B/snapshots/089866afb4cfa2231fa7822c67e1e8bc91eed46f'

try:
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    
    model = Alpamayo1_5.from_pretrained(
        model_path,
        dtype=torch.float16,
        attn_implementation='eager',
        device_map='auto',
        max_memory={0: '4GB', 'cpu': '30GB'},
        low_cpu_mem_usage=True,
    )
    
    gpu_mem = torch.cuda.memory_allocated(0)/1024**3
    print(f"  PASS: Model loaded. GPU Memory: {gpu_mem:.2f} GB")
    
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {str(e)[:100]}")
    sys.exit(1)

# Test 2: Data Loading
print("\n[Test 2/5] Data Loading...")
try:
    from alpamayo1_5 import helper
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
    
    clip_id = '030c760c-ae38-49aa-9ad8-f5650a545d26'
    data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
    print(f"  PASS: Data loaded. Frames: {len(data['image_frames'])}")
    
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {str(e)[:100]}")
    sys.exit(1)

# Test 3: Input Preparation
print("\n[Test 3/5] Input Preparation...")
try:
    messages = helper.create_message(data['image_frames'].flatten(0, 1))
    processor = helper.get_processor(model.tokenizer)
    
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors='pt',
    )
    
    model_inputs = {
        'tokenized_data': inputs,
        'ego_history_xyz': data['ego_history_xyz'],
        'ego_history_rot': data['ego_history_rot'],
    }
    model_inputs = helper.to_device(model_inputs, 'cuda')
    
    print(f"  PASS: Inputs prepared. Token length: {inputs['input_ids'].shape[1]}")
    
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {str(e)[:100]}")
    sys.exit(1)

# Test 4: Inference
print("\n[Test 4/5] Running Inference...")
print("  Note: This may take a few minutes due to CPU offloading...")
try:
    torch.cuda.manual_seed_all(42)
    
    with torch.autocast('cuda', dtype=torch.float16):
        with torch.no_grad():
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=1,
                max_generation_length=256,
                return_extra=True,
            )
    
    print("  PASS: Inference completed")
    
except torch.cuda.OutOfMemoryError:
    print("  FAIL: Out of GPU memory!")
    sys.exit(1)
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {str(e)[:100]}")
    sys.exit(1)

# Test 5: Metrics Calculation
print("\n[Test 5/5] Metrics Calculation...")
try:
    gt_xy = data['ego_future_xyz'].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = diff.min()
    
    print(f"  PASS: minADE = {min_ade:.4f} meters")
    print(f"  CoC: {extra['cot'][0][:80]}...")
    
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {str(e)[:100]}")
    sys.exit(1)

# Cleanup
print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
print("")
print("Summary:")
print(f"  GPU Memory Used: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
print(f"  minADE: {min_ade:.4f} meters")

# Cleanup
del model
del model_inputs
torch.cuda.empty_cache()
gc.collect()
PYEOF

echo ""
echo "Test completed successfully!"
