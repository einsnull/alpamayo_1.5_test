# Alpamayo 1.5 Docker Setup

Docker setup for NVlabs/Alpamayo 1.5, optimized for **8GB VRAM** (RTX 2060 SUPER) with CPU offloading.

## Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA with 8GB+ VRAM |
| RAM | 32GB+ recommended |
| Storage | ~30GB for model + data |
| CUDA | 12.x |

## Project Structure

```
alpamayo-docker/
├── .hf_token              # HuggingFace token (create this file)
├── .gitignore             # Ignores .hf_token and other sensitive files
├── Dockerfile              # Main Docker image definition
├── docker-compose.yml      # Docker Compose configuration
├── configs/
│   ├── accelerate_config.yaml  # Accelerate training config
│   └── lora_config.yaml       # QLoRA configuration
└── scripts/
    ├── 01_setup.sh            # Initial setup & model download
    ├── download_data.sh       # Download sample dataset
    ├── run_inference_lowmem.sh # Low-memory inference
    ├── test_inference.sh       # Test script (5 tests)
    ├── run_finetune.sh        # QLoRA fine-tuning
    ├── train_lora.py          # Detailed training script
    ├── train_accelerate.py    # Accelerate-based training
    └── merge_lora.py          # Merge LoRA weights
```

## Quick Start

### 1. Prerequisites

- NVIDIA GPU with CUDA 12.x
- Docker & Docker Compose with NVIDIA Container Toolkit
- HuggingFace account with access to:
  - `nvidia/Alpamayo-1.5-10B` model
  - `nvidia/PhysicalAI-Autonomous-Vehicles` dataset

### 2. Setup Environment

```bash
cd alpamayo-docker

# Create .hf_token file with your HuggingFace token
echo "hf_your_token_here" > .hf_token

# Build Docker image
docker compose build alpamayo
```

### 3. Initial Setup (Inside Container)

```bash
# Start container
docker compose run --rm alpamayo bash

# Inside container: Authenticate and download model
./scripts/01_setup.sh

# Download sample data
./scripts/download_data.sh
```

### 4. Run Tests

```bash
# Run all inference tests
./scripts/test_inference.sh
```

Expected output:
```
==========================================
Alpamayo 1.5 - Test Script
==========================================

[Test 1/5] Model Loading with CPU Offloading...
  PASS: Model loaded. GPU Memory: 2.26 GB
...
ALL TESTS PASSED!
```

## Usage

### Test Inference Script

```bash
./scripts/test_inference.sh
```

This runs 5 tests:
1. Model Loading with CPU Offloading
2. Data Loading
3. Input Preparation
4. Running Inference
5. Metrics Calculation

### Low-Memory Inference

```bash
./scripts/run_inference_lowmem.sh
```

### QLoRA Fine-tuning

```bash
./scripts/run_finetune.sh
```

### Merge LoRA Weights

After fine-tuning, merge LoRA weights with base model:

```bash
python scripts/merge_lora.py --lora_path /workspace/alpamayo/outputs/lora_checkpoint/final
```

## Memory Optimization Strategy

This setup uses **CPU offloading** to run Alpamayo 1.5 on limited VRAM:

| Setting | Value | Purpose |
|---------|-------|---------|
| GPU Memory Limit | 4GB | Restrict GPU usage |
| CPU Memory | 30GB | Store offloaded layers |
| Dtype | float16 | Reduce memory footprint |
| Attention | eager | Avoid Flash Attention overhead |

### How It Works

1. **Model Loading**: Most layers loaded to CPU, only essential layers on GPU
2. **Inference**: Layers swapped between CPU/GPU as needed
3. **Trade-off**: Slower inference speed, but works on 8GB VRAM

### Results

| Metric | Value |
|--------|-------|
| GPU Memory | ~2.26 GB |
| minADE | ~1.24 meters |
| CoC Output | Chain-of-Causation reasoning |

## Configuration

### Environment Variables

```bash
HF_TOKEN=your_token_here          # Stored in .hf_token file
HF_HOME=/root/.cache/huggingface  # Model cache location
OUTPUT_DIR=/workspace/alpamayo/outputs
```

### LoRA Configuration

Edit `configs/lora_config.yaml`:

```yaml
lora:
  r: 16                # Rank - lower = less memory
  lora_alpha: 32       # Scaling factor
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
```

## Troubleshooting

### Out of Memory (OOM)

1. Verify CPU offloading is enabled
2. Reduce batch size to 1
3. Clear GPU cache: `torch.cuda.empty_cache()`

### Model Download Fails

1. Verify .hf_token file exists
2. Check HuggingFace access: https://huggingface.co/nvidia/Alpamayo-1.5-10B
3. Dataset access: https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles

### Slow Inference

This is expected with CPU offloading. To speed up:
- Use a GPU with more VRAM (24GB+)
- Disable CPU offloading if you have enough VRAM

## License

This setup is for research purposes. Alpamayo 1.5 is licensed under Apache 2.0. See the original repository for details.
