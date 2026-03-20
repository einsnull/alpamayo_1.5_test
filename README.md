# Alpamayo 1.5 Docker Setup

Docker setup for NVlabs/Alpamayo 1.5, optimized for **7GB VRAM** with QLoRA fine-tuning support.

## Project Structure

```
alpamayo-docker/
├── Dockerfile                 # Main Docker image definition
├── docker-compose.yml         # Docker Compose configuration
├── .env.example               # Environment variables template
├── configs/
│   ├── accelerate_config.yaml  # Accelerate training config
│   └── lora_config.yaml       # QLoRA configuration
└── scripts/
    ├── 01_setup.sh            # Initial setup & model download
    ├── download_data.sh       # Download sample dataset
    ├── run_inference_lowmem.sh # Low-memory inference
    ├── run_finetune.sh        # QLoRA fine-tuning
    ├── train_lora.py          # Detailed training script
    ├── train_accelerate.py    # Accelerate-based training
    └── merge_lora.py           # Merge LoRA weights
```

## Quick Start

### 1. Prerequisites

- NVIDIA GPU with CUDA 12.x
- Docker & Docker Compose with NVIDIA Container Toolkit
- HuggingFace account with access to:
  - `nvidia/Alpamayo-1.5-10B` model
  - `PhysicalAI/PhysicalAI-Autonomous-Vehicles` dataset

### 2. Setup Environment

```bash
# Set HuggingFace token
export HF_TOKEN=your_token_here

# Copy environment file
cd alpamayo-docker
cp .env.example .env
# Edit .env and add your HF_TOKEN
```

### 3. Build Docker Image

```bash
cd alpamayo-docker
docker compose build alpamayo
```

### 4. Run Container

```bash
# Interactive mode
docker compose run --rm alpamayo

# Or with environment file
docker compose --env-file .env run --rm alpamayo
```

### 5. Initial Setup (Inside Container)

```bash
# Authenticate and download model
./scripts/01_setup.sh

# Download sample data
./scripts/download_data.sh
```

## Usage

### Inference (Optimized for 7GB VRAM)

```bash
./scripts/run_inference_lowmem.sh
```

Memory optimizations applied:
- SDPA attention (instead of Flash Attention)
- CPU offloading for large layers
- Minimal batch size (1 sample)
- Single trajectory sampling

### QLoRA Fine-tuning

```bash
./scripts/run_finetune.sh
```

Or use the detailed training script:

```bash
python scripts/train_accelerate.py
```

### Merge LoRA Weights

After fine-tuning, merge LoRA weights with base model:

```bash
python scripts/merge_lora.py --lora_path /workspace/alpamayo/outputs/lora_checkpoint/final
```

## Memory Optimization Tips for 7GB VRAM

If you encounter Out of Memory errors:

1. **Reduce LoRA rank** (in `lora_config.yaml`):
   ```yaml
   lora:
     r: 8  # Default is 64, reduce to 8-16 for very low memory
   ```

2. **Enable CPU offloading**:
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       ...,
       device_map="auto",
       max_memory={0: "5GB", "cpu": "30GB"}  # Limit GPU usage
   )
   ```

3. **Reduce sequence length**:
   ```yaml
   max_length: 512  # Instead of 2048
   ```

4. **Use gradient checkpointing**:
   ```python
   model.gradient_checkpointing_enable()
   ```

5. **Process samples one at a time**:
   ```bash
   export BATCH_SIZE=1
   export GRAD_ACCUM=8  # Increase accumulation for larger effective batch
   ```

## LoRA Fine-tuning Details

QLoRA (Quantized Low-Rank Adaptation) allows fine-tuning large models on limited VRAM by:

1. **4-bit Quantization (NF4)**: Reduces model weights to 4-bit precision (~4x memory reduction)
2. **LoRA Adapters**: Only trains small adapter matrices (~0.1% of parameters)
3. **Gradient Checkpointing**: Trades compute for memory

### What's Fine-tuned:
- LLM backbone attention layers (Q, K, V, O projections)
- FFN layers (gate, up, down projections)

### What's Frozen:
- Vision encoder (frozen for memory efficiency)
- Embedding layers
- Trajectory diffusion expert

## Configuration

### Environment Variables

```bash
HF_TOKEN=your_token_here
HF_HOME=/root/.cache/huggingface
OUTPUT_DIR=/workspace/alpamayo/outputs
LORA_RANK=16          # LoRA rank (8-64)
BATCH_SIZE=1           # Batch size per device
GRAD_ACCUM=4           # Gradient accumulation steps
LEARNING_RATE=2e-4     # Learning rate
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

1. Reduce LoRA rank: `r=8`
2. Enable CPU offloading
3. Reduce batch size to 1
4. Clear GPU cache: `torch.cuda.empty_cache()`

### Flash Attention Errors

The setup uses SDPA (Scaled Dot Product Attention) instead of Flash Attention for compatibility. If you have CUDA Toolkit installed and want to use Flash Attention:

```bash
# Install flash-attn manually
uv sync --active
```

### Model Download Fails

1. Verify HuggingFace token: `echo $HF_TOKEN`
2. Check model access: https://huggingface.co/nvidia/Alpamayo-1.5-10B
3. Login: `huggingface-cli login`

## License

This setup is for research purposes. Alpamayo 1.5 is licensed under Apache 2.0. See the original repository for details.
