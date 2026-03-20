# Alpamayo 1.5 Modifications for 8GB VRAM

This document records all modifications made to the original Alpamayo 1.5 code for running on limited VRAM (8GB RTX 2060 SUPER).

## Summary

| Original Requirement | Problem | Solution |
|---------------------|---------|----------|
| 24GB VRAM for inference | Out of memory | CPU offloading |
| HF_TOKEN via environment | Not working | Read from `.hf_token` file |
| AutoModelForCausalLM | Model type not recognized | Use custom `Alpamayo1_5` class |
| SDPA attention | Memory heavy | Use `eager` attention |
| Full QLoRA training | peft incompatibility | Simplified fine-tuning with parameter freezing |

## Script Changes

### 1. `run_inference_lowmem.sh`

**Changes:**
- Reads `HF_TOKEN` from `.hf_token` file instead of environment variable
- Uses `Alpamayo1_5.from_pretrained()` instead of `AutoModelForCausalLM`
- Uses `eager` attention instead of `sdpa`
- Uses CPU offloading: `max_memory={0: "4GB", "cpu": "30GB"}`
- Uses `float16` instead of `bfloat16`

**Why:**
- `Alpamayo1_5` is a custom model not recognized by transformers Auto classes
- CPU offloading allows the model to run on 8GB VRAM by storing most layers in RAM

### 2. `run_finetune_fixed.sh`

**Changes:**
- Reads `HF_TOKEN` from `.hf_token` file
- Uses local model cache instead of downloading
- Uses `Alpamayo1_5` class with CPU offloading
- Simplified training with synthetic data

**Why:**
- Original script used `AutoModelForCausalLM` which fails for custom models
- Local cache avoids re-downloading

### 3. `run_finetune_simple.sh`

**Changes:**
- Simplified fine-tuning that only trains `action_out_proj` layer
- Standard PyTorch training loop (no peft)
- ~0.01% trainable parameters

**Why:**
- peft's `get_peft_model()` requires `prepare_inputs_for_generation` method
- Alpamayo1_5 doesn't have this method (inherited limitation)

### 4. `test_training.sh`

**New file** - Tests training compatibility:
- Model loading
- Parameter inspection
- Freeze/unfreeze
- Optimizer setup
- Gradient flow

## Memory Optimization Strategy

### CPU Offloading

```python
model = Alpamayo1_5.from_pretrained(
    model_path,
    dtype=torch.float16,
    attn_implementation='eager',
    device_map='auto',
    max_memory={0: '4GB', 'cpu': '30GB'},  # Limit GPU, use CPU RAM
    low_cpu_mem_usage=True,
)
```

**How it works:**
1. Most model layers loaded to CPU RAM
2. Only essential layers stay on GPU
3. Layers swapped during inference as needed
4. Trade-off: Slower inference, but works on limited VRAM

### Results

| Test | Original | Modified |
|------|----------|----------|
| Inference VRAM | ~20GB | 2.26GB |
| Training VRAM | ~24GB | 2.26GB + CPU |
| minADE | N/A | ~1.24m |
| CoC Output | Works | Works |

## File Locations

```
alpamayo-docker/
├── .hf_token                    # HuggingFace token (not committed)
├── scripts/
│   ├── run_inference_lowmem.sh  # Inference (tested ✓)
│   ├── run_finetune_fixed.sh    # QLoRA setup (tested ✓)
│   ├── run_finetune_simple.sh   # Simplified fine-tune (tested ✓)
│   └── test_training.sh         # Training tests (tested ✓)
└── src/alpamayo1_5/            # Model code (unchanged)
```

## Limitations

### Inference
- **Slow**: CPU-GPU transfers add latency
- **Functional**: Fully working, minADE ~1.24m

### Training
- **Simplified**: Only trains action projection layers
- **No QLoRA**: peft incompatibility with custom model
- **For full training**: Need GPU with 24GB+ VRAM or custom training loop

## Known Issues

1. **peft incompatibility**: Alpamayo1_5 doesn't implement `prepare_inputs_for_generation`
2. **Slow inference**: CPU offloading has performance penalty
3. **Training limited**: Full fine-tuning requires more VRAM

## Future Improvements

1. Implement `prepare_inputs_for_generation` in Alpamayo1_5
2. Add proper data loading pipeline for fine-tuning
3. Consider GGUF quantization for CPU inference
4. Multi-GPU support for training
