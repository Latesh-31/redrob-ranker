"""Convert a Hugging Face / safetensors model to FP16 and save as safetensors.
Usage:
    python scripts/quantize_model_fp16.py --model-dir models/bge-small-en-v1.5 --output-dir models/bge-small-en-v1.5-fp16
"""
import argparse
import os
import shutil
import torch
from transformers import AutoModel, AutoTokenizer


def sizeof(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def human(n):
    for unit in ['B','KB','MB','GB']:
        if n < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', required=True)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()

    src = args.model_dir
    dst = args.output_dir
    if not os.path.isdir(src):
        raise SystemExit(f"model-dir not found: {src}")
    if os.path.exists(dst):
        raise SystemExit(f"output-dir already exists: {dst}")

    print('Source size:', human(sizeof(src)))

    cpu_only = not torch.cuda.is_available()
    print('CUDA available:', not cpu_only)

    print('Loading model...')
    if cpu_only:
        model = AutoModel.from_pretrained(src, low_cpu_mem_usage=True)
        model = model.half()
    else:
        model = AutoModel.from_pretrained(src, torch_dtype=torch.float16, device_map='auto')

    os.makedirs(dst, exist_ok=True)
    print('Saving quantized model to', dst)
    model.save_pretrained(dst, safe_serialization=True)

    # copy tokenizer files if present
    try:
        tok = AutoTokenizer.from_pretrained(src)
        tok.save_pretrained(dst)
    except Exception:
        # ignore if no tokenizer
        pass

    print('Output size:', human(sizeof(dst)))
    print('Done.')


if __name__ == '__main__':
    main()
