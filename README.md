# OptiSQL Reproduction (FULLFT-first)

This repository contains a runnable reproduction scaffold for **OptiSQL: Executable SQL Generation from Optical Tokens** (arXiv:2601.13695), with emphasis on the **FULLFT** setting:

- jointly fine-tune visual encoder + SQL decoder
- enable render-time augmentations (style variation + transpose)
- keep evaluation protocol aligned with EXAcc / EX-Can / robustness analysis

## Project structure

- `configs/`: default config templates
- `scripts/`: end-to-end pipeline scripts
  - `0_download_spider2_snow.py`
  - `1_build_visualized_dataset.py`
  - `2_train_fullft.py`
  - `3_eval_fullft.py`
- `optisql/`: core implementation
  - `render/`: SQL table extraction, grid building, HTML render
  - `models/`: encoder adapter + decoder composition
  - `train/`: optimization and training loop
  - `eval/`: canonicalization, execution evaluation, metrics

## Install

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

## Quickstart

1) Prepare Spider2-snow metadata

```bash
python scripts/0_download_spider2_snow.py --data_root data/spider2-snow --auto_download
```

2) Build visualized samples

```bash
python scripts/1_build_visualized_dataset.py \
  --data_root data/spider2-snow \
  --output_root data/visualized \
  --split train \
  --style_pool_size 4 \
  --transpose_prob 0.3
```

3) Train FULLFT

```bash
python scripts/2_train_fullft.py \
  --manifest data/visualized/manifest_train.jsonl \
  --images_root data/visualized \
  --decoder_name gpt2 \
  --max_steps 1000 \
  --batch_size 2 \
  --grad_accum 1 \
  --save_every 200 \
  --output_dir checkpoints/fullft
```

4) Evaluate

```bash
python scripts/3_eval_fullft.py \
  --manifest data/visualized/manifest_train.jsonl \
  --images_root data/visualized \
  --decoder_name gpt2 \
  --checkpoint checkpoints/fullft/last.pt
```

## Current status

This repo now includes concrete runnable implementations for data pipeline, rendering, training loop, and evaluation utilities. You can iterate from this baseline to swap in a real DeepSeek-OCR adapter and full Spider2.0-Snow experimental protocol.
