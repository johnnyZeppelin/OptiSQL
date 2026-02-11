# OptiSQL Reproduction (FULLFT-first)

This repository provides a pipeline for OptiSQL (arXiv:2601.13695), including:

- dataset preparation (real Spider2-snow and a toy fallback)
- table rendering to images (Playwright; automatic PIL fallback)
- visualized manifest construction with style/transpose augmentation
- FULLFT training loop (encoder + decoder)
- execution-based evaluation (EXAcc / EX-Can utilities)

## Install

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

> If Playwright/Chromium is unavailable, rendering automatically falls back to PIL so the pipeline remains runnable.

## Fast end-to-end run (toy data, guaranteed local runnable)

```bash
python scripts/0_make_toy_data.py --data_root data/toy

python scripts/1_build_visualized_dataset.py \
  --data_root data/toy \
  --output_root data/visualized_toy \
  --split train \
  --style_pool_size 2 \
  --transpose_prob 0.3

python scripts/2_train_fullft.py \
  --manifest data/visualized_toy/manifest_train.jsonl \
  --images_root data/visualized_toy \
  --decoder_name sshleifer/tiny-gpt2 \
  --max_steps 2 \
  --batch_size 1 \
  --grad_accum 1 \
  --save_every 1 \
  --output_dir checkpoints/toy

python scripts/3_eval_fullft.py \
  --manifest data/visualized_toy/manifest_train.jsonl \
  --images_root data/visualized_toy \
  --decoder_name sshleifer/tiny-gpt2 \
  --checkpoint checkpoints/toy/last.pt
```

## Real-data run (Spider2-snow)

```bash
python scripts/0_download_spider2_snow.py --data_root data/spider2-snow --auto_download

python scripts/1_build_visualized_dataset.py \
  --data_root data/spider2-snow \
  --output_root data/visualized \
  --split train \
  --style_pool_size 4 \
  --transpose_prob 0.3

python scripts/2_train_fullft.py \
  --manifest data/visualized/manifest_train.jsonl \
  --images_root data/visualized \
  --decoder_name gpt2 \
  --max_steps 1000 \
  --batch_size 2 \
  --grad_accum 1 \
  --save_every 200 \
  --output_dir checkpoints/fullft

python scripts/3_eval_fullft.py \
  --manifest data/visualized/manifest_train.jsonl \
  --images_root data/visualized \
  --decoder_name gpt2 \
  --checkpoint checkpoints/fullft/last.pt
```

## Notes

- `scripts/0_make_toy_data.py` is included specifically to guarantee full pipeline validation in any clean environment.
- You can later replace the placeholder encoder adapter with a production DeepSeek-OCR integration while keeping the same train/eval interfaces.
