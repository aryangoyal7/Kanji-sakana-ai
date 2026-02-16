#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p "$ROOT_DIR/logs"

MODEL_ID="${MODEL_ID:-runwayml/stable-diffusion-v1-5}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/checkpoints/${MODEL_ID//\//--}}"
RESOLUTION="${RESOLUTION:-256}"

python scripts/download_assets.py --model-id "$MODEL_ID"

python scripts/prepare_dataset.py \
  --kanjidic "$ROOT_DIR/data/raw/kanjidic2.xml" \
  --kanjivg "$ROOT_DIR/data/raw/kanjivg-20220427.xml" \
  --output-dir "$ROOT_DIR/data/processed/kanji${RESOLUTION}" \
  --resolution "$RESOLUTION" \
  --variants-per-kanji 2

python scripts/train_lora.py \
  --model-path "$MODEL_DIR" \
  --train-metadata "$ROOT_DIR/data/processed/kanji${RESOLUTION}/metadata_train.jsonl" \
  --data-root "$ROOT_DIR/data/processed/kanji${RESOLUTION}" \
  --output-dir "$ROOT_DIR/outputs/kanji_lora" \
  --log-dir "$ROOT_DIR/logs/kanji_lora" \
  --resolution "$RESOLUTION" \
  --train-batch-size 24 \
  --num-epochs 12 \
  --learning-rate 1e-4 \
  --save-every 500 \
  --rank 16

echo "Training done. TensorBoard: tensorboard --logdir '$ROOT_DIR/logs/kanji_lora'"
