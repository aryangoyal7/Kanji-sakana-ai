# Kanji Sakana AI

LoRA fine-tuning pipeline for Stable Diffusion 1.5 on Kanji data (KANJIDIC2 + KanjiVG), with 256x256 training images, checkpoint saves, and training-curve logging.

## 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
mkdir -p logs
```

## 2) Download data + base model

```bash
python scripts/download_assets.py --model-id runwayml/stable-diffusion-v1-5
```

## 3) Build dataset

```bash
python scripts/prepare_dataset.py \
  --kanjidic data/raw/kanjidic2.xml \
  --kanjivg data/raw/kanjivg-20220427.xml \
  --output-dir data/processed/kanji256 \
  --resolution 256 \
  --variants-per-kanji 2
```

## 4) Train (48GB GPU defaults)

Uses `train_batch_size=24` and saves LoRA checkpoints every `500` steps.

```bash
nohup python -u scripts/train_lora.py \
  --model-path checkpoints/runwayml--stable-diffusion-v1-5 \
  --train-metadata data/processed/kanji256/metadata_train.jsonl \
  --data-root data/processed/kanji256 \
  --output-dir outputs/kanji_lora \
  --log-dir logs/kanji_lora \
  --resolution 256 \
  --train-batch-size 24 \
  --num-epochs 12 \
  --learning-rate 1e-4 \
  --save-every 500 \
  --rank 16 \
  > logs/train_nohup.log 2>&1 &
```

Check live logs:

```bash
tail -f logs/train_nohup.log
```

## 5) Training logs and checkpoints

- TensorBoard scalars: `logs/kanji_lora`
- CSV metrics: `outputs/kanji_lora/train_metrics.csv`
- Loss curve PNG: `outputs/kanji_lora/training_curve.png`
- Intermediate LoRA checkpoints: `outputs/kanji_lora/checkpoint-*`
- Final LoRA weights: `outputs/kanji_lora/lora`

Run TensorBoard:

```bash
tensorboard --logdir logs/kanji_lora --port 6006
```

## 6) Inference

```bash
python scripts/infer_kanji.py \
  --model-path checkpoints/runwayml--stable-diffusion-v1-5 \
  --lora-path outputs/kanji_lora/lora \
  --text "water" \
  --steps 30 \
  --guidance-scale 7.0
```

## 7) Full pipeline with nohup

```bash
nohup bash scripts/run_full_pipeline.sh > logs/pipeline_nohup.log 2>&1 &
```
