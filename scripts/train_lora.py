#!/usr/bin/env python3
"""LoRA fine-tuning for Kanji text-to-image generation using Stable Diffusion."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.models import attention_processor as attn_proc
from PIL import Image
from torch.optim import AdamW
from torch.nn import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


@dataclass
class TrainConfig:
    model_path: str
    train_metadata: str
    data_root: str
    output_dir: str
    log_dir: str
    resolution: int
    train_batch_size: int
    learning_rate: float
    num_epochs: int
    max_steps: int
    save_every: int
    rank: int
    seed: int
    mixed_precision: str


class KanjiDataset(Dataset):
    def __init__(self, metadata_file: Path, data_root: Path, tokenizer: CLIPTokenizer, resolution: int):
        self.records = []
        with metadata_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

        self.data_root = data_root
        self.tokenizer = tokenizer
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[idx]
        image_path = self.data_root / record["file_name"]
        image = Image.open(image_path).convert("RGB")

        pixel_values = self.image_transform(image)
        tokenized = self.tokenizer(
            record["text"],
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized.input_ids[0],
        }


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    return {
        "pixel_values": pixel_values.to(memory_format=torch.contiguous_format).float(),
        "input_ids": input_ids,
    }


def extract_parameters_from_object(obj: Any) -> list[Parameter]:
    params: list[Parameter] = []
    seen_param_ids: set[int] = set()
    seen_obj_ids: set[int] = set()

    def add_param(param: Parameter) -> None:
        if id(param) in seen_param_ids:
            return
        seen_param_ids.add(id(param))
        params.append(param)

    def walk(node: Any) -> None:
        node_id = id(node)
        if node_id in seen_obj_ids:
            return
        seen_obj_ids.add(node_id)

        if isinstance(node, Parameter):
            add_param(node)
            return

        if isinstance(node, torch.nn.Module):
            for p in node.parameters():
                add_param(p)
            return

        if isinstance(node, dict):
            for v in node.values():
                walk(v)
            return

        if isinstance(node, (list, tuple, set)):
            for v in node:
                walk(v)
            return

        if hasattr(node, "__dict__"):
            for v in vars(node).values():
                walk(v)

    walk(obj)
    return params


def _instantiate_lora_cls(lora_cls: type, hidden_size: int, cross_attention_dim: int | None, rank: int):
    sig = inspect.signature(lora_cls.__init__)
    names = set(sig.parameters.keys())
    kw = {}
    if "hidden_size" in names:
        kw["hidden_size"] = hidden_size
    if "cross_attention_dim" in names:
        kw["cross_attention_dim"] = cross_attention_dim
    if "rank" in names:
        kw["rank"] = rank
    elif "r" in names:
        kw["r"] = rank
    if "network_alpha" in names:
        kw["network_alpha"] = rank
    elif "lora_alpha" in names:
        kw["lora_alpha"] = rank
    if "attention_op" in names:
        kw["attention_op"] = None

    return lora_cls(**kw)


def _candidate_lora_class_names(base_processor: Any) -> list[str]:
    base_name = base_processor.__class__.__name__
    candidates: list[str] = []

    if "AddedKV" in base_name:
        candidates.append("LoRAAttnAddedKVProcessor")
    if "XFormers" in base_name:
        candidates.append("LoRAXFormersAttnProcessor")
    if hasattr(F, "scaled_dot_product_attention"):
        candidates.append("LoRAAttnProcessor2_0")

    candidates.extend(
        [
            "LoRAAttnProcessor",
            "LoRAAttnProcessor2_0",
            "LoRAAttnAddedKVProcessor",
            "LoRAXFormersAttnProcessor",
        ]
    )

    seen: set[str] = set()
    ordered: list[str] = []
    for name in candidates:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def build_lora_attn_processor(
    base_processor: Any, hidden_size: int, cross_attention_dim: int | None, rank: int
):
    errors: list[str] = []
    for cls_name in _candidate_lora_class_names(base_processor):
        lora_cls = getattr(attn_proc, cls_name, None)
        if lora_cls is None:
            continue
        try:
            processor = _instantiate_lora_cls(lora_cls, hidden_size, cross_attention_dim, rank)
            if extract_parameters_from_object(processor):
                return processor
            errors.append(f"{cls_name}: no trainable params")
        except Exception as exc:
            errors.append(f"{cls_name}: {exc}")

    raise TypeError(
        "Could not initialize a trainable LoRA attention processor with this diffusers version. "
        f"Tried: {', '.join(errors[:6])}"
    )


def collect_trainable_lora_parameters(unet: UNet2DConditionModel) -> list[Parameter]:
    params: list[Parameter] = []
    seen_param_ids: set[int] = set()

    for processor in unet.attn_processors.values():
        for param in extract_parameters_from_object(processor):
            if id(param) in seen_param_ids:
                continue
            if not param.requires_grad:
                param.requires_grad_(True)
            seen_param_ids.add(id(param))
            params.append(param)

    if params:
        return params

    for name, param in unet.named_parameters():
        if "lora" in name.lower():
            param.requires_grad_(True)
            params.append(param)

    return params


def make_lora_layers(unet: UNet2DConditionModel, rank: int) -> list[Parameter]:
    lora_attn_procs = {}
    for name, base_processor in unet.attn_processors.items():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim

        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            raise ValueError(f"Unexpected attention processor name: {name}")

        lora_attn_procs[name] = build_lora_attn_processor(
            base_processor=base_processor,
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=rank,
        )

    unet.set_attn_processor(lora_attn_procs)
    trainable_params = collect_trainable_lora_parameters(unet)
    if not trainable_params:
        raise RuntimeError(
            "No trainable LoRA parameters were found after setting attention processors. "
            "Install a compatible diffusers version (recommended >=0.20)."
        )
    return trainable_params


def save_metrics(metrics: list[tuple[int, float, float]], output_dir: Path) -> None:
    csv_path = output_dir / "train_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "lr"])
        writer.writerows(metrics)

    if not metrics:
        return

    steps = [m[0] for m in metrics]
    losses = [m[1] for m in metrics]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, losses, label="train_loss", linewidth=1.5)
    plt.xlabel("Step")
    plt.ylabel("MSE Loss")
    plt.title("Kanji LoRA Fine-tuning Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_curve.png", dpi=160)
    plt.close()


def autocast_ctx(use_fp16: bool):
    if use_fp16:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LoRA on Kanji dataset")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--train-metadata", type=Path, default=Path("data/processed/kanji256/metadata_train.jsonl"))
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/kanji256"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/kanji_lora"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs/kanji_lora"))
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-epochs", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means train for all epochs")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mixed-precision",
        type=str,
        choices=["no", "fp16"],
        default="fp16",
        help="Use fp16 autocast on CUDA",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = args.mixed_precision == "fp16" and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(args.log_dir))

    print("[info] Loading model components...")
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.model_path, subfolder="unet")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    lora_parameters = make_lora_layers(unet, rank=args.rank)

    vae.to(device)
    text_encoder.to(device)
    unet.to(device)

    optimizer = AdamW(lora_parameters, lr=args.learning_rate)

    print("[info] Loading dataset...")
    train_dataset = KanjiDataset(
        metadata_file=args.train_metadata,
        data_root=args.data_root,
        tokenizer=tokenizer,
        resolution=args.resolution,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
    )

    max_steps = args.max_steps if args.max_steps > 0 else len(train_loader) * args.num_epochs
    global_step = 0
    metrics: list[tuple[int, float, float]] = []

    print(f"[info] Starting training on {device} for up to {max_steps} steps")
    for epoch in range(args.num_epochs):
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.num_epochs}")

        for batch in progress:
            if global_step >= max_steps:
                break

            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                encoder_hidden_states = text_encoder(input_ids)[0]

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=device,
                dtype=torch.long,
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with autocast_ctx(use_fp16):
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unsupported prediction type: {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            current_loss = float(loss.detach().cpu().item())
            current_lr = float(optimizer.param_groups[0]["lr"])
            metrics.append((global_step, current_loss, current_lr))

            writer.add_scalar("train/loss", current_loss, global_step)
            writer.add_scalar("train/lr", current_lr, global_step)

            progress.set_postfix(step=global_step, loss=f"{current_loss:.4f}")

            if global_step % args.save_every == 0:
                ckpt_dir = args.output_dir / f"checkpoint-{global_step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                unet.save_attn_procs(str(ckpt_dir))

        if global_step >= max_steps:
            break

    lora_out = args.output_dir / "lora"
    lora_out.mkdir(parents=True, exist_ok=True)
    unet.save_attn_procs(str(lora_out))

    config = TrainConfig(
        model_path=str(args.model_path),
        train_metadata=str(args.train_metadata),
        data_root=str(args.data_root),
        output_dir=str(args.output_dir),
        log_dir=str(args.log_dir),
        resolution=args.resolution,
        train_batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_steps=max_steps,
        save_every=args.save_every,
        rank=args.rank,
        seed=args.seed,
        mixed_precision=args.mixed_precision,
    )

    with (args.output_dir / "train_config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    save_metrics(metrics, args.output_dir)
    writer.close()

    print(f"[ok] Finished training at step {global_step}")
    print(f"[ok] LoRA weights: {lora_out}")
    print(f"[ok] TensorBoard logs: {args.log_dir}")
    print(f"[ok] CSV metrics + curve: {args.output_dir}")


if __name__ == "__main__":
    main()
