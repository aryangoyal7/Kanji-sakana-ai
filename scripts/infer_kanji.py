#!/usr/bin/env python3
"""Inference script for Kanji generation with a LoRA fine-tuned SD model."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


def build_prompt(text: str) -> str:
    return (
        "single centered black Japanese kanji glyph on white background, "
        f"kanji meaning or character: {text}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Kanji image from text")
    parser.add_argument("--model-path", type=Path, required=True, help="Base SD model path")
    parser.add_argument("--lora-path", type=Path, required=True, help="Trained LoRA folder")
    parser.add_argument("--text", type=str, required=True, help="Input text prompt, e.g. 'water' or '水'")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/inference"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(args.model_path, torch_dtype=dtype)
    pipe.load_lora_weights(str(args.lora_path))
    pipe = pipe.to(device)

    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    generator = torch.Generator(device=device).manual_seed(args.seed)
    prompt = build_prompt(args.text)

    result = pipe(
        prompt=prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )

    image = result.images[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output_dir / f"kanji_{timestamp}.png"
    image.save(output_path)

    print(f"[ok] Prompt: {prompt}")
    print(f"[ok] Saved image: {output_path}")


if __name__ == "__main__":
    main()
