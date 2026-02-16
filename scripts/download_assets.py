#!/usr/bin/env python3
"""Download Kanji datasets and base Stable Diffusion checkpoint."""

from __future__ import annotations

import argparse
import gzip
import shutil
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download

KANJIDIC2_URL = "https://www.edrdg.org/kanjidic/kanjidic2.xml.gz"
KANJIVG_URL = "https://github.com/KanjiVG/kanjivg/releases/download/r20220427/kanjivg-20220427.xml.gz"


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"[skip] {output_path} already exists")
        return

    print(f"[download] {url} -> {output_path}")
    with urllib.request.urlopen(url) as response, output_path.open("wb") as target:
        shutil.copyfileobj(response, target)


def gunzip_file(gz_path: Path, xml_path: Path) -> None:
    if xml_path.exists() and xml_path.stat().st_size > 0:
        print(f"[skip] {xml_path} already exists")
        return

    print(f"[extract] {gz_path} -> {xml_path}")
    with gzip.open(gz_path, "rb") as source, xml_path.open("wb") as target:
        shutil.copyfileobj(source, target)


def sanitize_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "--")


def download_model(repo_id: str, checkpoints_dir: Path, hf_token: str | None) -> Path:
    model_dir = checkpoints_dir / sanitize_repo_id(repo_id)
    model_dir.mkdir(parents=True, exist_ok=True)

    if any(model_dir.iterdir()):
        print(f"[skip] Model directory already has files: {model_dir}")
        return model_dir

    print(f"[download] Hugging Face model: {repo_id} -> {model_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
        token=hf_token,
        resume_download=True,
    )
    return model_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Kanji data and SD checkpoint")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to store raw data files",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory to store downloaded model checkpoints",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Base model repo ID on Hugging Face",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional Hugging Face token for gated/private models",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Only download the Kanji datasets",
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    kanjidic_gz = raw_dir / "kanjidic2.xml.gz"
    kanjidic_xml = raw_dir / "kanjidic2.xml"
    kanjivg_gz = raw_dir / "kanjivg-20220427.xml.gz"
    kanjivg_xml = raw_dir / "kanjivg-20220427.xml"

    download_file(KANJIDIC2_URL, kanjidic_gz)
    download_file(KANJIVG_URL, kanjivg_gz)
    gunzip_file(kanjidic_gz, kanjidic_xml)
    gunzip_file(kanjivg_gz, kanjivg_xml)

    if not args.skip_model:
        model_dir = download_model(args.model_id, args.checkpoints_dir, args.hf_token)
        print(f"[ok] model downloaded at: {model_dir}")

    print("[ok] downloads complete")


if __name__ == "__main__":
    main()
