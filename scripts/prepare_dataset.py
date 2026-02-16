#!/usr/bin/env python3
"""Build a PNG+caption dataset from KANJIDIC2 and KanjiVG XML files."""

from __future__ import annotations

import argparse
import json
import random
import xml.etree.ElementTree as ET
from hashlib import md5
from pathlib import Path

import cairosvg
from tqdm import tqdm


def localname(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def parse_kanjidic_meanings(kanjidic_xml: Path, max_meanings: int) -> dict[str, list[str]]:
    char_to_meanings: dict[str, list[str]] = {}
    tree = ET.parse(kanjidic_xml)
    root = tree.getroot()

    for character in root.findall("character"):
        literal = character.findtext("literal")
        if not literal:
            continue

        meanings: list[str] = []
        for meaning in character.findall(".//meaning"):
            # Keep only default meanings (typically English) and explicit English entries.
            lang = meaning.attrib.get("m_lang")
            if lang is None or lang == "en":
                text = (meaning.text or "").strip()
                if text and text not in meanings:
                    meanings.append(text)
            if len(meanings) >= max_meanings:
                break

        char_to_meanings[literal] = meanings

    return char_to_meanings


def extract_kanji_char(kanji_elem: ET.Element) -> str | None:
    for key in ("id", "{http://www.w3.org/XML/1998/namespace}id"):
        if key in kanji_elem.attrib:
            value = kanji_elem.attrib[key]
            break
    else:
        value = None

    if not value or "_" not in value:
        return None

    try:
        hex_code = value.split("_")[-1]
        return chr(int(hex_code, 16))
    except ValueError:
        return None


def collect_paths(kanji_elem: ET.Element) -> list[str]:
    paths: list[str] = []
    for elem in kanji_elem.iter():
        if localname(elem.tag) == "path":
            d = elem.attrib.get("d")
            if d:
                paths.append(d)
    return paths


def build_svg(paths: list[str], stroke_width: float, margin: float) -> str:
    scale = 109.0 - (2.0 * margin)
    transform = f"translate({margin} {margin}) scale({scale / 109.0:.6f})"

    path_fragments = []
    for d in paths:
        path_fragments.append(
            f'<path d="{d}" fill="none" stroke="#111111" stroke-width="{stroke_width:.3f}" '
            'stroke-linecap="round" stroke-linejoin="round" />'
        )

    body = "\n".join(path_fragments)
    return (
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 109 109\">"
        "<rect width=\"109\" height=\"109\" fill=\"white\"/>"
        f"<g transform=\"{transform}\">{body}</g>"
        "</svg>"
    )


def caption_for_char(char: str, meanings: list[str]) -> str:
    if meanings:
        joined = ", ".join(meanings)
        return (
            f"single centered black Japanese kanji glyph on white background, "
            f"character {char}, meaning: {joined}"
        )
    return (
        f"single centered black Japanese kanji glyph on white background, "
        f"character {char}"
    )


def split_bucket(key: str, val_pct: int) -> str:
    digest = md5(key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "val" if bucket < val_pct else "train"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Kanji image-text dataset")
    parser.add_argument("--kanjidic", type=Path, default=Path("data/raw/kanjidic2.xml"))
    parser.add_argument("--kanjivg", type=Path, default=Path("data/raw/kanjivg-20220427.xml"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/kanji256"))
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants-per-kanji", type=int, default=2)
    parser.add_argument("--val-percent", type=int, default=5)
    parser.add_argument("--max-meanings", type=int, default=2)
    parser.add_argument("--max-items", type=int, default=0, help="0 means no limit")
    args = parser.parse_args()

    random.seed(args.seed)

    output_dir = args.output_dir
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("[info] Parsing KANJIDIC2...")
    char_to_meanings = parse_kanjidic_meanings(args.kanjidic, args.max_meanings)

    print("[info] Parsing KanjiVG and rendering images...")
    tree = ET.parse(args.kanjivg)
    root = tree.getroot()

    train_meta_path = output_dir / "metadata_train.jsonl"
    val_meta_path = output_dir / "metadata_val.jsonl"

    train_count = 0
    val_count = 0
    processed = 0

    with train_meta_path.open("w", encoding="utf-8") as train_meta, val_meta_path.open(
        "w", encoding="utf-8"
    ) as val_meta:
        kanji_nodes = [elem for elem in root.iter() if localname(elem.tag) == "kanji"]

        for kanji_elem in tqdm(kanji_nodes):
            char = extract_kanji_char(kanji_elem)
            if char is None:
                continue

            paths = collect_paths(kanji_elem)
            if not paths:
                continue

            meanings = char_to_meanings.get(char, [])
            caption = caption_for_char(char, meanings)

            for variant_idx in range(args.variants_per_kanji):
                stroke_width = random.uniform(3.2, 4.8)
                margin = random.uniform(5.0, 9.0)

                svg = build_svg(paths, stroke_width=stroke_width, margin=margin)
                codepoint = f"{ord(char):05x}"
                image_name = f"{codepoint}_{variant_idx}.png"
                image_path = images_dir / image_name

                cairosvg.svg2png(
                    bytestring=svg.encode("utf-8"),
                    write_to=str(image_path),
                    output_width=args.resolution,
                    output_height=args.resolution,
                )

                rel_path = f"images/{image_name}"
                record = {
                    "file_name": rel_path,
                    "text": caption,
                    "char": char,
                    "meanings": meanings,
                }

                split = split_bucket(f"{char}:{variant_idx}", args.val_percent)
                if split == "val":
                    val_meta.write(json.dumps(record, ensure_ascii=False) + "\n")
                    val_count += 1
                else:
                    train_meta.write(json.dumps(record, ensure_ascii=False) + "\n")
                    train_count += 1

                processed += 1
                if args.max_items > 0 and processed >= args.max_items:
                    break

            if args.max_items > 0 and processed >= args.max_items:
                break

    summary = {
        "resolution": args.resolution,
        "variants_per_kanji": args.variants_per_kanji,
        "train_examples": train_count,
        "val_examples": val_count,
        "output_dir": str(output_dir),
    }

    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[ok] Dataset written to: {output_dir}")
    print(f"[ok] Train examples: {train_count}")
    print(f"[ok] Val examples: {val_count}")


if __name__ == "__main__":
    main()
