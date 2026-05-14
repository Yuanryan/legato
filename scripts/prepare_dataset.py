"""
Convert a folder of raw ABC + PNG files into a HuggingFace DatasetDict
saved to disk (Arrow format), ready for use with train_vision_lora.py.

Expected input layout:
    <input_dir>/
        abc_files/
            score_00001.abc
            score_00002.abc
            ...
        png_files/
            score_00001_01.png   ← page 1 of score 00001
            score_00001_02.png   ← page 2 of score 00001
            score_00002_01.png
            ...

Output: a DatasetDict with train / val / test splits, each example having:
    image          PIL Image  (pages concatenated vertically)
    transcription  str        (full ABC text)
    filename       str        (score ID, e.g. "score_00001")

Usage:
    python scripts/prepare_dataset.py \
        --input_dir  music_dataset_10k_v2 \
        --output_dir datasets/music_10k
"""

import argparse
import random
import re
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Value
from datasets import Image as HFImage
from PIL import Image


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",   required=True,
                   help="Root folder containing abc_files/ and png_files/")
    p.add_argument("--output_dir",  required=True,
                   help="Destination for save_to_disk() output")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio",   type=float, default=0.1)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-score loading
# ---------------------------------------------------------------------------

def load_example(score_id: str, abc_dir: Path, png_dir: Path) -> dict | None:
    abc_path = abc_dir / f"{score_id}.abc"
    if not abc_path.exists():
        return None

    transcription = abc_path.read_text(encoding="utf-8").strip()

    page_paths = sorted(
        png_dir.glob(f"{score_id}_*.png"),
        key=lambda p: int(re.search(r"_(\d+)\.png$", p.name).group(1)),
    )
    if not page_paths:
        return None

    pages = [Image.open(p).convert("RGB") for p in page_paths]

    if len(pages) == 1:
        image = pages[0]
    else:
        w = max(p.width for p in pages)
        h = sum(p.height for p in pages)
        image = Image.new("RGB", (w, h), color=(255, 255, 255))
        y = 0
        for page in pages:
            image.paste(page, (0, y))
            y += page.height

    return {"image": image, "transcription": transcription, "filename": score_id}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    abc_dir   = input_dir / "abc_files"
    png_dir   = input_dir / "png_files"

    if not abc_dir.is_dir() or not png_dir.is_dir():
        sys.exit(f"Expected abc_files/ and png_files/ inside {input_dir}")

    # Discover all score IDs
    score_ids = sorted(p.stem for p in abc_dir.glob("*.abc"))
    print(f"Found {len(score_ids)} ABC files")

    # Shuffle and split by ID (no data in memory yet)
    rng = random.Random(args.seed)
    rng.shuffle(score_ids)

    n        = len(score_ids)
    n_train  = int(n * args.train_ratio)
    n_val    = int(n * args.val_ratio)

    id_splits = {
        "train": score_ids[:n_train],
        "val":   score_ids[n_train : n_train + n_val],
        "test":  score_ids[n_train + n_val :],
    }
    for name, ids in id_splits.items():
        print(f"  {name}: {len(ids)} scores")

    features = Features({
        "image":         HFImage(),
        "transcription": Value("string"),
        "filename":      Value("string"),
    })

    # Build each split with a generator so only one image is in memory at a time
    def make_generator(ids):
        def _gen():
            for i, score_id in enumerate(ids):
                if i % 500 == 0:
                    print(f"    {i}/{len(ids)} ...")
                example = load_example(score_id, abc_dir, png_dir)
                if example is None:
                    print(f"    WARNING: skipping {score_id} (missing files)")
                    continue
                yield example
        return _gen

    dataset_dict = {}
    for split_name, split_ids in id_splits.items():
        print(f"\nBuilding {split_name} split ...")
        dataset_dict[split_name] = Dataset.from_generator(
            make_generator(split_ids),
            features=features,
        )

    ds = DatasetDict(dataset_dict)
    print(f"\nSaving to {args.output_dir} ...")
    ds.save_to_disk(args.output_dir)
    print("Done.")
    print(ds)


if __name__ == "__main__":
    main()
