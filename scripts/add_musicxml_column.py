"""
Add a musicxml column to an existing dataset saved with save_to_disk().

Usage:
    python scripts/add_musicxml_column.py --dataset_path datasets/music_10kv2
    python scripts/add_musicxml_column.py --dataset_path datasets/music_10kv2 --num_proc 8
"""

import argparse
import subprocess
import sys
from pathlib import Path

from datasets import load_from_disk

ABC2XML = [sys.executable, str(Path(__file__).parent.parent / "utils" / "abc2xml.py"), "-"]


def add_musicxml(example):
    result = subprocess.run(
        ABC2XML,
        input=example["transcription"].encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {"musicxml": result.stdout.decode("utf-8")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True, help="Path to the dataset saved with save_to_disk()")
    p.add_argument("--num_proc", type=int, default=4, help="Number of parallel workers")
    args = p.parse_args()

    print(f"Loading dataset from {args.dataset_path} ...")
    ds = load_from_disk(args.dataset_path)

    for split in ds:
        if "musicxml" in ds[split].column_names:
            print(f"  {split}: musicxml column already exists — skipping.")
        else:
            print(f"  {split}: converting {len(ds[split])} examples ...")
            ds[split] = ds[split].map(add_musicxml, num_proc=args.num_proc, desc=f"{split}")

    tmp_path = args.dataset_path.rstrip("/\\") + "_tmp"
    print(f"Saving to {tmp_path} ...")
    ds.save_to_disk(tmp_path)

    import shutil
    print(f"Replacing {args.dataset_path} ...")
    shutil.rmtree(args.dataset_path)
    shutil.move(tmp_path, args.dataset_path)
    print("Done.")


if __name__ == "__main__":
    main()
