"""
Package a trained vision-LoRA checkpoint for inference or sharing.

After training, each checkpoint already contains:
  <checkpoint_dir>/
      model.safetensors    decoder + projector weights
      adapter/             vision LoRA adapter (PEFT)
      projector.pt         projector weights (reference copy)
      trainable_parameters.json

This script verifies the checkpoint is complete, attaches the processor,
and optionally pushes everything to HuggingFace Hub.

Load the result with:
    from legato.models import LegatoModel
    from peft import PeftModel
    from transformers import AutoProcessor

    model = LegatoModel.from_pretrained("<checkpoint_dir>")
    model.model.vision_model = PeftModel.from_pretrained(
        model.vision_model, "<checkpoint_dir>/adapter"
    )
    processor = AutoProcessor.from_pretrained("<checkpoint_dir>")

Usage:
    # Verify + save processor into the checkpoint dir
    python scripts/export_model.py --checkpoint_dir outputs/vision_lora/checkpoint-65000

    # Also push to HuggingFace Hub
    python scripts/export_model.py \\
        --checkpoint_dir outputs/vision_lora/checkpoint-65000 \\
        --push_to_hub    your-username/legato-finetuned
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi
from transformers import AutoProcessor


REQUIRED_FILES = ["model.safetensors", "config.json"]
REQUIRED_DIRS  = ["adapter"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", required=True,
                   help="Checkpoint directory produced by train_vision_lora.py")
    p.add_argument("--push_to_hub", default=None, metavar="REPO_ID",
                   help="HuggingFace Hub repo to push to (e.g. your-username/legato-finetuned)")
    p.add_argument("--base_model", default="guangyangmusic/legato",
                   help="Base model used for training (default: guangyangmusic/legato)")
    return p.parse_args()


def verify_checkpoint(ckpt: Path) -> None:
    missing = []
    for f in REQUIRED_FILES:
        if not (ckpt / f).exists():
            missing.append(f)
    for d in REQUIRED_DIRS:
        if not (ckpt / d).is_dir():
            missing.append(f"{d}/")
    if missing:
        raise FileNotFoundError(
            f"Checkpoint at {ckpt} is missing: {missing}\n"
            f"Make sure SaveLoraCallback ran during training."
        )
    print(f"[1/3] Checkpoint OK — all required files present.")


def ensure_processor(ckpt: Path, base_model: str) -> None:
    if (ckpt / "tokenizer_config.json").exists():
        print(f"[2/3] Processor already in checkpoint dir — skipping.")
        return

    print(f"[2/3] Saving processor from {base_model} into checkpoint dir ...")
    processor = AutoProcessor.from_pretrained(base_model)
    processor.save_pretrained(ckpt)


def print_loading_instructions(ckpt: Path, repo_id: str | None) -> None:
    source = f'"{repo_id}"' if repo_id else f'"{ckpt}"'
    adapter = f'{source} + "/adapter"' if not repo_id else f'"{repo_id}", subfolder="adapter"'
    print()
    print("=" * 60)
    print(" Load your model with:")
    print("=" * 60)
    print(f"""
from legato.models import LegatoModel
from peft import PeftModel
from transformers import AutoProcessor

model = LegatoModel.from_pretrained({source})
model.model.vision_model = PeftModel.from_pretrained(
    model.vision_model, {adapter}
)
processor = AutoProcessor.from_pretrained({source})
""")
    print("=" * 60)


def push_to_hub(ckpt: Path, repo_id: str) -> None:
    print(f"[3/3] Pushing to HuggingFace Hub: {repo_id} ...")
    api = HfApi()
    api.create_repo(repo_id, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=str(ckpt),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["*.pyc", "__pycache__"],
    )
    print(f"      Done → https://huggingface.co/{repo_id}")


def main():
    args = parse_args()
    ckpt = Path(args.checkpoint_dir)

    if not ckpt.is_dir():
        raise NotADirectoryError(f"Checkpoint directory not found: {ckpt}")

    verify_checkpoint(ckpt)
    ensure_processor(ckpt, args.base_model)

    if args.push_to_hub:
        push_to_hub(ckpt, args.push_to_hub)
    else:
        print("[3/3] Skipping Hub push (no --push_to_hub given).")

    print_loading_instructions(ckpt, args.push_to_hub)


if __name__ == "__main__":
    main()
