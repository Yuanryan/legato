#!/usr/bin/env python3
"""
Short run to confirm vision LoRA (and optionally projector) weights actually update.

Expects the same CLI as ``train_vision_lora.py``, but defaults to a tiny training run
with no eval. ``configure_trainable_parameters`` already logs::

    Trainable parameters: X / Y (Z%)

After training, prints mean absolute weight change per parameter containing ``lora`` in
its name (case-insensitive).

Example (from repo root)::

    DS_IGNORE_CUDA_DETECTION=1 .venv/bin/python scripts/verify_vision_lora_learning.py \\
        --pretrained_model guangyangmusic/legato \\
        --dataset_path guangyangmusic/PDMX-Synth \\
        --model_config guangyangmusic/legato \\
        --output_dir outputs/verify-lora \\
        --remove_unused_columns False \\
        --do_train --do_eval False \\
        --max_steps 200 --logging_steps 10 \\
        --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \\
        --learning_rate 1e-4 --bf16 True \\
        --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \\
        --lora_target_modules q_proj,k_proj,v_proj,o_proj \\
        --projector_name_substrings multi_modal_projector,vision_projection \\
        --dummy_data False --report_to none

Optional: append ``--report_projector`` to also print mean |Δ| for projector tensors.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.distributed as dist
from accelerate.logging import get_logger
from transformers import HfArgumentParser, Seq2SeqTrainingArguments, set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load sibling script as a module (not installed as a package).
_tv_spec = importlib.util.spec_from_file_location(
    "train_vision_lora",
    REPO_ROOT / "scripts" / "train_vision_lora.py",
)
_train_vision_lora = importlib.util.module_from_spec(_tv_spec)
_tv_spec.loader.exec_module(_train_vision_lora)

create_vision_lora_trainer = _train_vision_lora.create_vision_lora_trainer
VisionLoraArguments = _train_vision_lora.VisionLoraArguments

from legato.config import DataArguments, ModelArguments  # noqa: E402


def _strip_verify_only_flags(argv: List[str]) -> tuple[list[str], bool]:
    report_projector = False
    out: List[str] = []
    for a in argv:
        if a == "--report_projector":
            report_projector = True
        else:
            out.append(a)
    return out, report_projector


def _snapshot_trainable_matching(
    module: torch.nn.Module,
    name_filter,
) -> Dict[str, torch.Tensor]:
    snap: Dict[str, torch.Tensor] = {}
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if name_filter(name):
            snap[name] = param.detach().float().cpu().clone()
    return snap


def _print_weight_deltas(
    module: torch.nn.Module,
    before: Dict[str, torch.Tensor],
    title: str,
) -> None:
    print(f"\n=== {title} (mean |Δ|) ===")
    if not before:
        print("(no tensors matched — check adapter / trainable setup)")
        return
    any_movement = False
    for name, param in module.named_parameters():
        if name not in before:
            continue
        delta = (param.detach().float().cpu() - before[name]).abs().mean().item()
        print(f"  {name}: {delta:.8f}")
        if delta > 1e-10:
            any_movement = True
    if not any_movement:
        print(
            "\n  Warning: all reported deltas are ~0 — LoRA may be frozen, LR too small, "
            "or steps too few."
        )


def main() -> None:
    sys.argv, report_projector = _strip_verify_only_flags(sys.argv)

    parser = HfArgumentParser((Seq2SeqTrainingArguments, DataArguments, ModelArguments, VisionLoraArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        training_args, data_args, model_args, lora_args = parser.parse_json_file(
            json_file=str(Path(sys.argv[1]).resolve())
        )
    else:
        training_args, data_args, model_args, lora_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)
    logging.basicConfig(
        level=training_args.get_process_log_level(),
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    )
    logger = get_logger(__name__)

    torch.set_float32_matmul_precision("high")
    if training_args.torch_compile:
        torch._dynamo.config.cache_size_limit = 256

    training_args.do_train = True
    training_args.do_eval = False
    training_args.do_predict = False

    trainer, _processor, _trainable_names = create_vision_lora_trainer(
        training_args, data_args, model_args, lora_args, logger
    )

    model = trainer.model
    lora_before = _snapshot_trainable_matching(model, lambda n: "lora" in n.lower())
    projector_subs = [s.strip() for s in lora_args.projector_name_substrings.split(",") if s.strip()]
    projector_before: Dict[str, torch.Tensor] = {}
    if report_projector:
        projector_before = _snapshot_trainable_matching(
            model, lambda n, subs=projector_subs: any(s in n for s in subs)
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if trainer.is_world_process_zero():
        print(
            f"\nTrainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)\n",
            flush=True,
        )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    if trainer.is_world_process_zero():
        _print_weight_deltas(unwrapped, lora_before, "LoRA weights")
        if report_projector:
            _print_weight_deltas(unwrapped, projector_before, "Projector weights")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
