import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from accelerate.logging import get_logger
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoConfig,
    AutoModel,
    AutoProcessor,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrainerState,
    set_seed,
)
from transformers.trainer import TRAINER_STATE_NAME

from legato.config import DataArguments, ModelArguments
from legato.metrics import compute_error_rates
from legato.models import LegatoModel
from legato.trainer import LegatoTrainer


@dataclass
class VisionLoraArguments:
    lora_r: int = field(default=16, metadata={"help": "LoRA rank for the vision encoder."})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha for the vision encoder."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout for the vision encoder."})
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj",
        metadata={"help": "Comma-separated module suffixes in the vision encoder to receive LoRA adapters."},
    )
    projector_name_substrings: str = field(
        default="multi_modal_projector,vision_projection",
        metadata={"help": "Comma-separated parameter name substrings to keep trainable as the vision projector."},
    )
    adapter_output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Optional output directory for the LoRA adapter. Defaults to output_dir/adapter."},
    )
    projector_output_name: str = field(
        default="projector.pt",
        metadata={"help": "Filename used to save trainable projector weights under output_dir."},
    )


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def freeze_all_parameters(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def enable_projector_training(model: torch.nn.Module, name_substrings: List[str]) -> List[str]:
    trainable_names = []
    for name, param in model.named_parameters():
        if any(substring in name for substring in name_substrings):
            param.requires_grad = True
            trainable_names.append(name)

    if not trainable_names:
        available = sorted({name.rsplit(".", 1)[0] for name, _ in model.named_parameters()})
        sample = "\n".join(available[:40])
        raise ValueError(
            "No projector parameters matched "
            f"{name_substrings}. Check --projector_name_substrings.\n"
            f"First available module names:\n{sample}"
        )

    return trainable_names


def attach_vision_lora(model: LegatoModel, lora_args: VisionLoraArguments) -> PeftModel:
    if model.vision_model is None:
        raise ValueError("The Legato vision model is not loaded; cannot attach LoRA adapters.")

    target_modules = _split_csv(lora_args.lora_target_modules)
    if not target_modules:
        raise ValueError("--lora_target_modules must contain at least one module name.")

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias="none",
    )
    model.model.vision_model = get_peft_model(model.vision_model, lora_config)
    return model.vision_model


def configure_trainable_parameters(
    model: LegatoModel,
    lora_args: VisionLoraArguments,
    logger: logging.Logger,
) -> List[str]:
    freeze_all_parameters(model)
    vision_lora = attach_vision_lora(model, lora_args)
    projector_names = enable_projector_training(model, _split_csv(lora_args.projector_name_substrings))

    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total_params = sum(param.numel() for param in model.parameters())
    logger.info(
        "Trainable parameters: %s / %s (%.4f%%)",
        f"{trainable_params:,}",
        f"{total_params:,}",
        100 * trainable_params / total_params,
    )
    logger.info("Vision LoRA adapter attached to: %s", vision_lora.__class__.__name__)
    logger.info("Projector trainable parameters: %s", projector_names)
    logger.info("All trainable parameter names: %s", trainable_names)
    return trainable_names


def save_trainable_artifacts(
    trainer: LegatoTrainer,
    output_dir: str,
    lora_args: VisionLoraArguments,
    trainable_names: List[str],
    logger: logging.Logger,
    processor=None,
) -> None:
    if not trainer.is_world_process_zero():
        return

    os.makedirs(output_dir, exist_ok=True)
    unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
    vision_model = unwrapped_model.vision_model
    if not isinstance(vision_model, PeftModel):
        raise TypeError("Expected vision_model to be a PEFT model before saving LoRA artifacts.")

    adapter_dir = lora_args.adapter_output_dir or os.path.join(output_dir, "adapter")
    vision_model.save_pretrained(adapter_dir)

    projector_substrings = _split_csv(lora_args.projector_name_substrings)
    projector_state = {
        name: tensor.detach().cpu()
        for name, tensor in unwrapped_model.state_dict().items()
        if any(substring in name for substring in projector_substrings)
    }
    if not projector_state:
        raise ValueError("No projector weights were found while saving trainable artifacts.")
    torch.save(projector_state, os.path.join(output_dir, lora_args.projector_output_name))

    if processor is not None:
        processor.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "trainable_parameters.json"), "w") as f:
        json.dump(
            {
                "lora": asdict(lora_args),
                "adapter_dir": adapter_dir,
                "projector_output_name": lora_args.projector_output_name,
                "projector_keys": sorted(projector_state.keys()),
                "trainable_parameter_names": trainable_names,
            },
            f,
            indent=2,
        )

    logger.info("Saved vision LoRA adapter to %s", adapter_dir)
    logger.info("Saved projector weights to %s", os.path.join(output_dir, lora_args.projector_output_name))


class SaveLoraCallback(TrainerCallback):
    """Saves LoRA adapter and projector weights alongside every regular trainer checkpoint.

    Without this, LegatoTrainer._save strips all vision_model.* params, so intermediate
    checkpoints would be missing the LoRA weights entirely.
    """

    def __init__(
        self,
        trainer: LegatoTrainer,
        lora_args: VisionLoraArguments,
        trainable_names: List[str],
        logger: logging.Logger,
    ):
        self._trainer = trainer
        self.lora_args = lora_args
        self.trainable_names = trainable_names
        self.logger = logger

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        save_trainable_artifacts(
            self._trainer,
            ckpt_dir,
            self.lora_args,
            self.trainable_names,
            self.logger,
        )


def create_vision_lora_trainer(
    training_args: Seq2SeqTrainingArguments,
    data_args: DataArguments,
    model_args: ModelArguments,
    lora_args: VisionLoraArguments,
    logger: logging.Logger,
) -> Tuple[LegatoTrainer, object, List[str]]:
    """Load data, attach LoRA + projector, and build ``LegatoTrainer`` (same as ``train_vision_lora`` main)."""

    logger.info("Loading dataset from Hugging Face: %s", data_args.dataset_path)
    dataset = load_dataset(data_args.dataset_path)
    if "val" not in dataset and "validation" in dataset:
        dataset["val"] = dataset["validation"]
    for split, mini_file in [("val", data_args.mini_val_file), ("test", data_args.mini_test_file)]:
        if mini_file:
            logger.info("Using mini %s set: %s", split, mini_file)
            with open(mini_file, "r") as f:
                filenames = json.load(f)
            dataset[split] = dataset[split].select(
                [dataset[split]["filename"].index(filename) for filename in filenames]
            )

    if data_args.dummy_data:
        logger.info("Using dummy data (32 items) for debugging only...")
        dataset["train"] = dataset["train"].select(range(32))
        dataset["val"] = dataset["val"].select(range(32))
        dataset["test"] = dataset["test"].select(range(32))

    set_seed(training_args.seed)

    if model_args.pretrained_model:
        model = AutoModel.from_pretrained(model_args.pretrained_model)
    else:
        config = AutoConfig.from_pretrained(model_args.model_config)
        model = LegatoModel(config)

    processor = AutoProcessor.from_pretrained(model_args.model_config)
    tokenizer = processor.tokenizer
    trainable_names = configure_trainable_parameters(model, lora_args, logger)

    def get_metric_target(examples):
        return {
            "label_ids": processor(
                text=examples["transcription"],
                add_special_tokens=False,
                verbose=False,
                truncation=False,
            )["input_ids"],
        }

    map_num_proc = training_args.dataloader_num_workers or None
    if not training_args.do_predict:
        metric_targets = dataset["val"].map(
            get_metric_target,
            remove_columns=dataset["val"].column_names,
            num_proc=map_num_proc,
            batched=True,
        ).to_dict()
    else:
        metric_targets = (
            dataset["test"]
            .map(
                get_metric_target,
                remove_columns=dataset["test"].column_names,
                num_proc=map_num_proc,
                batched=True,
            )
            .to_dict()
            if "transcription" in dataset["test"].column_names
            else None
        )

    tokens_to_mask = torch.tensor([*tokenizer.additional_special_tokens_ids, tokenizer.pad_token_id])

    def collate_fn(examples):
        outputs = processor(
            images=[example["image"] for example in examples],
            text=[example["transcription"] for example in examples],
            return_num_tiles=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        gen_outputs = processor(
            num_tiles=outputs.pop("num_tiles"),
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        outputs.update({f"gen_{k}": outputs[k] if k not in gen_outputs else gen_outputs[k] for k in outputs})
        outputs["labels"] = outputs["input_ids"].clone().masked_fill(
            torch.isin(outputs["input_ids"], tokens_to_mask), -100
        )
        return outputs

    special_tokens = [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id, -100]

    def remove_special_tokens(array):
        masks = np.isin(array, special_tokens, invert=True)
        return [a[mask] for a, mask in zip(array, masks)]

    def predictions_to_token_ids(predictions):
        """Teacher-forcing eval returns float logits [batch, seq, vocab]; metrics need int token ids."""
        arr = np.asarray(predictions)
        if arr.ndim == 3:
            return arr.argmax(axis=-1)
        if arr.dtype.kind in "fc":
            return np.rint(arr).astype(np.int64)
        return arr

    def metric_fn(p):
        preds = remove_special_tokens(predictions_to_token_ids(p.predictions))
        results = [
            compute_error_rates(tokenizer, training_args.dataloader_num_workers, *metric_targets.values(), preds)
        ] if training_args.process_index == 0 else [None]
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(results, src=0)
        return results[0]

    trainer = LegatoTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        compute_metrics=metric_fn,
    )
    trainer.add_callback(SaveLoraCallback(trainer, lora_args, trainable_names, logger))

    return trainer, processor, trainable_names


def main():
    parser = HfArgumentParser((Seq2SeqTrainingArguments, DataArguments, ModelArguments, VisionLoraArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        training_args, data_args, model_args, lora_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
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

    trainer, processor, trainable_names = create_vision_lora_trainer(
        training_args, data_args, model_args, lora_args, logger
    )

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        save_trainable_artifacts(trainer, training_args.output_dir, lora_args, trainable_names, logger, processor=processor)

    if training_args.do_eval:
        if not training_args.do_train and not model_args.pretrained_model:
            ckpts = [ckpt for ckpt in os.listdir(training_args.output_dir) if ckpt.startswith("checkpoint")]
            assert len(ckpts) > 0, f"No checkpoints found in {training_args.output_dir}"
            best_ckpt, best_result = None, None
            for ckpt in sorted(ckpts):
                logger.info("Evaluating checkpoint %s...", ckpt)
                trainer._load_from_checkpoint(os.path.join(training_args.output_dir, ckpt))
                trainer.state = TrainerState.load_from_json(
                    os.path.join(training_args.output_dir, ckpt, TRAINER_STATE_NAME)
                )
                trainer.state.init_training_references(trainer, trainer.state.max_steps, trainer.state.num_train_epochs, None)
                trainer._load_callback_state()
                result = trainer.evaluate()
                if best_result is None or result["eval_SER"] < best_result["eval_SER"]:
                    best_result, best_ckpt = result, ckpt
                trainer.log_metrics("eval", result)

            logger.info("Best checkpoint: %s", best_ckpt)
            trainer._load_from_checkpoint(os.path.join(training_args.output_dir, best_ckpt))
        else:
            best_result = trainer.evaluate()

        final_val_results = {k.replace("eval_", "eval_best_"): v for k, v in best_result.items() if k.startswith("eval_")}
        trainer.log_metrics("best eval", final_val_results)
        trainer.log(final_val_results)

    if training_args.do_eval and not training_args.do_train:
        save_trainable_artifacts(trainer, training_args.output_dir, lora_args, trainable_names, logger, processor=processor)

    if training_args.do_predict:
        outputs = trainer.predict(dataset["test"])

        if trainer.is_world_process_zero():
            os.makedirs(training_args.output_dir, exist_ok=True)
            pred_ids = predictions_to_token_ids(outputs.predictions)
            abc_outputs = processor.batch_decode(pred_ids, skip_special_tokens=True)
            preds = remove_special_tokens(pred_ids)
            with open(os.path.join(training_args.output_dir, "test_predictions.json"), "w") as f:
                json.dump({"abc_transcription": abc_outputs, "tokens": [p.tolist() for p in preds]}, f)

            if metric_targets:
                results = compute_error_rates(tokenizer, training_args.dataloader_num_workers, *metric_targets.values(), preds)
                trainer.log_metrics("test", results)


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
