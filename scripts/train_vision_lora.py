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
from datasets import load_dataset, load_from_disk
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
    # Decoder LoRA (optional — empty string disables it)
    decoder_lora_target_modules: str = field(
        default="",
        metadata={"help": "Comma-separated decoder modules to apply LoRA to (e.g. 'q_proj,v_proj'). "
                          "Empty string disables decoder LoRA."},
    )
    decoder_adapter_output_name: str = field(
        default="decoder_adapter",
        metadata={"help": "Subfolder name for the decoder LoRA adapter under output_dir."},
    )
    # Per-component learning rates (all default to training_args.learning_rate when unset)
    vision_lora_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Learning rate for vision LoRA adapters. Defaults to training_args.learning_rate."},
    )
    projector_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Learning rate for the multimodal projector. Defaults to training_args.learning_rate."},
    )
    decoder_lora_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Learning rate for decoder LoRA adapters. Defaults to training_args.learning_rate."},
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


def attach_decoder_lora(model: LegatoModel, lora_args: VisionLoraArguments) -> Optional[PeftModel]:
    target_modules = _split_csv(lora_args.decoder_lora_target_modules)
    if not target_modules:
        return None

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias="none",
    )
    model.model.language_model = get_peft_model(model.model.language_model, lora_config)
    return model.model.language_model


def configure_trainable_parameters(
    model: LegatoModel,
    lora_args: VisionLoraArguments,
    logger: logging.Logger,
) -> List[str]:
    freeze_all_parameters(model)
    vision_lora = attach_vision_lora(model, lora_args)
    decoder_lora = attach_decoder_lora(model, lora_args)
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
    if decoder_lora is not None:
        logger.info("Decoder LoRA adapter attached to: %s", decoder_lora.__class__.__name__)
    else:
        logger.info("Decoder LoRA: disabled")
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

    decoder_lm = unwrapped_model.model.language_model
    if isinstance(decoder_lm, PeftModel):
        decoder_adapter_dir = os.path.join(output_dir, lora_args.decoder_adapter_output_name)
        decoder_lm.save_pretrained(decoder_adapter_dir)
        logger.info("Saved decoder LoRA adapter to %s", decoder_adapter_dir)

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


class VisionLoraTrainer(LegatoTrainer):
    """LegatoTrainer extended with per-component learning rates for LoRA training."""

    def __init__(self, *args, lora_args: VisionLoraArguments, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_args = lora_args

    def create_optimizer(self):
        lora_args = self.lora_args
        base_lr = self.args.learning_rate

        # If no custom LRs are set, or DeepSpeed is managing the optimizer, use standard behaviour.
        if self.is_deepspeed_enabled or not any(
            [lora_args.vision_lora_lr, lora_args.projector_lr, lora_args.decoder_lora_lr]
        ):
            return super().create_optimizer()

        projector_substrings = _split_csv(lora_args.projector_name_substrings)
        vision_lora_params, projector_params, decoder_lora_params = [], [], []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(s in name for s in projector_substrings):
                projector_params.append(param)
            elif "language_model" in name:
                decoder_lora_params.append(param)
            else:
                vision_lora_params.append(param)

        param_groups = []
        if vision_lora_params:
            param_groups.append({"params": vision_lora_params, "lr": lora_args.vision_lora_lr or base_lr})
        if projector_params:
            param_groups.append({"params": projector_params, "lr": lora_args.projector_lr or base_lr})
        if decoder_lora_params:
            param_groups.append({"params": decoder_lora_params, "lr": lora_args.decoder_lora_lr or base_lr})

        self.optimizer = torch.optim.AdamW(
            param_groups,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
            weight_decay=self.args.weight_decay,
        )
        return self.optimizer


def predictions_to_token_ids(predictions):
    arr = np.asarray(predictions)
    if arr.ndim == 3:
        return arr.argmax(axis=-1)
    if arr.dtype.kind in "fc":
        return np.rint(arr).astype(np.int64)
    return arr


def create_vision_lora_trainer(
    training_args: Seq2SeqTrainingArguments,
    data_args: DataArguments,
    model_args: ModelArguments,
    lora_args: VisionLoraArguments,
    logger: logging.Logger,
) -> Tuple[LegatoTrainer, object, List[str], object, str]:
    """Load data, attach LoRA + projector, and build ``LegatoTrainer`` (same as ``train_vision_lora`` main)."""

    logger.info("Loading dataset from: %s", data_args.dataset_path)
    _path = data_args.dataset_path
    if not os.path.exists(_path):
        # Treat as a HuggingFace Hub dataset ID (e.g. "guangyangmusic/OpenScore-StringQuartets")
        dataset = load_dataset(_path)
    elif _path.endswith(".parquet"):
        dataset = load_dataset("parquet", data_files=_path)
    elif os.path.isfile(os.path.join(_path, "train-00000-of-00001.parquet")):
        dataset = load_dataset("parquet", data_files={"train": os.path.join(_path, "*.parquet")})
    else:
        dataset = load_from_disk(_path)
    if "val" not in dataset and "validation" in dataset:
        dataset["val"] = dataset["validation"]
    for split, mini_file in [("val", data_args.mini_val_file), ("test", data_args.mini_test_file)]:
        if mini_file:
            logger.info("Using mini %s set: %s", split, mini_file)
            with open(mini_file, "r") as f:
                filenames = json.load(f)
            filename_to_idx = {name: i for i, name in enumerate(dataset[split]["filename"])}
            dataset[split] = dataset[split].select(
                [filename_to_idx[filename] for filename in filenames]
            )

    if data_args.dummy_data:
        logger.info("Using dummy data (32 items) for debugging only...")
        dataset["train"] = dataset["train"].select(range(32))
        dataset["val"] = dataset["val"].select(range(32))
        dataset["test"] = dataset["test"].select(range(32))

    predict_split = data_args.predict_split
    if data_args.max_predict_samples is not None and predict_split in dataset:
        n = min(data_args.max_predict_samples, len(dataset[predict_split]))
        logger.info("Truncating predict split '%s' to %d samples.", predict_split, n)
        dataset[predict_split] = dataset[predict_split].select(range(n))

    set_seed(training_args.seed)

    # Load the base model first to guarantee full vision_model weights are present.
    # The LoRA fine-tuned checkpoint stores vision_model.* under the PEFT-wrapped
    # naming convention (model.vision_model.base_model.model.*), which AutoModel
    # silently fails to load into a fresh LegatoModel — leaving the vision encoder
    # randomly initialized. We work around this by always loading the base model
    # (which has full vision weights) and then overlaying the non-vision weights
    # from the checkpoint (LM is frozen during LoRA training; projector is trained
    # and stored under model.multi_modal_projector.*).
    base_path = model_args.model_config or model_args.pretrained_model
    if base_path:
        model = AutoModel.from_pretrained(base_path)
    else:
        config = AutoConfig.from_pretrained(model_args.model_config)
        model = LegatoModel(config)

    if model_args.pretrained_model and model_args.pretrained_model != base_path:
        import os as _os
        ckpt_safetensors = _os.path.join(model_args.pretrained_model, "model.safetensors")
        if _os.path.isfile(ckpt_safetensors):
            from safetensors.torch import load_file as _load_safetensors
            ckpt_state = _load_safetensors(ckpt_safetensors)
            # Skip any vision_model.* keys (whether plain or PEFT-wrapped) so we keep
            # the base model's correctly-loaded vision weights.
            ckpt_state = {
                k: v for k, v in ckpt_state.items()
                if "vision_model" not in k
            }
            missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
            logger.info(
                "Loaded %d non-vision tensors from checkpoint %s "
                "(missing=%d, unexpected=%d)",
                len(ckpt_state), ckpt_safetensors, len(missing), len(unexpected),
            )

    # Apply paper-style generation defaults (beam=10, max_length=2048, repetition_penalty=1.1).
    # Critical: repetition_penalty stops the model from collapsing into degenerate token loops
    # like ' !tenuto!e!tenuto!e | !tenuto!e!tenuto!e | ...' that fill the entire context.
    # Without this, LoRA-fine-tuned checkpoints in particular tend to loop after a few measures.
    if model.generation_config is not None:
        model.generation_config.repetition_penalty = 1.1

    processor_source = model_args.model_config or model_args.pretrained_model
    processor = AutoProcessor.from_pretrained(processor_source)
    tokenizer = processor.tokenizer
    trainable_names = configure_trainable_parameters(model, lora_args, logger)

    # When loading from a checkpoint, restore trained LoRA weights instead of keeping random ones
    if model_args.pretrained_model:
        from peft import load_peft_weights, set_peft_model_state_dict
        adapter_dir = os.path.join(model_args.pretrained_model, "adapter")
        if os.path.isdir(adapter_dir):
            logger.info("Loading trained vision LoRA weights from %s", adapter_dir)
            set_peft_model_state_dict(model.vision_model, load_peft_weights(adapter_dir))
        decoder_adapter_dir = os.path.join(model_args.pretrained_model, lora_args.decoder_adapter_output_name)
        if os.path.isdir(decoder_adapter_dir) and isinstance(model.model.language_model, PeftModel):
            logger.info("Loading trained decoder LoRA weights from %s", decoder_adapter_dir)
            set_peft_model_state_dict(model.model.language_model, load_peft_weights(decoder_adapter_dir))

    image_col = data_args.image_column
    transcription_col = data_args.transcription_column

    def get_metric_target(examples):
        return {
            "label_ids": processor(
                text=examples[transcription_col],
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
            dataset[predict_split]
            .map(
                get_metric_target,
                remove_columns=dataset[predict_split].column_names,
                num_proc=map_num_proc,
                batched=True,
            )
            .to_dict()
            if transcription_col in dataset[predict_split].column_names
            else None
        )

    tokens_to_mask = torch.tensor([*tokenizer.additional_special_tokens_ids, tokenizer.pad_token_id])

    def collate_fn(examples):
        outputs = processor(
            images=[example[image_col] for example in examples],
            text=[example.get(transcription_col, "") for example in examples],
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

    def metric_fn(p):
        if metric_targets is None:
            return {}
        preds = remove_special_tokens(predictions_to_token_ids(p.predictions))
        results = [
            compute_error_rates(tokenizer, training_args.dataloader_num_workers, *metric_targets.values(), preds)
        ] if training_args.process_index == 0 else [None]
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(results, src=0)
        return results[0]

    training_args.remove_unused_columns = False

    trainer = VisionLoraTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset.get("train"),
        eval_dataset=dataset.get("val"),
        compute_metrics=metric_fn,
        lora_args=lora_args,
    )
    trainer.add_callback(SaveLoraCallback(trainer, lora_args, trainable_names, logger))

    return trainer, processor, trainable_names, dataset, predict_split, remove_special_tokens, metric_targets, tokenizer


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

    trainer, processor, trainable_names, dataset, predict_split, remove_special_tokens, metric_targets, tokenizer = create_vision_lora_trainer(
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
        outputs = trainer.predict(dataset[predict_split])

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
