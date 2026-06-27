# LEGATO Reproduction — Session Handoff

Context for a fresh (SSH'd) Claude Code session picking up this work. Read this
first, then verify the live facts (model file sizes, GPU) before acting.

## Goal

**Reproduce the original LEGATO from-scratch training** (paper arXiv:2506.19065)
using `scripts/train.py` + `configs/legato.json` on the PDMX-Synth dataset.

**SCOPE — IMPORTANT:** Do the **original-repo training only**. Do **NOT** run or
pursue the LoRA / vision-LoRA path (`scripts/train_vision_lora.py`). The LoRA work
is mentioned below for context only — it is explicitly out of scope for this task.

This repo (`Yuanryan/Legato`) is a fork of the official `guang-yng/legato` that
also added a vision-LoRA path; ignore that path entirely for this run.

## THE KEY FINDING (this overturns earlier assumptions)

LEGATO is **NOT** a fine-tuned 11B model. The HF page (`guangyangmusic/legato`)
shows **0.1B params** and the config confirms why:

- **Vision encoder**: full Mllama vision tower (32 layers, hidden 1280), loaded at
  runtime from `meta-llama/Llama-3.2-11B-Vision`, **FROZEN** (no grads/optimizer).
- **Text decoder**: a **tiny ~0.1B custom transformer trained FROM SCRATCH** —
  from `config.json` text_config: `hidden_size=768`, `num_hidden_layers=18`,
  `intermediate_size=1526`, `vocab_size=4097` (tiny ABC-music vocab),
  `num_attention_heads=12`, `num_key_value_heads=6`, cross_attention_layers
  `[3,7,11,15]`.

So "11B" only refers to the reused/frozen vision encoder + cross-attn plumbing.
The trainable part is ~0.1B. **Optimizer states are ~1GB, not ~88GB.**

### Consequence for compute

- **No memory wall.** Full training fits on a SINGLE A100-80GB with huge headroom
  (would fit on an L40S-48GB, likely a 24GB card too).
- The original team's multi-GPU node was for **throughput** (data parallelism),
  NOT memory. One GPU reproduces the real model; just slower wall-clock.
- Real remaining costs: ~19GB dataset download + training time (238k imgs × 10
  epochs = many cheap steps). Can scale down via subset / fewer epochs if needed.

## Hardware available to the user (credits-based cloud provider, region TW-04)

- 1× **A100-SXM4-80GB**, 16 vCPU, 80GB RAM, ~1.2 cr/hr  ← BEST single card, use this
- 1× L40S-48GB, up to 256GB RAM, ~0.83–0.95 cr/hr
- 8× L40S-48GB, 1792GB RAM, ~6.89 cr/hr (currently Avail Unit = 0)
- Image: Ubuntu 24.04 Server
**Recommendation: single A100-80GB is more than enough for the full model.**

## Dataset facts

- Original training set: **`guangyangmusic/PDMX-Synth`** on HF — **gated**
  (must accept CC-BY-4.0 + log in), ~19.1GB parquet, ~238,386 image–ABC pairs
  (from PDMX's 250K MusicXML, ~5% filtered for aspect ratio >10:1).
- Generation (if ever regenerating): MuseScore 3.6.2 (XML→PNG) + abcm2ps 8.14.15 +    CairoSVG (ABC→SVG→PNG); ABC canonicalized (line break every 5 bars, L:1/8,
  text→placeholder tokens).
- **Gated access the user must accept (cannot be clicked by Claude):**
  `meta-llama/Llama-3.2-11B-Vision` AND `guangyangmusic/PDMX-Synth`.

## Local datasets present (the FORK's own data, NOT PDMX-Synth)

- `datasets/music_10kv1`, `datasets/music_10kv2` — 8k-train custom OMR sets
  (image, transcription, filename, musicxml), used for LoRA fine-tuning.
- `datasets/openscore_string_quartets` — eval set.

## LoRA work — OUT OF SCOPE (context only, do not act on)

A prior LoRA fine-tune experiment exists (`results/music_10k_v2/eval_outputs/`)
but is **not part of this task**. Do not run, extend, or evaluate the LoRA path.
The local `datasets/music_10kv*` sets belong to that LoRA work and are NOT the
training data for this reproduction — use PDMX-Synth.

## Repo layout that matters

- Model: `legato/models/modeling_legato.py` (LegatoModel ⊂ MllamaForConditionalGeneration;
  vision frozen at L26; save_pretrained strips vision weights at L82 — why the
  checkpoint is tiny). Config: `legato/models/configuration_legato.py`.
- Train from scratch / eval: `scripts/train.py` + `configs/legato.json`
  (full recipe: 10 epochs, lr 3e-4, bs2×grad_accum4, ZeRO-2, predict_with_generate,
  beams 3, gen max 2048). NOTE: `train.py` uses `load_from_disk` ONLY — PDMX-Synth
  is parquet, so either pre-`save_to_disk` a subset or reuse the fork's loader.
- LoRA further-training: `scripts/train_vision_lora.py` (handles Hub/parquet/disk;
  per-component LRs; supports decoder LoRA via flag).
- DeepSpeed: `configs/zero2.yaml` (already num_processes:1), `configs/zero3.yaml`.
- Eval/metrics: `scripts/compute_ER.py`, `compute_TEDn.py`, `compute_OMR-NED.py`;
  needs MuseScore at `software/mscore` + GUI (DISPLAY) for ABC→MusicXML, plus
  `musicdiff` (from `git+ssh://...guang-yng/efficient-musicdiff`) and `zss`.
- `requirements.txt`: transformers==4.54.0, torch==2.6.0, peft, deepspeed, etc.
  (bitsandbytes NOT included — add if using 8-bit Adam, though not needed given
  the model is tiny).

## Recommended plan (staged, fail-cheap-first) on the single A100-80GB

0. Smoke test: `scripts/train.py` with `--dummy_data` (32 items), 1 epoch,
   torch_compile off — confirm load/train/save/eval work.
1. Eval reproduction FIRST: download `guangyangmusic/legato`, run inference + ER on
   the ~800-sample PDMX-Synth test split; confirm ~23% CER / 25.8% SER (ABC).
   Validates the whole pipeline cheaply. (BASE already reproduced paper OMR-NED to
   within ~1.3pp per ANALYSIS.md.)
2. Reproduction training: with the corrected understanding, the FULL recipe fits
   one A100. Either run full (slower) or a subset/fewer-epochs scaled run to show
   the loss/SER trend matches. No 8-bit Adam / offload tricks needed — model is 0.1B.

## Open verifications for the next session (do before training)

- Confirm `guangyangmusic/legato` safetensors file size (should be small, ~0.1B)
  to lock the exact trainable param count.
- Confirm the user accepted both HF gated licenses.
- Decide: full PDMX-Synth run vs scaled subset; full model is feasible either way.

## Access / how Claude operates on the box

HF token is cached locally at `~/.cache/huggingface/token`. Claude has no
independent SSH; user must either (A) hold an SSH session and pipe commands, or
(B) run Claude Code natively ON the A100 box (recommended for multi-day runs).
Gated-license click-through and instance up/down stay with the user.
