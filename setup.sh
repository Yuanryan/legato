#!/usr/bin/env bash
# Setup script for legato-train on a Linux training compute.
# Tested with CUDA 12.4.
#
# Usage:
#   bash setup.sh            # creates conda env "legato" and installs everything
#   bash setup.sh --no-conda # installs into the current Python environment

set -euo pipefail

USE_CONDA=true
ENV_NAME="legato"
PYTHON_VERSION="3.12"

# Detect the right pip/python commands
if command -v pip3 &>/dev/null; then
    PIP="pip3"
elif command -v pip &>/dev/null; then
    PIP="pip"
elif command -v python3 &>/dev/null; then
    PIP="python3 -m pip"
elif command -v python &>/dev/null; then
    PIP="python -m pip"
else
    echo "ERROR: No pip or python found. Install Python 3 first."
    exit 1
fi
echo "Using pip command: ${PIP}"

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
for arg in "$@"; do
    case $arg in
        --no-conda) USE_CONDA=false ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Conda environment
# ---------------------------------------------------------------------------
if $USE_CONDA && ! command -v conda &>/dev/null; then
    echo "WARNING: conda not found — continuing without conda."
    USE_CONDA=false
fi

if $USE_CONDA; then

    if conda env list | grep -q "^${ENV_NAME} "; then
        echo "Conda env '${ENV_NAME}' already exists — skipping creation."
    else
        echo "Creating conda env '${ENV_NAME}' with Python ${PYTHON_VERSION} ..."
        conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
    fi

    # Activate inside the script
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
    echo "Activated conda env: ${ENV_NAME}"
fi

# ---------------------------------------------------------------------------
# System build dependencies (needed by DeepSpeed / Triton C extensions)
# ---------------------------------------------------------------------------
if command -v apt-get &>/dev/null; then
    echo "Installing system build dependencies ..."
    sudo apt-get install -y python3.12-dev gcc g++ unzip 2>/dev/null || true

    # NOTE: Do NOT install nvidia-cuda-toolkit via apt — it ships older NVIDIA
    # libraries that conflict with whatever driver the compute node has pre-installed.
    # The CUDA toolkit (nvcc) should already be present on a properly set-up GPU node.
    # If nvcc is missing, ask the cluster admin or install from NVIDIA's official repo.
fi

# Set CUDA_HOME if not already set (needed by DeepSpeed and Triton at import time)
if [[ -z "${CUDA_HOME:-}" ]]; then
    if command -v nvcc &>/dev/null; then
        export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
        echo "Set CUDA_HOME=${CUDA_HOME}"
    elif [[ -d /usr/local/cuda ]]; then
        export CUDA_HOME=/usr/local/cuda
        echo "Set CUDA_HOME=${CUDA_HOME} (from /usr/local/cuda)"
    else
        echo "WARNING: CUDA_HOME not set and nvcc not found. DeepSpeed import may fail."
        echo "         Install the CUDA toolkit or set CUDA_HOME manually."
    fi
fi
# Add CUDA bin/lib to PATH for this session
if [[ -n "${CUDA_HOME:-}" ]]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
    # Persist across future shells
    grep -qxF "export CUDA_HOME=${CUDA_HOME}" ~/.bashrc 2>/dev/null || \
        echo "export CUDA_HOME=${CUDA_HOME}" >> ~/.bashrc
    grep -qF 'CUDA_HOME/bin' ~/.bashrc 2>/dev/null || {
        echo 'export PATH="${CUDA_HOME}/bin:${PATH}"' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"' >> ~/.bashrc
    }
fi

# ---------------------------------------------------------------------------
# PyTorch (CUDA 12.4)
# Install before everything else so other packages link against the right torch.
# ---------------------------------------------------------------------------
echo ""
echo "Installing PyTorch 2.6.0 + CUDA 12.4 ..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# ---------------------------------------------------------------------------
# Core requirements (everything except torch, deepspeed, musicdiff)
# ---------------------------------------------------------------------------
echo ""
echo "Installing core requirements ..."
pip install \
    accelerate==1.8.0 \
    datasets==3.2.0 \
    transformers==4.54.0 \
    peft \
    pillow==11.1.0 \
    numpy==1.26.4 \
    fire==0.7.0 \
    Levenshtein \
    tqdm \
    wandb \
    pyparsing \
    zss \
    gdown

# ---------------------------------------------------------------------------
# DeepSpeed (Linux only — requires a C++ compiler and CUDA toolkit headers)
# ---------------------------------------------------------------------------
echo ""
echo "Installing DeepSpeed ..."

if ! command -v nvcc &>/dev/null; then
    echo "WARNING: nvcc not found on PATH. DeepSpeed ops will be JIT-compiled at"
    echo "         first use, which is slower. To pre-compile, install the full"
    echo "         CUDA toolkit and re-run this script."
    pip install deepspeed
else
    CUDA_VERSION=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+")
    # Derive CUDA_HOME from nvcc location if not already set
    if [[ -z "${CUDA_HOME:-}" ]]; then
        export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
        echo "Set CUDA_HOME=${CUDA_HOME}"
    fi
    echo "Detected CUDA ${CUDA_VERSION} — building DeepSpeed with pre-compiled ops ..."
    DS_BUILD_OPS=1 CUDA_HOME="${CUDA_HOME}" pip install deepspeed
fi

# ---------------------------------------------------------------------------
# Optional: musicdiff (needed only for compute_OMR-NED.py)
# ---------------------------------------------------------------------------
# echo ""
# read -r -p "Install musicdiff (needed for OMR-NED evaluation)? [y/N] " install_musicdiff
# if [[ "${install_musicdiff,,}" == "y" ]]; then
#     pip install "musicdiff @ git+ssh://git@github.com/guang-yng/efficient-musicdiff.git"
# fi

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
echo ""
echo "Verifying installation ..."
python - <<'EOF'
import torch, transformers, accelerate, peft, datasets, deepspeed
print(f"  torch        {torch.__version__}  (CUDA available: {torch.cuda.is_available()})")
print(f"  transformers {transformers.__version__}")
print(f"  accelerate   {accelerate.__version__}")
print(f"  peft         {peft.__version__}")
print(f"  datasets     {datasets.__version__}")
print(f"  deepspeed    {deepspeed.__version__}")
if torch.cuda.is_available():
    print(f"  GPU          {torch.cuda.get_device_name(0)}")
    print(f"  CUDA         {torch.version.cuda}")
EOF

echo ""
echo "Setup complete."
if $USE_CONDA; then
    echo "Activate your environment with:  conda activate ${ENV_NAME}"
fi

# ---------------------------------------------------------------------------
# Post-install checklist
# ---------------------------------------------------------------------------
echo ""
echo "========================================================"
echo " Post-install checklist"
echo "========================================================"

# 1. HuggingFace login
# Llama-3.2-11B-Vision is a gated model — you must:
#   (a) Accept the license at https://huggingface.co/meta-llama/Llama-3.2-11B-Vision
#   (b) Log in here with a token that has read access
echo ""
echo "[1/4] HuggingFace login (required to download Llama-3.2-11B-Vision) ..."
echo "      If you haven't already, accept the model license at:"
echo "      https://huggingface.co/meta-llama/Llama-3.2-11B-Vision"
echo ""
read -r -p "      Log in now? [y/N] " do_hf_login
if [[ "${do_hf_login,,}" == "y" ]]; then
    huggingface-cli login
fi

# 2. Wandb login
echo ""
echo "[2/4] Weights & Biases login (for experiment tracking) ..."
read -r -p "      Log in now? [y/N] " do_wandb_login
if [[ "${do_wandb_login,,}" == "y" ]]; then
    wandb login
fi

# 3. Dataset download
DATASET_GDRIVE_ID="13zBecrkUHHQaiGIVtQKxEAI94Rr-Qn_E"
DATASET_DEST="datasets/music_10kv2"
DATASET_ZIP="datasets/music_10kv2.zip"

echo ""
echo "[3/5] Dataset download ..."
if [[ -d "${DATASET_DEST}" ]]; then
    echo "      Dataset already exists at ${DATASET_DEST} — skipping."
else
    read -r -p "      Download dataset from Google Drive (~several GB)? [y/N] " do_download
    if [[ "${do_download,,}" == "y" ]]; then
        mkdir -p datasets
        echo "      Downloading ..."
        gdown "${DATASET_GDRIVE_ID}" -O "${DATASET_ZIP}"
        echo "      Extracting to datasets/ ..."
        unzip -q "${DATASET_ZIP}" -d datasets/
        rm "${DATASET_ZIP}"
        echo "      Dataset ready at ${DATASET_DEST}"
    else
        echo "      Skipped. Download manually later with:"
        echo "        gdown ${DATASET_GDRIVE_ID} -O ${DATASET_ZIP}"
        echo "        unzip ${DATASET_ZIP} -d datasets/"
    fi
fi

# 4. Accelerate config
echo ""
echo "[4/5] Accelerate config ..."
echo "      Pre-made configs are in configs/ (zero2.yaml, inference.yaml)."
echo "      Run the following if you want to auto-detect your GPU setup instead:"
echo "        accelerate config"

# 5. Reminders
echo ""
echo "[5/5] Other things to set up:"
echo "      - Any local model checkpoints you want to resume from"
echo "      - Set WANDB_PROJECT if you use a custom W&B project name:"
echo "          export WANDB_PROJECT=legato-vision-lora"
echo ""
echo "========================================================"
