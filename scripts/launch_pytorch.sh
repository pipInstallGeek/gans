#!/usr/bin/env bash
# Launch training on all available GPUs (fallback to single GPU/CPU).
#
# This script detects the number of CUDA devices available and, if more than
# one GPU is present, launches the training script under torchrun to enable
# multi-process distributed training.  For single-GPU or CPU-only machines it
# falls back to a normal python invocation.
set -euo pipefail

# Query the number of GPUs using Python to avoid shell dependencies
NGPU=$(python - <<'PY'
import torch, os
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)

echo "Detected $NGPU GPU(s)"

if [ "$NGPU" -gt 1 ]; then
  # Use torchrun for multi-process distributed training
  torchrun --standalone --nproc_per_node="$NGPU" main.py --mode train "$@"
else
  # Single GPU or CPU fallback
  python main.py --mode train "$@"
fi