#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_root="$repo_root/.venv"

cuda_candidates=("$venv_root"/lib/python*/site-packages/nvidia/cu13)
cuda_root="${cuda_candidates[0]}"

if [[ ! -x "$cuda_root/bin/nvcc" || ! -x "$cuda_root/bin/ptxas" ]]; then
    echo "CUDA 13 compiler tools were not found under $venv_root." >&2
    echo "Install the project's locked dependencies before starting vLLM." >&2
    exit 1
fi

# FlashInfer invokes nvcc by absolute path, but nvcc finds ptxas through PATH.
# Keep both tools on CUDA 13 so CUDA 13 PTX is not sent to CUDA 12.9 ptxas.
export CUDA_HOME="$cuda_root"
export PATH="$CUDA_HOME/bin:$venv_root/bin:$PATH"
export LIBRARY_PATH="$cuda_root/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$cuda_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ ! -e "$cuda_root/lib/libcudart.so" && -e "$cuda_root/lib/libcudart.so.13" ]]; then
    ln -s libcudart.so.13 "$cuda_root/lib/libcudart.so"
fi

exec "$venv_root/bin/vllm" "$@"
