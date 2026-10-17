#!/usr/bin/env bash
# Local Python+Triton kernel build: no GPU-name or machine-path decisions.
set -euo pipefail
BENCH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$BENCH_DIR/.." && pwd)
UV=${UV:-uv}
BENCH_VENV=${BENCH_VENV:-"$REPO_DIR/.venv"}
export UV_CACHE_DIR=${UV_CACHE_DIR:-"$REPO_DIR/.cache/uv"}
if [[ $# -gt 1 || ( $# -eq 1 && $1 != --check ) ]]; then
    echo 'Usage: bash benchmarks/setup.sh [--check]' >&2; exit 2
fi
if [[ ${1:-} != --check ]]; then
    if [[ ! -x "$BENCH_VENV/bin/python" ]]; then
        "$UV" venv --python "${PYTHON_VERSION:-3.12}" "$BENCH_VENV"
    fi
    "$UV" pip install --python "$BENCH_VENV/bin/python" -r "$BENCH_DIR/requirements.lock"
    "$UV" pip install --python "$BENCH_VENV/bin/python" 'scikit-build-core==0.11.6' 'cmake==3.31.6'
    CMAKE_ARGS='-DFASTVIDEO_KERNEL_BUILD_EXTENSION=OFF' \
        "$UV" pip install --python "$BENCH_VENV/bin/python" --no-deps --no-build-isolation "$REPO_DIR/fastvideo-kernel"
    "$UV" pip install --python "$BENCH_VENV/bin/python" --no-deps --no-build-isolation -e "$REPO_DIR"
    flash_attn_package=${FLASH_ATTN_PACKAGE:-flash-attn==2.8.3}
    if [[ -z ${FLASH_ATTN_PACKAGE+x} ]] && "$BENCH_VENV/bin/python" -c \
        'import sys, torch; sys.exit(tuple(map(int, torch.__version__.split(".")[:2])) < (2, 12))'; then
        # Our build compatibility adaptation: PyTorch >=2.12 requires C++20.
        # Only setup.py compiler flags change; FlashAttention kernels stay upstream.
        flash_attn_package=$("$BENCH_VENV/bin/python" - "$REPO_DIR/.cache/benchmark" <<'PY'
import hashlib
from pathlib import Path
import sys
import tarfile
import urllib.request

root = Path(sys.argv[1]) / 'flash-attn-build'
root.mkdir(parents=True, exist_ok=True)
archive = root / 'flash_attn-2.8.3.tar.gz'
if not archive.exists():
    urllib.request.urlretrieve(
        'https://files.pythonhosted.org/packages/3b/b2/8d76c41ad7974ee264754709c22963447f7f8134613fd9ce80984ed0dab7/flash_attn-2.8.3.tar.gz',
        archive)
assert hashlib.sha256(archive.read_bytes()).hexdigest() == '1e71dd64a9e0280e0447b8a0c2541bad4bf6ac65bdeaa2f90e51a9e57de0370d'
source = root / 'flash_attn-2.8.3'
if not source.exists():
    with tarfile.open(archive) as package:
        package.extractall(root, filter='data')
setup = source / 'setup.py'
setup.write_text(setup.read_text().replace('-std=c++17', '-std=c++20'))
print(source)
PY
)
    fi
    # Build against this environment's torch/glibc; release wheels may require a newer glibc.
    FLASH_ATTENTION_FORCE_BUILD=${FLASH_ATTENTION_FORCE_BUILD:-TRUE} MAX_JOBS=${MAX_JOBS:-8} \
        "$UV" pip install --python "$BENCH_VENV/bin/python" --no-deps --no-build-isolation \
        "$flash_attn_package"
    "$BENCH_VENV/bin/python" - <<'PY'
import os
from pathlib import Path
import sys
import imageio_ffmpeg

# Expose the existing packaged executable under its native command name.
target = Path(sys.executable).parent / 'ffmpeg'
if not target.exists():
    target.symlink_to(os.path.relpath(imageio_ffmpeg.get_ffmpeg_exe(), target.parent))
PY
fi
"$UV" pip check --python "$BENCH_VENV/bin/python"
cd "$REPO_DIR"
"$BENCH_VENV/bin/python" - <<'PY'
import importlib.metadata as metadata
import torch
import flash_attn
import subprocess
import sys
from pathlib import Path
from fastvideo import VideoGenerator
from fastvideo.pipelines.basic.minimax_h3.minimax_h3_pipeline import MiniMaxH3ModularPipeline
from fastvideo_kernel.block_sparse_attn import block_sparse_attn
print({name: metadata.version(name) for name in
       ['torch', 'torchvision', 'torchaudio', 'transformers', 'diffusers', 'fastvideo-kernel', 'triton', 'av']})
print('CUDA runtime:', torch.version.cuda)
print('FastH3 pipeline:', MiniMaxH3ModularPipeline.__name__)
print('FlashAttention:', flash_attn.__version__)
subprocess.run([str(Path(sys.executable).parent / 'ffmpeg'), '-version'], check=True, capture_output=True)
PY
mkdir -p "$REPO_DIR/.cache/benchmark"
"$UV" pip freeze --python "$BENCH_VENV/bin/python" > "$REPO_DIR/.cache/benchmark/environment.freeze.txt"
