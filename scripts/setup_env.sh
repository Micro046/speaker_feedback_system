#!/usr/bin/env bash
set -euo pipefail

cd /notebooks

echo ">>> Setting up Python environment (uv + CUDA 12.1)"

# install uv if missing
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# recreate venv
rm -rf /notebooks/.venv
uv venv --python 3.11
source /notebooks/.venv/bin/activate

# Prefer wheel-provided CUDA libraries over system CUDA libs (fixes nvJitLink/cusparse symbol issues)
export LD_LIBRARY_PATH="/notebooks/.venv/lib/python3.11/site-packages/nvidia/nvjitlink/lib:/notebooks/.venv/lib/python3.11/site-packages/nvidia/cusparse/lib:/notebooks/.venv/lib/python3.11/site-packages/nvidia/cublas/lib:/notebooks/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"

# also bake it into future activations
if ! grep -q "Prefer wheel-provided CUDA libraries" /notebooks/.venv/bin/activate; then
  cat >> /notebooks/.venv/bin/activate <<'EOT'

# Prefer wheel-provided CUDA libraries over system CUDA libs (fixes nvJitLink/cusparse symbol issues)
export LD_LIBRARY_PATH="/notebooks/.venv/lib/python3.11/site-packages/nvidia/nvjitlink/lib:/notebooks/.venv/lib/python3.11/site-packages/nvidia/cusparse/lib:/notebooks/.venv/lib/python3.11/site-packages/nvidia/cublas/lib:/notebooks/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"
EOT
fi

echo ">>> Installing build tooling"
uv pip install -U pip setuptools wheel ninja pybind11 \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  --index-strategy unsafe-best-match

echo ">>> Installing requirements (excluding detectron2 first)"
grep -vE '^\s*detectron2\s*@\s*git\+https://github\.com/facebookresearch/detectron2\.git\s*$' \
  /notebooks/requirements.txt > /notebooks/requirements.no_d2.txt

uv pip install -r /notebooks/requirements.no_d2.txt \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  --index-strategy unsafe-best-match

echo ">>> Torch sanity check"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"

echo ">>> Installing detectron2 (no build isolation)"
uv pip install --no-build-isolation \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  --index-strategy unsafe-best-match \
  "detectron2 @ git+https://github.com/facebookresearch/detectron2.git"

echo ">>> Installing Jupyter kernel"
uv pip install ipykernel \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  --index-strategy unsafe-best-match

python -m ipykernel install --user --name uv-cu121 --display-name "Python (uv cu121)"

echo ">>> Baking CUDA loader fix into kernel.json"
KDIR="$(jupyter --data-dir)/kernels/uv-cu121"
mkdir -p "$KDIR"

cat > "$KDIR/kernel.json" << 'KJSON'
{
  "argv": ["/notebooks/.venv/bin/python", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
  "display_name": "Python (uv cu121)",
  "language": "python",
  "env": {
    "LD_LIBRARY_PATH": "/notebooks/.venv/lib/python3.11/site-packages/nvidia/nvjitlink/lib:/notebooks/.venv/lib/python3.11/site-packages/nvidia/cusparse/lib:/notebooks/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib:/notebooks/.venv/lib/python3.11/site-packages/nvidia/cublas/lib"
  }
}
KJSON

echo ">>> Final sanity check"
python -c "import detectron2; import torch; print('detectron2 ok; cuda avail:', torch.cuda.is_available())"

echo ">>> Environment ready"