#!/usr/bin/env bash
set -e
echo "=== PMOOS-RAG v0.11.0 — установка (Linux/macOS) ==="
command -v python3 >/dev/null 2>&1 || { echo "Python 3 не найден"; exit 1; }
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
echo "Установка PyTorch (CUDA 12.4; для CPU/Mac см. README)..."
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 || pip install torch==2.6.0
pip install -r requirements.txt
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
echo "Готово. Запуск: ./run.sh"
