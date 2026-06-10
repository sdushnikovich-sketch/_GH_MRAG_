#!/usr/bin/env bash
# PMOOS-RAG v0.14.2 — установка (Linux/macOS), устойчивая к медленной сети.
set -e
echo "=== PMOOS-RAG v0.14.2 — установка ==="
command -v python3 >/dev/null 2>&1 || { echo "Python 3 не найден"; exit 1; }
export PIP_DEFAULT_TIMEOUT=120

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip --timeout 120 --retries 10

echo "[torch] CUDA 12.4 (для CPU/Mac будет обычная сборка)..."
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 --timeout 180 --retries 10 \
  || pip install torch==2.6.0 --timeout 180 --retries 10

echo "[deps] попытка 1: индекс по умолчанию (fail-fast)..."
if ! pip install --prefer-binary --timeout 60 --retries 2 -r requirements.txt; then
  echo "[deps] индекс по умолчанию нестабилен — переключаюсь на зеркало..."
  export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
  export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
  if ! pip install --prefer-binary --timeout 120 --retries 10 -r requirements.txt; then
    echo "[deps] ставлю по одному пакету (успешные сохранятся)..."
    grep -vE '^\s*#' requirements.txt | grep -vE '^\s*$' | awk '{print $1}' | while read -r pkg; do
      echo "   -> $pkg"
      pip install --prefer-binary --timeout 120 --retries 10 "$pkg" || true
    done
  fi
fi

python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
echo "Готово. Запуск: ./run.sh  (если что-то не докачалось — запустите снова)"
