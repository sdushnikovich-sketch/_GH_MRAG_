"""Работа с локальным Ollama.

Замечание пользователя: «предлагать в списке все локальные модели через
провайдера ollama, найденные и установленные».
Здесь — определение запущен ли ollama и список реально установленных моделей.
"""
from __future__ import annotations

import shutil
import subprocess


def ollama_available(base_url: str = "http://localhost:11434") -> bool:
    import requests
    try:
        r = requests.get(base_url.rstrip("/") + "/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return shutil.which("ollama") is not None


def list_installed_models(base_url: str = "http://localhost:11434") -> list[str]:
    """Список установленных моделей. Сначала через HTTP API, затем через CLI."""
    import requests
    # 1) HTTP API
    try:
        r = requests.get(base_url.rstrip("/") + "/api/tags", timeout=5)
        if r.status_code == 200:
            data = r.json() or {}
            names = [m.get("name", "") for m in data.get("models", [])]
            names = [n for n in names if n]
            if names:
                return sorted(set(names))
    except Exception:
        pass
    # 2) CLI fallback: `ollama list`
    try:
        out = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        names = []
        for line in out.stdout.splitlines()[1:]:
            parts = line.split()
            if parts:
                names.append(parts[0])
        return sorted(set(n for n in names if n))
    except Exception:
        return []
