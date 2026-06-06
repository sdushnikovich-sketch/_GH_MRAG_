"""Реранкер кандидатов на базе BAAI/bge-reranker-v2-m3 (CrossEncoder).

Гибридный поиск (dense+BM25) даёт пул кандидатов, а кросс-энкодер точно
переупорядочивает их по релевантности к замечанию. Это заметно повышает
качество ответов — главный приоритет пользователя.

CVE-2025-32434: грузим только safetensors (model_kwargs use_safetensors=True),
device берём из конфигурации (auto -> cuda на 3070ti).
"""
from __future__ import annotations

from typing import Any

from ..config import Config
from ..core.device import resolve_device


class Reranker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model_name = cfg.get("reranker.model", "BAAI/bge-reranker-v2-m3")
        self.enabled = bool(cfg.get("reranker.enabled", True))
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        import os
        from sentence_transformers import CrossEncoder
        from ..paths import models_dir

        device = resolve_device(self.cfg.get("embedding.device", "auto"))
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        kwargs: dict[str, Any] = {
            "device": device,
            "max_length": 512,
            "cache_folder": str(models_dir()),
            # форсируем safetensors (фикс CVE-2025-32434)
            "automodel_args": {"use_safetensors": True},
        }
        if token:
            kwargs["trust_remote_code"] = False
        try:
            self._model = CrossEncoder(self.model_name, **kwargs)
        except TypeError:
            # старые версии sentence-transformers без automodel_args
            kwargs.pop("automodel_args", None)
            self._model = CrossEncoder(self.model_name, device=device,
                                       max_length=512, cache_folder=str(models_dir()))
        return self._model

    def rerank(self, query: str, candidates: list[dict], *, top: int = 8,
               text_key: str = "text") -> list[dict]:
        """Переупорядочивает кандидатов. Если реранкер отключён или недоступен —
        возвращает исходный список (обрезанный до top)."""
        if not candidates:
            return []
        if not self.enabled:
            return candidates[:top]
        try:
            model = self._load()
        except Exception:
            return candidates[:top]
        pairs = [(query, c.get(text_key, "")) for c in candidates]
        scores = model.predict(pairs, batch_size=int(self.cfg.get("embedding.batch_size", 16)))
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        ranked = sorted(candidates, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return ranked[:top]
