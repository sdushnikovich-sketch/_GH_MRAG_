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

        device = resolve_device(self.cfg.get("embedding.device", "auto"))
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        # ВАЖНО: НЕ передаём cache_folder/cache_dir — у CrossEncoder в
        # sentence-transformers 3.3 такого параметра нет (был краш TypeError).
        # Кэш моделей единый и задаётся переменной HF_HOME (см. config.py):
        # <данные>/models/hub — туда же качают setup_models.py и предзагрузка М2.
        base: dict[str, Any] = {
            "device": device,
            "max_length": 512,
        }
        if token:
            base["trust_remote_code"] = False

        def _try(force_safetensors: bool):
            kw = dict(base)
            if force_safetensors:
                # форсируем safetensors (фикс CVE-2025-32434)
                kw["automodel_args"] = {"use_safetensors": True}
            # Сигнатура CrossEncoder различается между версиями
            # sentence-transformers: при TypeError снимаем необязательные
            # аргументы по одному (лесенка), а не один фиксированный.
            while True:
                try:
                    return CrossEncoder(self.model_name, **kw)
                except TypeError:
                    for opt in ("automodel_args", "trust_remote_code", "max_length"):
                        if opt in kw:
                            kw.pop(opt)
                            break
                    else:
                        raise

        try:
            self._model = _try(True)
        except OSError as e:
            # Репозиторий модели без safetensors (только .bin) — безопасный откат
            # (torch>=2.6: weights_only=True по умолчанию).
            if "model.safetensors" in str(e):
                print(f"[reranker] у {self.model_name} нет model.safetensors — "
                      f"загружаю .bin (безопасно: torch>=2.6, weights_only)", flush=True)
                self._model = _try(False)
            else:
                raise
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
