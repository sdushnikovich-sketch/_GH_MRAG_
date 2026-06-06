"""Эмбеддинги (BAAI/bge-m3) с исправлением сразу нескольких проблем из логов:

  1. «загрузка BAAI/bge-m3 на cpu» вместо GPU — теперь device определяется
     автоматически (cuda при наличии), и это видно в логе.
  2. Падение ValueError: torch.load … CVE-2025-32434 — модель грузится из
     safetensors (use_safetensors=True), что снимает запрет на torch.load.
  3. Предупреждение HF про неавторизованные запросы — токен берётся из env,
     если задан; иначе работаем анонимно без падения.
  4. «нет кэша эмбеддингов» — добавлен sqlite-кэш sha256(model+text):
     повторная индексация почти бесплатна, что критично для пауза/возобновление.
  5. OOM на 8 ГБ VRAM — при нехватке памяти автоматически уменьшаем батч и
     чистим кэш CUDA.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import threading

import numpy as np

from ..config import Config
from ..paths import emb_cache_path, models_dir

_LOCK = threading.Lock()


def pick_device(requested: str = "auto") -> str:
    if requested and requested != "auto":
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class _EmbCache:
    """Дисковый кэш векторов: ключ = sha256(model|text).

    WAL-режим (по замечанию ревью) — устойчивее к параллельным воркерам и
    'database is locked'. Колонка last_used позволяет чистить старые записи,
    чтобы кэш не рос бесконечно.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.con = sqlite3.connect(str(emb_cache_path()), check_same_thread=False)
        try:
            self.con.execute("PRAGMA journal_mode=WAL")
            self.con.execute("PRAGMA synchronous=NORMAL")
            self.con.execute("PRAGMA busy_timeout=5000")
        except Exception:  # noqa: BLE001
            pass
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB, last_used REAL)")
        # миграция старой схемы без last_used
        cols = [r[1] for r in self.con.execute("PRAGMA table_info(emb)").fetchall()]
        if "last_used" not in cols:
            try:
                self.con.execute("ALTER TABLE emb ADD COLUMN last_used REAL")
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def key(model: str, text: str) -> str:
        return hashlib.sha256(f"{model}\u0000{text}".encode("utf-8")).hexdigest()

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        if not keys:
            return out
        with _LOCK:
            qmarks = ",".join("?" * len(keys))
            rows = self.con.execute(
                f"SELECT k, v FROM emb WHERE k IN ({qmarks})", keys
            ).fetchall()
            if rows:
                import time as _t
                now = _t.time()
                self.con.executemany("UPDATE emb SET last_used=? WHERE k=?",
                                     [(now, k) for k, _ in rows])
                self.con.commit()
        for k, v in rows:
            out[k] = np.frombuffer(v, dtype=np.float32)
        return out

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        if not items:
            return
        import time as _t
        now = _t.time()
        with _LOCK:
            self.con.executemany(
                "INSERT OR REPLACE INTO emb (k, v, last_used) VALUES (?, ?, ?)",
                [(k, v.astype(np.float32).tobytes(), now) for k, v in items.items()],
            )
            self.con.commit()

    def cleanup(self, *, max_rows: int = 2_000_000) -> int:
        """Удалить самые старые записи, если кэш превысил max_rows. Возвращает
        число удалённых. Вызывать по желанию (обслуживание)."""
        with _LOCK:
            (n,) = self.con.execute("SELECT COUNT(*) FROM emb").fetchone()
            if n <= max_rows:
                return 0
            to_del = n - max_rows
            self.con.execute(
                "DELETE FROM emb WHERE k IN (SELECT k FROM emb ORDER BY last_used ASC LIMIT ?)",
                (to_del,))
            self.con.commit()
            return to_del


class Embedder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model_name = cfg.get("embedding.model", "BAAI/bge-m3")
        self.device = pick_device(cfg.get("embedding.device", "auto"))
        self.batch_size = int(cfg.get("embedding.batch_size", 16))
        self.max_length = int(cfg.get("embedding.max_length", 1024))
        self.use_safetensors = bool(cfg.get("embedding.use_safetensors", True))
        self._model = None
        self._dim: int | None = None
        self._cache: _EmbCache | None = None

    def _load(self):
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer
        print(f"[embeddings] загрузка {self.model_name} на {self.device}", flush=True)
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        kwargs = dict(device=self.device, cache_folder=str(models_dir()))
        # фикс CVE: заставляем грузить safetensors, а не .bin через torch.load
        model_kwargs = {"use_safetensors": True} if self.use_safetensors else {}
        try:
            self._model = SentenceTransformer(
                self.model_name, model_kwargs=model_kwargs, token=token, **kwargs
            )
        except TypeError:
            # старые версии sentence-transformers без model_kwargs/token
            self._model = SentenceTransformer(self.model_name, **kwargs)
        try:
            self._model.max_seq_length = self.max_length
        except Exception:
            pass
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            m = self._load()
            # метод переименован в новых версиях — поддержим оба
            if hasattr(m, "get_embedding_dimension"):
                self._dim = int(m.get_embedding_dimension())
            else:
                self._dim = int(m.get_sentence_embedding_dimension())
        return self._dim

    def _cache_obj(self) -> _EmbCache:
        if self._cache is None:
            self._cache = _EmbCache(self.dim)
        return self._cache

    def _encode(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        bs = self.batch_size
        while True:
            try:
                vecs = model.encode(
                    texts, batch_size=bs, normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False,
                )
                return vecs.astype(np.float32)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and bs > 1:
                    bs = max(1, bs // 2)
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    print(f"[embeddings] OOM — уменьшаю батч до {bs}", flush=True)
                    continue
                raise

    def embed(self, texts: list[str], *, use_cache: bool = True) -> np.ndarray:
        """Векторизует тексты с использованием дискового кэша."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if not use_cache:
            return self._encode(texts)
        cache = self._cache_obj()
        keys = [cache.key(self.model_name, t) for t in texts]
        have = cache.get_many(list(dict.fromkeys(keys)))
        miss_idx = [i for i, k in enumerate(keys) if k not in have]
        if miss_idx:
            new_vecs = self._encode([texts[i] for i in miss_idx])
            new_items = {keys[i]: new_vecs[j] for j, i in enumerate(miss_idx)}
            cache.put_many(new_items)
            have.update(new_items)
        return np.vstack([have[k] for k in keys]).astype(np.float32)

    def embed_documents(self, texts: list[str], *, use_cache: bool = True) -> np.ndarray:
        return self.embed(texts, use_cache=use_cache)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        # для bge-m3 префиксы не обязательны; кэшируем запросы тоже
        return self.embed(texts, use_cache=True)
