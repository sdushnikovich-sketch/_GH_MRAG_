"""Векторная база (Qdrant). Исправления:

  * размерность берётся из модели (embedder.dim), а не зашита EMBEDDING_DIM=1024
    — иначе при смене модели Qdrant получает векторы другой длины;
  * стабильные ID чанков (приходят из chunking.build_chunks);
  * корректное закрытие клиента через atexit — это убирает ошибку при выходе
    «Exception ignored in QdrantClient.__del__ … sys.meta_path is None»;
  * два режима: embedded (path, по умолчанию) и server (url, Docker) — для
    больших баз рекомендуется server.

База лежит в каталоге данных (qdrant_dir()), отдельно от приложения.
"""
from __future__ import annotations

import atexit

from ..config import Config
from ..paths import qdrant_dir, slugify

_CLIENTS: list = []


def _register_close(client) -> None:
    _CLIENTS.append(client)


@atexit.register
def _close_all() -> None:  # вызывается при нормальном завершении процесса
    for c in _CLIENTS:
        try:
            c.close()
        except Exception:
            pass
    _CLIENTS.clear()


def collection_name(project: str) -> str:
    return f"pmoos_{slugify(project)}"


class VectorStore:
    def __init__(self, cfg: Config, dim: int):
        self.cfg = cfg
        self.dim = dim
        self._client = None

    def client(self):
        if self._client is not None:
            return self._client
        from qdrant_client import QdrantClient
        mode = self.cfg.get("qdrant.mode", "embedded")
        if mode == "server":
            url = self.cfg.get("qdrant.url", "http://localhost:6333")
            self._client = QdrantClient(url=url, timeout=60)
        else:
            self._client = QdrantClient(path=str(qdrant_dir()))
        _register_close(self._client)
        return self._client

    def ensure_collection(self, project: str) -> None:
        from qdrant_client.http import models as qm
        c = self.client()
        name = collection_name(project)
        existing = {col.name for col in c.get_collections().collections}
        if name not in existing:
            c.create_collection(
                collection_name=name,
                vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
            )
            # payload-индексы (в server-режиме ускоряют фильтрацию;
            # в embedded — игнорируются, но ошибки не вызывают)
            for field in ("project", "section", "file", "doc_sha"):
                try:
                    c.create_payload_index(
                        collection_name=name, field_name=field,
                        field_schema=qm.PayloadSchemaType.KEYWORD,
                    )
                except Exception:
                    pass

    def existing_doc_shas(self, project: str) -> set[str]:
        """Список уже проиндексированных отпечатков документов (для дедупликации)."""
        from qdrant_client.http import models as qm
        c = self.client()
        name = collection_name(project)
        existing = {col.name for col in c.get_collections().collections}
        if name not in existing:
            return set()
        shas: set[str] = set()
        offset = None
        while True:
            points, offset = c.scroll(
                collection_name=name, with_payload=["doc_sha"],
                with_vectors=False, limit=1000, offset=offset,
            )
            for p in points:
                sha = (p.payload or {}).get("doc_sha")
                if sha:
                    shas.add(sha)
            if offset is None:
                break
        return shas

    def upsert_chunks(self, project: str, chunks: list[dict], vectors) -> None:
        from qdrant_client.http import models as qm
        c = self.client()
        name = collection_name(project)
        points = [
            qm.PointStruct(id=ch["id"], vector=vectors[i].tolist(),
                           payload={**ch["payload"], "text": ch["text"]})
            for i, ch in enumerate(chunks)
        ]
        for i in range(0, len(points), 128):
            c.upsert(collection_name=name, points=points[i : i + 128])

    def delete_by_file(self, project: str, file_rel: str) -> None:
        from qdrant_client.http import models as qm
        c = self.client()
        c.delete(
            collection_name=collection_name(project),
            points_selector=qm.FilterSelector(filter=qm.Filter(
                must=[qm.FieldCondition(key="file", match=qm.MatchValue(value=file_rel))]
            )),
        )

    def search(self, project: str, query_vector, *, top: int = 8,
               sections: list[str] | None = None,
               exclude_sections: list[str] | None = None) -> list[dict]:
        from qdrant_client.http import models as qm
        c = self.client()
        must = []
        if sections:
            must.append(qm.FieldCondition(key="section", match=qm.MatchAny(any=list(sections))))
        must_not = []
        if exclude_sections:
            must_not.append(qm.FieldCondition(key="section", match=qm.MatchAny(any=list(exclude_sections))))
        flt = qm.Filter(must=must or None, must_not=must_not or None) if (must or must_not) else None
        hits = c.search(
            collection_name=collection_name(project), query_vector=query_vector.tolist(),
            limit=top, query_filter=flt, with_payload=True,
        )
        return [{"id": h.id, "score": float(h.score), "text": (h.payload or {}).get("text", ""),
                 "payload": h.payload or {}} for h in hits]

    def count(self, project: str) -> int:
        c = self.client()
        try:
            return int(c.count(collection_name=collection_name(project)).count)
        except Exception:
            return 0
