"""Гибридный поиск: плотные векторы (Qdrant) + BM25 + RRF, затем реранк.

Зачем гибрид: канцелярский язык замечаний плохо ловится только семантикой,
а точные термины/номера ГОСТ/обозначения веществ лучше находит лексический
BM25. Объединяем результаты через Reciprocal Rank Fusion и финально уточняем
кросс-энкодером.

BM25 строится по чанкам коллекции проекта. Чтобы не гонять весь индекс на
каждый запрос, корпус кэшируется в памяти на время сессии пайплайна.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..config import Config
from ..index.embeddings import Embedder
from ..index.vectorstore import VectorStore, collection_name
from .query_expansion import expand_query
from .reranker import Reranker

_WORD = re.compile(r"[\w\-]+", re.UNICODE)


def _tok(text: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(text or "")]


def expand_hits(hits: list[dict], index: dict, *, neighbors: int = 1,
                merge_tables: bool = True) -> list[dict]:
    """Расширить найденные чанки контекстом (пункт 3B).

    Для каждого попадания добавляет соседние чанки того же файла (±neighbors по
    chunk_index); если чанк табличный и merge_tables=True — захватывает все ПОДРЯД
    идущие табличные чанки (восстанавливает таблицу целиком, напр. таблицу выбросов).
    Текст попадания заменяется на склейку в порядке chunk_index. Дедупликация по
    id, чтобы один и тот же блок не уходил в ИИ много раз.

    index: dict[(file, chunk_index)] -> {"id", "text", "is_table"}.
    """
    if not index:
        return hits
    used: set[str] = set()
    out: list[dict] = []
    for h in hits:
        pl = h.get("payload", {}) or {}
        f = pl.get("file")
        idx = pl.get("chunk_index")
        if f is None or idx is None or (f, idx) not in index:
            out.append(h)
            continue
        want = set(range(idx - neighbors, idx + neighbors + 1))
        if merge_tables and pl.get("is_table"):
            lo = idx
            while (f, lo - 1) in index and index[(f, lo - 1)].get("is_table"):
                lo -= 1
            hi = idx
            while (f, hi + 1) in index and index[(f, hi + 1)].get("is_table"):
                hi += 1
            want |= set(range(lo, hi + 1))
        parts, merged = [], []
        for i in sorted(want):
            cell = index.get((f, i))
            if not cell or cell["id"] in used:
                continue
            if cell.get("text"):
                parts.append(cell["text"])
                merged.append(cell["id"])
        if not parts:
            out.append(h)
            continue
        used.update(merged)
        new = dict(h)
        new["text"] = "\n…\n".join(parts).strip()
        new["expanded_from"] = merged
        out.append(new)
    return out


@dataclass
class _Corpus:
    ids: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    payloads: list[dict] = field(default_factory=list)
    bm25: Any = None
    by_index: dict = field(default_factory=dict)  # (file, chunk_index) -> {id,text,is_table}


class HybridRetriever:
    def __init__(self, cfg: Config, *, embedder: Embedder | None = None,
                 store: VectorStore | None = None, reranker: Reranker | None = None):
        self.cfg = cfg
        self.embedder = embedder or Embedder(cfg)
        self.store = store or VectorStore(cfg, dim=self.embedder.dim)
        self.reranker = reranker or Reranker(cfg)
        self._corpus_cache: dict[str, _Corpus] = {}

    # ---- BM25 corpus -----------------------------------------------------
    def _load_corpus(self, project: str) -> _Corpus:
        if project in self._corpus_cache:
            return self._corpus_cache[project]
        corp = _Corpus()
        try:
            from qdrant_client.http import models as qm  # noqa: F401
            client = self.store.client()
            name = collection_name(project)
            offset = None
            while True:
                points, offset = client.scroll(
                    collection_name=name, with_payload=True, with_vectors=False,
                    limit=512, offset=offset,
                )
                for p in points:
                    pl = p.payload or {}
                    corp.ids.append(str(p.id))
                    corp.texts.append(pl.get("text", ""))
                    corp.payloads.append(pl)
                    f = pl.get("file")
                    ci = pl.get("chunk_index")
                    if f is not None and ci is not None:
                        corp.by_index[(f, ci)] = {
                            "id": str(p.id), "text": pl.get("text", ""),
                            "is_table": bool(pl.get("is_table")),
                        }
                if offset is None:
                    break
        except Exception:
            pass
        if corp.texts and self.cfg.get("retrieval.use_bm25", True):
            try:
                from rank_bm25 import BM25Okapi
                corp.bm25 = BM25Okapi([_tok(t) for t in corp.texts])
            except Exception:
                corp.bm25 = None
        self._corpus_cache[project] = corp
        return corp

    # ---- single components ----------------------------------------------
    def _dense(self, project: str, query: str, *, candidates: int,
               sections, exclude_sections) -> list[dict]:
        qv = self.embedder.embed_queries([query])[0]
        return self.store.search(project, qv, top=candidates,
                                 sections=sections, exclude_sections=exclude_sections)

    def _bm25(self, project: str, query: str, *, candidates: int,
              sections, exclude_sections) -> list[dict]:
        corp = self._load_corpus(project)
        if not corp.bm25:
            return []
        scores = corp.bm25.get_scores(_tok(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[dict] = []
        sset = set(sections) if sections else None
        xset = set(exclude_sections) if exclude_sections else None
        for i in order:
            pl = corp.payloads[i]
            sec = pl.get("section")
            if sset and sec not in sset:
                continue
            if xset and sec in xset:
                continue
            out.append({"id": corp.ids[i], "score": float(scores[i]),
                        "text": corp.texts[i], "payload": pl})
            if len(out) >= candidates:
                break
        return out

    @staticmethod
    def _rrf(result_lists: list[list[dict]], *, k: int = 60) -> list[dict]:
        """Reciprocal Rank Fusion по id чанка."""
        agg: dict[str, dict] = {}
        for lst in result_lists:
            for rank, item in enumerate(lst):
                cid = item["id"]
                node = agg.setdefault(cid, {"item": item, "rrf": 0.0})
                node["rrf"] += 1.0 / (k + rank + 1)
        fused = sorted(agg.values(), key=lambda x: x["rrf"], reverse=True)
        out = []
        for n in fused:
            it = dict(n["item"])
            it["rrf_score"] = n["rrf"]
            out.append(it)
        return out

    # ---- public API ------------------------------------------------------
    def search(self, project: str, query: str, *, top: int | None = None,
               candidates: int | None = None, sections: list[str] | None = None,
               exclude_sections: list[str] | None = None,
               use_expansion: bool | None = None) -> list[dict]:
        top = top or int(self.cfg.get("retrieval.top_k", 8))
        candidates = candidates or int(self.cfg.get("retrieval.candidates", 40))
        use_expansion = self.cfg.get("retrieval.use_query_expansion", True) if use_expansion is None else use_expansion

        queries = [query]
        if use_expansion:
            queries = expand_query(query, self.cfg, n=int(self.cfg.get("retrieval.expansions", 3)))

        lists: list[list[dict]] = []
        for q in queries:
            lists.append(self._dense(project, q, candidates=candidates,
                                     sections=sections, exclude_sections=exclude_sections))
        # BM25 по исходному замечанию (точные термины важнее перефраза)
        if self.cfg.get("retrieval.use_bm25", True):
            lists.append(self._bm25(project, query, candidates=candidates,
                                    sections=sections, exclude_sections=exclude_sections))

        fused = self._rrf(lists)
        pool = fused[: max(candidates, top)]

        if self.cfg.get("retrieval.use_rerank", True) and pool:
            results = self.reranker.rerank(query, pool, top=top)
        else:
            results = pool[:top]

        # расширение контекста соседями + склейка таблиц (пункт 3B)
        if self.cfg.get("retrieval.expand_context", True) and results:
            corp = self._load_corpus(project)
            results = expand_hits(
                results, corp.by_index,
                neighbors=int(self.cfg.get("retrieval.context_neighbors", 1)),
                merge_tables=bool(self.cfg.get("retrieval.merge_tables", True)),
            )
        return results

    def batch_search(self, project: str, queries: list[str], **kw) -> list[list[dict]]:
        """Подготавливает корпус один раз и ищет по списку замечаний.

        Корпус BM25 и модель эмбеддингов загружаются единожды — это и есть
        батчевый retrieval, о котором писали ревью-ИИ (75 замечаний без
        повторной загрузки тяжёлых ресурсов)."""
        self._load_corpus(project)
        return [self.search(project, q, **kw) for q in queries]
