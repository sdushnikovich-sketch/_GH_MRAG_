"""Фоновая индексация проекта в векторную БД (МОДУЛЬ 2).

Ключевые требования пользователя, которые здесь закрыты:
  * прогресс виден (сколько проиндексировано / сколько осталось) — index_state.json;
  * пауза и возобновление, переживающие закрытие вкладки и перезапуск процесса;
  * работа в фоне (отдельным процессом, не блокирующим UI);
  * БД хранится ОТДЕЛЬНО от приложения (см. paths.qdrant_dir), не индексируется заново;
  * дедупликация по sha256 содержимого (existing_doc_shas);
  * оптимизация под 3070ti — эмбеддинги на GPU, батчи, кэш.

Состояние (index_state.json):
{
  "status": "running|paused|done|error|idle",
  "pause_requested": false,
  "pid": 12345,
  "total_files": N, "done_files": M,
  "total_chunks": X, "done_chunks": Y,
  "files": {"<rel>": {"status": "pending|done|skipped|error", "chunks": n, "section": "..."}},
  "current_file": "...", "message": "...", "updated_at": "ISO"
}
"""
from __future__ import annotations

import json
import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import Config, load_config
from ..paths import project_paths
from ..ingest.sections import classify_filename, detect_version_hint
from ..ingest.loaders import extract_file, SUPPORTED_EXT
from ..ingest.dedup import doc_fingerprint
from ..ingest.chunking import build_chunks


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_state(project: str) -> dict[str, Any]:
    p = project_paths(project)["index_state"]
    if not p.exists():
        return {"status": "idle", "pause_requested": False, "total_files": 0,
                "done_files": 0, "total_chunks": 0, "done_chunks": 0, "files": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "idle", "pause_requested": False, "files": {}}


def write_state(project: str, state: dict[str, Any]) -> None:
    state["updated_at"] = _now()
    p = project_paths(project)["index_state"]
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def request_pause(project: str) -> None:
    st = read_state(project)
    st["pause_requested"] = True
    write_state(project, st)


def clear_pause(project: str) -> None:
    st = read_state(project)
    st["pause_requested"] = False
    write_state(project, st)


def is_running(project: str) -> bool:
    st = read_state(project)
    if st.get("status") != "running":
        return False
    pid = st.get("pid")
    if not pid:
        return False
    return _pid_alive(int(pid))


def _pid_alive(pid: int) -> bool:
    try:
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                                 capture_output=True, text=True)
            return str(pid) in out.stdout
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _iter_source_files(upload_dir: Path) -> list[Path]:
    files: list[Path] = []
    for p in sorted(upload_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT:
            files.append(p)
    return files


def run_indexing(project: str, cfg: Config | None = None, *, object_type: str | None = None) -> dict:
    """Синхронная индексация (вызывается в фоновом процессе или напрямую).

    Переиндексация безопасна: уже загруженные документы (по doc_sha) пропускаются,
    ID чанков детерминированы, поэтому повторная загрузка не плодит дубли.
    """
    cfg = cfg or load_config()
    object_type = object_type or cfg.get("object_type", "площадной")
    paths = project_paths(project)
    upload_dir = paths["uploads"]
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Ленивая загрузка тяжёлых модулей — чтобы UI стартовал быстро.
    from .embeddings import Embedder
    from .vectorstore import VectorStore

    embedder = Embedder(cfg)
    store = VectorStore(cfg, dim=embedder.dim)
    store.ensure_collection(project)
    known_shas = store.existing_doc_shas(project)

    files = _iter_source_files(upload_dir)
    state = read_state(project)
    state.update({
        "status": "running", "pause_requested": False, "pid": os.getpid(),
        "total_files": len(files), "done_files": 0,
        "message": "Индексация запущена", "current_file": "",
    })
    state.setdefault("files", {})
    state.setdefault("total_chunks", 0)
    state.setdefault("done_chunks", 0)
    write_state(project, state)

    try:
        for fpath in files:
            rel = str(fpath.relative_to(upload_dir))
            finfo = state["files"].get(rel, {})
            if finfo.get("status") in ("done", "skipped"):
                state["done_files"] += 1
                write_state(project, state)
                continue

            # Проверка паузы перед каждым файлом.
            if read_state(project).get("pause_requested"):
                state["status"] = "paused"
                state["message"] = "Пауза (возобновляемо)"
                write_state(project, state)
                return state

            state["current_file"] = rel
            state["message"] = f"Обработка: {rel}"
            write_state(project, state)

            try:
                cls = classify_filename(fpath.name, object_type, top=1)
                section = cls[0]["code"] if cls else "UNKNOWN"
                ver = detect_version_hint(fpath.name) or ""
                pages = extract_file(
                    fpath,
                    ocr=cfg.get("ocr.enabled", True),
                    min_text_chars=cfg.get("ocr.min_text_chars", 200),
                    ocr_lang=cfg.get("ocr.lang", "rus+eng"),
                )
                sha = doc_fingerprint(pages)
                # подпись содержимого для контентного сравнения версий (пункт 4)
                try:
                    from ..ingest.dedup import content_signature
                    from ..versioning.versions import save_content_sig
                    save_content_sig(project, fpath.name, content_signature(pages))
                except Exception:  # noqa: BLE001
                    pass
                if sha in known_shas:
                    state["files"][rel] = {"status": "skipped", "chunks": 0,
                                           "section": section, "version": ver,
                                           "reason": "дубликат (sha256)"}
                    state["done_files"] += 1
                    write_state(project, state)
                    continue

                chunks = build_chunks(
                    project=project, file_rel=rel, section_code=section,
                    doc_sha=sha, pages=pages,
                    size=cfg.get("chunking.size", 1200),
                    overlap=cfg.get("chunking.overlap", 200),
                    min_chunk=cfg.get("chunking.min_chunk", 80),
                )
                if chunks:
                    vectors = embedder.embed_documents([c["text"] for c in chunks])
                    store.upsert_chunks(project, chunks, vectors)
                    known_shas.add(sha)

                state["files"][rel] = {"status": "done", "chunks": len(chunks),
                                       "section": section, "version": ver}
                state["done_chunks"] = state.get("done_chunks", 0) + len(chunks)
                state["total_chunks"] = state.get("total_chunks", 0) + len(chunks)
                state["done_files"] += 1
                write_state(project, state)
            except Exception as e:  # noqa: BLE001
                state["files"][rel] = {"status": "error", "chunks": 0, "error": str(e)}
                state["done_files"] += 1
                state["message"] = f"Ошибка в {rel}: {e}"
                write_state(project, state)

        state["status"] = "done"
        state["current_file"] = ""
        state["message"] = "Индексация завершена"
        write_state(project, state)
        return state
    except Exception as e:  # noqa: BLE001
        state["status"] = "error"
        state["message"] = f"Критическая ошибка: {e}"
        write_state(project, state)
        return state


def start_background(project: str, *, object_type: str | None = None) -> int:
    """Запускает индексацию отдельным detached-процессом.

    Процесс продолжит работу даже если вкладка Streamlit закрыта. Для
    возобновления после паузы/перезапуска просто вызвать ещё раз — уже
    готовые файлы пропускаются.
    """
    clear_pause(project)
    env = dict(os.environ)
    args = [sys.executable, "-m", "pmoos.index.indexer", "--project", project]
    if object_type:
        args += ["--object-type", object_type]
    kwargs: dict[str, Any] = {"env": env}
    if os.name == "nt":
        # CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS
        kwargs["creationflags"] = 0x00000200 | 0x00000008
    else:
        kwargs["start_new_session"] = True
    kwargs["stdout"] = subprocess.DEVNULL
    kwargs["stderr"] = subprocess.DEVNULL
    proc = subprocess.Popen(args, **kwargs)
    st = read_state(project)
    st["pid"] = proc.pid
    st["status"] = "running"
    st["message"] = "Фоновая индексация запущена"
    write_state(project, st)
    return proc.pid


def progress_summary(project: str) -> dict:
    st = read_state(project)
    total_f = st.get("total_files", 0)
    done_f = st.get("done_files", 0)
    pct = (done_f / total_f * 100.0) if total_f else 0.0
    return {
        "status": st.get("status", "idle"),
        "files_done": done_f, "files_total": total_f,
        "chunks_done": st.get("done_chunks", 0),
        "percent": round(pct, 1),
        "current_file": st.get("current_file", ""),
        "message": st.get("message", ""),
        "running": is_running(project),
    }


def _main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Фоновая индексация PMOOS-RAG")
    ap.add_argument("--project", required=True)
    ap.add_argument("--object-type", default=None)
    a = ap.parse_args()
    run_indexing(a.project, object_type=a.object_type)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
