"""Блок 1 (МОДУЛЬ 4): найти ответ на каждое замечание ПМООС с указанием источника.

Логика:
  1. Загрузить ВСЕ замечания (ingest.remarks) — не теряем ни одного.
  2. Один раз поднять ресурсы (эмбеддер, BM25-корпус, реранкер) — батчевый
     retrieval вместо N независимых поисков (ускорение для 75 замечаний).
  3. По каждому замечанию найти релевантные фрагменты разделов-источников
     (ТКР/ПОС/ИЭИ/…) с провенансом (раздел/файл/страница).
  4. Сгенерировать ответ ИИ (провайдер/модель — автоматически под модуль),
     параллельными запросами (batch_chat).
  5. Прогнать проверку согласованности (consistency) и каскад (cascade).
  6. Сохранить предложения в answers.json со статусом «proposed» — финальное
     принятие за пользователем (human-in-the-loop).

Ничего из проектных файлов не сохраняется отдельно — работаем по индексу.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import Config, load_config
from ..paths import project_paths
from ..ingest.remarks import load_remarks, Remark
from ..ingest.sections import source_section_codes
from ..retrieval.hybrid import HybridRetriever
from ..core.ai_providers import batch_chat
from ..core.json_utils import extract_json_safe
from .consistency import compare
from ..graph.cascade import explain_cascade, downstream

_SYS = (
    "Ты — главный инженер-эколог, готовишь ответы на замечания государственной "
    "экспертизы к разделу ПМООС/ООС проектной документации (Постановление "
    "Правительства РФ №87). Отвечай профессионально, по существу, со ссылками на "
    "конкретные данные проекта и действующие нормативы. Не выдумывай данные, "
    "которых нет в предоставленных фрагментах: если данных не хватает — прямо "
    "укажи, какой раздел/расчёт нужно дополнить."
)

_USER_TMPL = (
    "ЗАМЕЧАНИЕ ЭКСПЕРТА №{num}:\n«{remark}»\n\n"
    "НАЙДЕННЫЕ ФРАГМЕНТЫ ПРОЕКТНОЙ ДОКУМЕНТАЦИИ (источники):\n{context}\n\n"
    "Сформируй ответ строго в формате JSON:\n"
    "{{\n"
    '  "answer": "текст ответа эксперту (что сделано/уточнено в ПМООС)",\n'
    '  "correction": "какую правку внести в раздел ПМООС (конкретно)",\n'
    '  "used_sources": [номера фрагментов, реально использованных, напр. [1,3]],\n'
    '  "confidence": "high|medium|low",\n'
    '  "missing_data": "чего не хватает в документации (или пусто)"\n'
    "}}\n"
    "Верни ТОЛЬКО JSON."
)


def _format_context(hits: list[dict], limit: int = 8) -> tuple[str, list[dict]]:
    lines, srcs = [], []
    for i, h in enumerate(hits[:limit], 1):
        pl = h.get("payload", {})
        loc = pl.get("loc", "")
        file = pl.get("file", "")
        sec = pl.get("section", "")
        snippet = (h.get("text", "") or "")[:900]
        lines.append(f"[{i}] (раздел: {sec}; файл: {file}; место: {loc})\n{snippet}")
        srcs.append({"n": i, "file": file, "loc": loc, "section": sec,
                     "score": round(float(h.get("rerank_score", h.get("rrf_score", h.get("score", 0.0)))), 4),
                     "snippet": snippet[:300]})
    return "\n\n".join(lines), srcs


def _process(raw: str) -> dict:
    data = extract_json_safe(raw, expect="object") or {}
    if not isinstance(data, dict):
        data = {}
    return data


def run_block1(project: str, cfg: Config | None = None, *,
               remarks_path: str | Path | None = None,
               object_type: str | None = None,
               progress=None) -> dict[str, Any]:
    cfg = cfg or load_config()
    object_type = object_type or cfg.get("object_type", "площадной")
    paths = project_paths(project)

    # 1) замечания
    if remarks_path is None:
        # ищем файл замечаний в загрузках по эвристике имени
        cand = []
        if paths["uploads"].exists():
            for fp in paths["uploads"].rglob("*"):
                if fp.is_file() and any(k in fp.name.lower() for k in ("замечан", "remark")):
                    cand.append(fp)
        remarks_path = cand[0] if cand else None
    if not remarks_path:
        raise FileNotFoundError("Не найден файл замечаний (ожидается имя со словом «замечания»).")
    remarks: list[Remark] = load_remarks(Path(remarks_path), cfg)
    if not remarks:
        raise ValueError("Из файла замечаний не удалось извлечь ни одного пункта.")

    # 2) ресурсы и батчевый retrieval только по разделам-источникам
    retr = HybridRetriever(cfg)
    src_codes = source_section_codes(object_type)
    queries = [r.text for r in remarks]
    if progress:
        progress(0, len(remarks), "Поиск источников по замечаниям…")
    hits_per = retr.batch_search(project, queries, sections=src_codes or None,
                                 top=int(cfg.get("retrieval.top_k", 8)))

    # 3) формируем задания для ИИ (с few-shot из памяти прошлых проектов)
    use_mem = bool(cfg.get("memory.enabled", True))
    mem_k = int(cfg.get("memory.k", 2))
    jobs, ctx_sources = [], []
    for r, hits in zip(remarks, hits_per):
        ctx, srcs = _format_context(hits, limit=int(cfg.get("retrieval.top_k", 8)))
        ctx_sources.append((srcs, hits))
        user_msg = _USER_TMPL.format(num=r.number, remark=r.text, context=ctx or "(не найдено)")
        if use_mem:
            try:
                from ..memory import fewshot_block
                fs = fewshot_block(r.text, k=mem_k, exclude_project=project)
            except Exception:  # noqa: BLE001
                fs = ""
            if fs:
                user_msg = fs + "\n\n" + user_msg
        jobs.append([
            {"role": "system", "content": _SYS},
            {"role": "user", "content": user_msg},
        ])

    if progress:
        progress(0, len(remarks), "Генерация ответов ИИ (параллельно)…")
    results = batch_chat(cfg, jobs, processor=_process, module="module4",
                         role="answer", json_mode=True)

    # 4) сборка ответов + consistency + cascade
    answers = []
    for idx, (r, (srcs, hits)) in enumerate(zip(remarks, ctx_sources)):
        res = results[idx]
        data = res.get("result") if res.get("ok") else {}
        data = data or {}
        used = data.get("used_sources") or []
        used_sources = [s for s in srcs if s["n"] in set(used)] or srcs[:3]

        answer_text = data.get("answer", "").strip()
        # источник для consistency = объединённый текст использованных фрагментов
        src_text = "\n".join(h.get("text", "") for h in hits[:5])
        cons = compare(src_text, answer_text + " " + data.get("correction", ""))

        # каскад: какие разделы затронет правка (по разделам источников)
        affected_codes = sorted({s["section"] for s in used_sources if s["section"]})
        cascade = downstream(project, affected_codes) if affected_codes else {"changed": [], "affected": []}

        answers.append({
            "number": r.number,
            "remark": r.text,
            "category": r.category,
            "answer": answer_text,
            "correction": data.get("correction", ""),
            "confidence": data.get("confidence", ""),
            "missing_data": data.get("missing_data", ""),
            "sources": used_sources,
            "consistency": cons,
            "cascade": cascade,
            "cascade_text": explain_cascade(project, affected_codes) if affected_codes else "",
            "status": "proposed",          # proposed|accepted|rejected|edited
            "user_answer": None,
            "error": res.get("error"),
        })
        if progress:
            progress(idx + 1, len(remarks), f"Замечание {r.number}")

    out = {
        "project": project, "object_type": object_type,
        "block": 1, "count": len(answers),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "answers": answers,
    }
    _save(project, out)
    return out


def _save(project: str, data: dict) -> Path:
    p = project_paths(project)["answers"]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_answers(project: str) -> dict[str, Any]:
    p = project_paths(project)["answers"]
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def set_decision(project: str, number: str, *, status: str,
                 user_answer: str | None = None) -> dict:
    """Пользователь принимает/правит/отклоняет конкретное предложение."""
    data = load_answers(project)
    for a in data.get("answers", []):
        if str(a["number"]) == str(number):
            a["status"] = status
            if user_answer is not None:
                a["user_answer"] = user_answer
            break
    _save(project, data)
    # лог решений (для будущего обучения)
    dec = project_paths(project)["decisions"]
    with dec.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.now().isoformat(), "number": number,
                            "status": status, "user_answer": user_answer},
                           ensure_ascii=False) + "\n")
    # пополняем память экспертизы принятыми/правлеными ответами (обучение на проектах)
    if status in ("accepted", "edited"):
        try:
            from ..memory import record_one
            ans_obj = next((a for a in data.get("answers", []) if str(a["number"]) == str(number)), None)
            if ans_obj:
                final = (ans_obj.get("user_answer") or ans_obj.get("answer") or "").strip()
                sec = (ans_obj.get("sources") or [{}])[0].get("section", "")
                record_one(remark=ans_obj.get("remark", ""), answer=final,
                           correction=ans_obj.get("correction", ""), section=sec,
                           project=project, number=number)
        except Exception:  # noqa: BLE001
            pass
    return data
