"""Расширение запроса (query expansion) под поиск ответов на замечания.

Замечание эксперта часто сформулировано канцелярским языком и не совпадает
лексически с текстом разделов ПД. Генерируем несколько перефразировок и
ключевых терминов, чтобы повысить полноту гибридного поиска.
"""
from __future__ import annotations

from .. import config as _cfg
from ..core.ai_providers import chat_json, LLMError

_SYS = (
    "Ты — инженер-эколог, эксперт по проектной документации (ПМООС/ООС) "
    "и государственной экспертизе по Постановлению Правительства РФ №87. "
    "Твоя задача — переформулировать замечание эксперта в несколько поисковых "
    "запросов к технической документации (разделы ТКР, ПОС, ИЭИ, ПМООС и др.)."
)

_PROMPT = (
    "Замечание эксперта:\n«{remark}»\n\n"
    "Сгенерируй {n} коротких поисковых запросов (на русском), которые помогут "
    "найти в проектной документации данные для ответа: используй синонимы, "
    "нормативную терминологию, названия величин, разделов и расчётов. "
    "Верни СТРОГО JSON-массив строк без пояснений."
)


def expand_query(remark_text: str, cfg: _cfg.Config, *, n: int = 3, module: str = "module4") -> list[str]:
    """Возвращает список запросов: исходный + перефразировки.

    При недоступности ИИ (нет ключа и т.п.) деградирует мягко: возвращает
    только исходный текст, не роняя пайплайн.
    """
    base = [remark_text.strip()]
    if n <= 0:
        return base
    try:
        msg = [
            {"role": "system", "content": _SYS},
            {"role": "user", "content": _PROMPT.format(remark=remark_text.strip(), n=n)},
        ]
        data = chat_json(cfg, msg, expect="array", module=module, role="expand")
        extra = [str(x).strip() for x in (data or []) if str(x).strip()]
        # дедуп с сохранением порядка
        seen = {base[0].lower()}
        for q in extra:
            if q.lower() not in seen:
                base.append(q)
                seen.add(q.lower())
        return base[: n + 1]
    except (LLMError, Exception):
        return base
