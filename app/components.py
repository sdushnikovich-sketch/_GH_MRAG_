"""Переиспользуемые компоненты Streamlit для ПМООС-RAG.

Здесь сосредоточены: панель настроек ИИ (с авто-сменой модели по провайдеру и
списком локальных моделей Ollama), карта разделов (таблица + версии), панель
индексации (прогресс/пауза/возобновление/фон) и контакты.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

PROVIDERS = ["deepseek", "openai", "gemini", "anthropic", "kimi", "mistral", "ollama"]
MODULES = [
    ("module1", "М1 · Систематизация ПД"),
    ("module3", "М3 · Граф связей"),
    ("module4", "М4 · Ответы на замечания"),
]
PROVIDER_LABEL = {
    "deepseek": "DeepSeek", "openai": "OpenAI (GPT)", "gemini": "Google Gemini",
    "anthropic": "Anthropic (Claude)", "kimi": "Kimi (Moonshot)",
    "mistral": "Mistral", "ollama": "Ollama (локально)",
}


# ─────────────────────────────── НАСТРОЙКИ ИИ ───────────────────────────────
def module_ai_selector(cfg, module: str, *, title: str = "🤖 ИИ для этого модуля") -> None:
    """Компактный выбор провайдера/модели/ключа ПРЯМО во вкладке модуля
    (замечание: «модель по модулям выбирается в соответствующих вкладках»)."""
    from pmoos.config import write_env_key
    from pmoos.core.ollama_utils import ollama_available, list_installed_models

    cur = cfg.resolve_provider(module)
    with st.expander(title, expanded=not (cfg.has_key(cur) or cur == "ollama")):
        opts = ["(по умолчанию)"] + PROVIDERS
        override = cfg.get(f"ai.modules.{module}.provider")
        idx = (PROVIDERS.index(override) + 1) if override in PROVIDERS else 0
        choice = st.selectbox(
            "Провайдер для модуля", opts, index=idx,
            format_func=lambda p: PROVIDER_LABEL.get(p, p) if p != "(по умолчанию)" else "(по умолчанию)",
            key=f"msel_{module}",
        )
        if choice == "(по умолчанию)":
            if override:
                cfg.set(f"ai.modules.{module}", {}); cfg.save(); st.rerun()
            provider = cfg.default_provider()
        else:
            if choice != override:
                cfg.set(f"ai.modules.{module}.provider", choice); cfg.save(); st.rerun()
            provider = choice

        st.caption(f"Активный провайдер: **{PROVIDER_LABEL.get(provider, provider)}** · "
                   f"модель ответа/проверки: `{cfg.model_for(provider, 'answer')}` · "
                   f"парсинг: `{cfg.model_for(provider, 'extract')}`")

        if provider == "ollama":
            if ollama_available():
                installed = list_installed_models()
                if installed:
                    cur_model = cfg.model_for("ollama", "answer")
                    mdl = st.selectbox("Локальная модель Ollama", installed,
                                       index=installed.index(cur_model) if cur_model in installed else 0,
                                       key=f"mol_{module}")
                    if mdl != cur_model:
                        for role in ("answer", "review", "extract", "expand"):
                            cfg.set(f"ai.providers.ollama.{role}", mdl)
                        cfg.save(); st.success(f"Модель Ollama: {mdl}"); st.rerun()
                else:
                    st.warning("Ollama запущена, но модели не найдены. Напр.: `ollama pull qwen2.5:7b-instruct`")
            else:
                st.warning("Ollama не обнаружена на :11434. Запустите `ollama serve` "
                           "(или задайте адрес в переменной OLLAMA_HOST).")
        else:
            if cfg.has_key(provider):
                st.caption("Ключ API задан ✓")
            else:
                val = st.text_input(f"Ключ API {PROVIDER_LABEL.get(provider, provider)} (ввести вручную)",
                                    value="", type="password", key=f"mkey_{module}")
                if val:
                    write_env_key(provider, val)
                    st.success("Ключ сохранён в .env"); st.rerun()


def ai_settings_panel(cfg) -> None:
    from pmoos.config import write_env_key
    from pmoos.core.ollama_utils import ollama_available, list_installed_models
    from pmoos.core.model_cache import model_status

    st.subheader("⚙️ Настройки ИИ")
    st.caption("Качество важнее скорости: для ответов/проверки используются "
               "сильные модели, для парсинга/перефраза — дешёвые.")

    default_prov = cfg.default_provider()
    new_default = st.selectbox(
        "Провайдер по умолчанию", PROVIDERS,
        index=PROVIDERS.index(default_prov) if default_prov in PROVIDERS else 0,
        format_func=lambda p: PROVIDER_LABEL.get(p, p),
    )
    if new_default != default_prov:
        cfg.set("ai.default_provider", new_default)
        cfg.save()
        st.rerun()

    # Ollama: показать реально установленные модели
    installed: list[str] = []
    if new_default == "ollama" or st.checkbox("Показать локальные модели Ollama", value=(new_default == "ollama")):
        if ollama_available():
            installed = list_installed_models()
            if installed:
                st.success("Найдены локальные модели Ollama: " + ", ".join(installed))
            else:
                st.info("Ollama запущена, но модели не найдены. Установите, напр.: `ollama pull qwen2.5:7b-instruct`")
        else:
            st.warning("Ollama не обнаружена на http://localhost:11434 — запустите `ollama serve`.")

    # Ключи API (пишутся в data_dir/.env через dotenv.set_key — корректное экранирование)
    with st.expander("🔑 Ключи API (хранятся в .env, не в config)"):
        for prov in PROVIDERS:
            if prov == "ollama":
                continue
            has = cfg.has_key(prov)
            cols = st.columns([3, 1])
            with cols[0]:
                val = st.text_input(
                    f"{PROVIDER_LABEL[prov]} API key",
                    value="", type="password",
                    placeholder=("ключ задан ✓" if has else "не задан"),
                    key=f"key_{prov}",
                )
            with cols[1]:
                st.write("✅" if has else "—")
            if val:
                write_env_key(prov, val)
                st.success(f"Ключ {PROVIDER_LABEL[prov]} сохранён в .env")
                st.rerun()

    # Авто-сменяемые модели по модулям (главное требование пользователя)
    st.markdown("**Модель по модулям (меняется автоматически с провайдером):**")
    rows = []
    for mod, label in MODULES:
        prov = cfg.resolve_provider(mod)
        rows.append({
            "Модуль": label,
            "Провайдер": PROVIDER_LABEL.get(prov, prov),
            "Ответ/Проверка": cfg.model_for(prov, "answer"),
            "Парсинг/Перефраз": cfg.model_for(prov, "extract"),
            "Ключ": "✓" if cfg.has_key(prov) else "✗",
        })
    st.dataframe(rows, width='stretch', hide_index=True)

    with st.expander("Переопределить провайдер для отдельного модуля"):
        for mod, label in MODULES:
            cur = cfg.resolve_provider(mod)
            choice = st.selectbox(
                label, ["(по умолчанию)"] + PROVIDERS,
                index=(PROVIDERS.index(cur) + 1) if cfg.get(f"ai.modules.{mod}.provider") in PROVIDERS else 0,
                format_func=lambda p: PROVIDER_LABEL.get(p, p) if p != "(по умолчанию)" else p,
                key=f"prov_{mod}",
            )
            if choice == "(по умолчанию)" and cfg.get(f"ai.modules.{mod}.provider"):
                cfg.set(f"ai.modules.{mod}", {})
                cfg.save(); st.rerun()
            elif choice in PROVIDERS and choice != cfg.get(f"ai.modules.{mod}.provider"):
                cfg.set(f"ai.modules.{mod}.provider", choice)
                cfg.save(); st.rerun()

        # для ollama — выбрать конкретную модель из установленных
        if installed:
            st.markdown("**Модель Ollama для ответов:**")
            cur_model = cfg.model_for("ollama", "answer")
            mdl = st.selectbox("Ollama answer-модель", installed,
                               index=installed.index(cur_model) if cur_model in installed else 0)
            if mdl != cur_model:
                for role in ("answer", "review", "extract", "expand"):
                    cfg.set(f"ai.providers.ollama.{role}", mdl)
                cfg.save(); st.success(f"Модель Ollama: {mdl}"); st.rerun()

    # Статус скачанных локальных моделей (эмбеддер/реранкер) — «проверять перед скачиванием»
    with st.expander("📦 Локальные модели (эмбеддер/реранкер) — статус кэша"):
        emb = cfg.get("embedding.model", "BAAI/bge-m3")
        rer = cfg.get("reranker.model", "BAAI/bge-reranker-v2-m3")
        for name in (emb, rer):
            stt = model_status(name)
            mark = "✅ скачана" if stt.get("cached") else "⬇️ будет скачана при первом запуске"
            st.write(f"`{name}` — {mark}")


# ─────────────────────────────── КАРТА РАЗДЕЛОВ ───────────────────────────────
def section_map(project: str, object_type: str) -> None:
    from pmoos.ingest.inventory import section_overview

    rows = section_overview(project, object_type)
    table = [{
        "": "✅" if r["present"] else "—",
        "№ ПП-87": r.get("num", ""),
        "Код": r["code"],
        "Раздел": r["name"] + (" (доп.)" if r.get("extra") else ""),
        "Файлов": r["n_files"],
    } for r in rows]
    st.dataframe(table, width='stretch', hide_index=True)

    present = sum(1 for r in rows if r["present"])
    st.caption(f"Присутствует разделов: {present} · отсутствует обязательных: "
               f"{sum(1 for r in rows if not r['present'] and not r.get('extra'))}")


def version_map(project: str, object_type: str) -> None:
    from pmoos.versioning.versions import analyze_versions, set_current_version

    vers = analyze_versions(project, object_type=object_type)
    groups = vers.get("groups", {})
    multi = {k: g for k, g in groups.items() if len(g.get("versions", [])) > 1}
    if not multi:
        st.info("Несколько версий одного раздела не обнаружено.")
        return
    st.caption("Для разделов с несколькими версиями отметьте актуальную:")
    from pmoos.ingest.sections import section_name
    for gkey, g in multi.items():
        st.markdown(f"**{section_name(g.get('section',''))}** — «{g.get('base','')}»")
        files = [v["file"] for v in g["versions"]]
        cur = g.get("current_file") or files[-1]
        labels = {v["file"]: f"{v.get('label','')} · {v.get('date') or 'без даты'} · {v['file']}"
                  for v in g["versions"]}
        chosen = st.radio("версия", files, index=files.index(cur) if cur in files else 0,
                          format_func=lambda f: labels.get(f, f), key=f"ver_{gkey}",
                          label_visibility="collapsed")
        if chosen != cur:
            set_current_version(project, gkey, chosen)
            st.success(f"Актуальная версия: {chosen}")


# ─────────────────────────────── ИНДЕКСАЦИЯ ───────────────────────────────
def indexing_panel(project: str, object_type: str) -> None:
    from pmoos.index.indexer import (
        progress_summary, start_background, request_pause, clear_pause, is_running,
    )

    s = progress_summary(project)
    running = s["running"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("▶ Индексировать", disabled=running, width='stretch'):
            start_background(project, object_type=object_type)
            st.rerun()
    with c2:
        if st.button("⏸ Пауза", disabled=not running, width='stretch'):
            request_pause(project)
            st.rerun()
    with c3:
        if st.button("⏯ Продолжить", disabled=running, width='stretch'):
            clear_pause(project)
            start_background(project, object_type=object_type)
            st.rerun()
    with c4:
        if st.button("🔄 Обновить", width='stretch'):
            st.rerun()

    st.progress(min(1.0, s["percent"] / 100.0),
                text=f"{s['percent']}% · файлов {s['files_done']}/{s['files_total']} · "
                     f"чанков {s['chunks_done']}")
    badge = {"running": "🟢 выполняется", "paused": "🟡 пауза", "done": "✅ завершено",
             "error": "🔴 ошибка", "idle": "⚪ не запускалось"}.get(s["status"], s["status"])
    st.write(f"Статус: {badge}" + (f" · {s.get('current_file','')}" if s.get("current_file") else ""))
    if s["status"] == "running":
        st.caption(f"Пульс процесса: {s.get('heartbeat_age', '—')} с назад "
                   f"(норма ≤ 10 с; статус обновляйте кнопкой «Обновить»)")
    if s.get("message"):
        if s["status"] == "error":
            st.error(s["message"])
        elif s["status"] == "done":
            st.success(s["message"])
        else:
            st.info(s["message"])
    d1, d2 = st.columns([1, 3])
    with d1:
        if st.button("🛑 Сбросить статус", width='stretch', key="idx_reset",
                     help="Если статус «завис» — сбросьте и запустите индексацию заново. "
                          "Прогресс по уже обработанным файлам сохраняется."):
            from pmoos.index.indexer import reset_state
            reset_state(project)
            st.rerun()
    with d2:
        st.caption("Индексация идёт в фоне и переживает закрытие вкладки; пауза/возобновление "
                   "переживают перезапуск. Первый запуск скачивает модель bge-m3 (~2.3 ГБ) — "
                   "ход загрузки виден в журнале ниже; при сбое там же будет точная причина.")

    from pmoos.index.indexer import log_tail, log_path
    with st.expander("📜 Журнал индексации (index_log.txt)"):
        tail = log_tail(project, 60)
        if tail:
            st.code(tail, language="text")
        else:
            st.caption("Журнал пока пуст — появится после запуска индексации.")
        st.caption(f"Файл журнала: {log_path(project)}")


# ─────────────────────────────── КОНТАКТЫ ───────────────────────────────
def contacts_panel(project: str) -> None:
    from pmoos.ingest.inventory import load_contacts, save_contacts

    data = load_contacts(project)
    st.markdown("**Проектировщики**")
    designers = data.get("designers", [])
    dn = st.text_input("ФИО / организация", key="dn_name")
    dr = st.text_input("Роль (ГИП, ГАП, инженер-эколог…)", key="dn_role")
    de = st.text_input("E-mail / телефон", key="dn_email")
    if st.button("➕ Добавить проектировщика") and dn:
        designers.append({"name": dn, "role": dr, "email": de})
        save_contacts(project, {"designers": designers, "experts": data.get("experts", [])})
        st.rerun()
    for d in designers:
        st.write(f"• {d.get('name','')} — {d.get('role','')} {d.get('email','')}")

    st.markdown("**Эксперты**")
    experts = data.get("experts", [])
    en = st.text_input("ФИО эксперта", key="ex_name")
    eo = st.text_input("Экспертная организация", key="ex_org")
    if st.button("➕ Добавить эксперта") and en:
        experts.append({"name": en, "org": eo})
        save_contacts(project, {"designers": designers, "experts": experts})
        st.rerun()
    for e in experts:
        st.write(f"• {e.get('name','')} — {e.get('org','')}")
