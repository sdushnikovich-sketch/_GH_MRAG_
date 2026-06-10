"""ПМООС-RAG v0.14.6 «Modular» — единый интерфейс (Streamlit).

Запуск:  streamlit run app/hub.py
Модули также запускаются по отдельности из папки modules/ (CLI).

Здесь учтены требования пользователя:
  #4  — убран «запрос смежникам» (его здесь нет);
  #5/#8 — переключатель площадной/линейный меняет состав ПД; раздел файла —
          это догадка с кандидатами, пользователь подтверждает (не навязываем);
  #7  — карта разделов таблицей + версии + хронология; #9 — файлы не сохраняем;
  #11/#12 — индексация в фоне с прогрессом/паузой/возобновлением;
  #13/#14 — список моделей Ollama и авто-смена модели по провайдеру.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pmoos import __version__
from pmoos.config import load_config
from pmoos.paths import project_paths
from pmoos.projects import list_projects, register_project
import app.components as C  # type: ignore

st.set_page_config(page_title="ПМООС-RAG", page_icon="🌍", layout="wide")


# ─────────────────────────────── состояние ───────────────────────────────
def _cfg():
    if "cfg" not in st.session_state:
        st.session_state.cfg = load_config()
    return st.session_state.cfg


def _save_uploads(project: str, files) -> int:
    """Сохранить загруженные файлы во ВРЕМЕННУЮ папку проекта (для индексации).

    Файлы не входят в постоянное хранилище — после индексации их можно удалить
    (кнопка «Очистить временные файлы»). В базе остаются только чанки/токены.
    """
    up = project_paths(project)["uploads"]
    up.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in files or []:
        try:
            (up / f.name).write_bytes(f.getbuffer())
            n += 1
        except Exception as e:  # noqa: BLE001
            st.error(f"Не удалось сохранить {f.name}: {e}")
    return n


# ─────────────────────────────── сайдбар ───────────────────────────────
def sidebar() -> tuple[str, str]:
    st.sidebar.title("🌍 ПМООС-RAG")
    st.sidebar.caption(f"v{__version__} «Modular»")

    projects = list_projects()
    mode = st.sidebar.radio("Проект", ["Выбрать", "Создать новый"],
                            horizontal=True, label_visibility="collapsed")
    if mode == "Создать новый" or not projects:
        name = st.sidebar.text_input("Название проекта", placeholder="ОПОЧКА-ДУБРОВКА 83-26С")
        if st.sidebar.button("Создать проект", disabled=not name):
            register_project(name)
            st.session_state.project = name
            st.rerun()
        project = st.session_state.get("project", name or "")
    else:
        idx = projects.index(st.session_state["project"]) if st.session_state.get("project") in projects else 0
        project = st.sidebar.selectbox("Проект", projects, index=idx)
        st.session_state.project = project

    # Тип объекта (влияет на состав разделов ПД по ПП-87)
    cfg = _cfg()
    cur_ot = st.session_state.get("object_type", cfg.get("object_type", "площадной"))
    object_type = st.sidebar.radio(
        "Тип объекта (ПП-87)", ["площадной", "линейный"],
        index=0 if cur_ot == "площадной" else 1,
        help="Меняет обязательный состав разделов ПД. У линейных объектов есть ТКР (Раздел 3).",
    )
    st.session_state.object_type = object_type

    st.sidebar.divider()
    with st.sidebar:
        C.ai_settings_panel(cfg)

    return project, object_type


# ─────────────────────────────── модули ───────────────────────────────
def tab_m1(project: str, object_type: str) -> None:
    st.header("МОДУЛЬ 1 · Загрузка и систематизация ПД (ПП-87)")
    C.module_ai_selector(_cfg(), "module1")
    st.caption("Файлы проекта НЕ сохраняются: в базе остаются только карта разделов "
               "и (после М2) векторные чанки. Временные файлы можно удалить после индексации.")

    files = st.file_uploader(
        "Загрузите файлы ПД (pdf / docx / xlsx). Можно перетащить много файлов.",
        type=["pdf", "docx", "xlsx", "xlsm", "txt", "md", "csv"],
        accept_multiple_files=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📥 Загрузить и систематизировать", disabled=not files, width='stretch'):
            n = _save_uploads(project, files)
            from pmoos.ingest.inventory import build_inventory
            build_inventory(project, object_type=object_type)
            st.success(f"Учтено файлов: {n}. Карта разделов обновлена.")
    with c2:
        if st.button("🔁 Пересобрать карту разделов", width='stretch'):
            from pmoos.ingest.inventory import build_inventory
            build_inventory(project, object_type=object_type)
            st.success("Карта разделов пересобрана.")
    with c3:
        if st.button("🧹 Очистить временные файлы", width='stretch',
                     help="Удалить загруженные файлы ПД (карта разделов и база сохранятся)"):
            import shutil
            up = project_paths(project)["uploads"]
            if up.exists():
                shutil.rmtree(up, ignore_errors=True)
            st.success("Временные файлы удалены. Карта разделов и RAG-база сохранены.")

    st.subheader("Карта разделов проектной документации")
    C.section_map(project, object_type)

    with st.expander("✏️ Подтвердить / исправить раздел и версию файла (догадку не навязываем)"):
        from pmoos.ingest.inventory import load_inventory, set_file_section, set_file_version
        from pmoos.ingest.sections import required_sections, SURVEYS, section_name
        inv = load_inventory(project)
        if inv and inv.get("files"):
            st.caption(f"Тип объекта проекта: **{inv.get('object_type', object_type)}** "
                       f"(меняется слева; влияет на состав разделов и определение версий)")
            codes = [s["code"] for s in required_sections(object_type)] + [s["code"] for s in SURVEYS] + ["UNKNOWN"]
            for it in inv["files"]:
                cands = ", ".join(f"{c['code']}({c['score']})" for c in it.get("candidates", [])) or "—"
                cols = st.columns([3, 2, 2, 2])
                cols[0].write(f"📄 {it['name']}")
                cols[1].caption(f"кандидаты: {cands}")
                new = cols[2].selectbox(
                    "раздел", codes, index=codes.index(it["section"]) if it["section"] in codes else len(codes) - 1,
                    format_func=lambda c: f"{c} · {section_name(c)}" if c != "UNKNOWN" else "не определён",
                    key=f"sec_{it['rel']}", label_visibility="collapsed",
                )
                if new != it["section"]:
                    set_file_section(project, it["rel"], new)
                    st.rerun()
                cur_ver = it.get("version_override") or it.get("version_hint") or ""
                ver = cols[3].text_input("версия", value=cur_ver, key=f"ver_{it['rel']}",
                                         placeholder="версия", label_visibility="collapsed")
                if ver and ver != cur_ver:
                    set_file_version(project, it["rel"], ver)
                    st.rerun()
        else:
            st.info("Сначала загрузите файлы.")

    st.subheader("Версии разделов")
    C.version_map(project, object_type)

    st.subheader("Контакты проектировщиков и экспертов")
    C.contacts_panel(project)


def tab_m2(project: str, object_type: str) -> None:
    st.header("МОДУЛЬ 2 · RAG-база (индексация)")
    st.caption("База Qdrant хранится отдельно от приложения — повторный запуск не "
               "переиндексирует уже загруженное (дедупликация по содержимому, "
               "стабильные ID чанков). Оптимизировано под GPU (RTX 3070 Ti).")
    C.indexing_panel(project, object_type)


def tab_m3(project: str, object_type: str) -> None:
    st.header("МОДУЛЬ 3 · Граф связей разделов и каскад изменений")
    C.module_ai_selector(_cfg(), "module3")
    from pmoos.graph.dependency import build_and_save, to_vis
    from pmoos.graph.cascade import downstream

    g = build_and_save(project)
    vis = to_vis(g)
    st.write(f"Узлов: {g.number_of_nodes()} · связей: {g.number_of_edges()}")

    # По умолчанию — таблица связей (быстро). Интерактивный граф рисуем по запросу
    # (тяжелее и каждый раз перерисовывается), плюс это убирает лишние предупреждения.
    show_graph = st.toggle("Показать интерактивный граф (pyvis)", value=False, key="m3_show_graph")
    if show_graph:
        from pmoos.graph.dependency import write_vis_html
        try:
            html_path = write_vis_html(project, vis)
            html = Path(html_path).read_text(encoding="utf-8")
            try:
                components.html(html, height=620, scrolling=True)
            except Exception:
                _download(html_path)  # на крайний случай — скачать HTML
        except Exception as e:  # noqa: BLE001
            st.caption(f"Не удалось построить интерактивный граф: {e}")
    st.dataframe([{"Источник": e["from"], "→": "→", "Потребитель": e["to"],
                   "По данным": e.get("title", "")} for e in vis["edges"]],
                 width='stretch', hide_index=True)

    st.subheader("Каскад изменений")
    nodes = sorted(n["id"] for n in vis["nodes"])
    changed = st.multiselect("Какие разделы изменились?", nodes,
                             help="Покажем, какие разделы/расчёты нужно перепроверить.")
    if changed:
        res = downstream(project, changed)
        if res["affected"]:
            st.dataframe([{"Затронуто": a["label"], "Глубина": a["depth"],
                           "Путь": " → ".join(a["via"])} for a in res["affected"]],
                         width='stretch', hide_index=True)
            st.info("Порядок пересчёта: " + " → ".join(res["order"]))
        else:
            st.success("Прямых зависимых разделов не обнаружено.")

    st.divider()
    st.subheader("🧠 Накопительный граф знаний по всем проектам")
    st.caption("Копит, какая техника/ЗВ встречалась в каких разделах и проектах "
               "(растёт при приёме ответов и по кнопке ниже). Хранится на диске, без сервера.")
    from pmoos.graph.knowledge import update_from_project, stats, to_vis as kg_vis
    cols = st.columns([1, 2])
    if cols[0].button("➕ Обновить граф знаний из этого проекта", width='stretch'):
        kn = update_from_project(project)
        st.success(f"Добавлено сущностей: {kn['entities']}. Узлов: {kn['nodes']}, связей: {kn['edges']}.")
    s = stats()
    cols[1].write(f"Сейчас в графе знаний: **{s['nodes']}** узлов, **{s['edges']}** связей "
                  f"({', '.join(f'{k}: {v}' for k, v in s['by_kind'].items()) or '—'})")
    q = st.text_input("Поиск: в каких проектах встречалась техника/ЗВ",
                      placeholder="напр. экскаватор или азота диоксид")
    if q:
        from pmoos.graph.knowledge import projects_with_entity
        projs = projects_with_entity(q)
        st.write("Найдено в проектах: " + (", ".join(projs) if projs else "—"))


def tab_m4(project: str, object_type: str) -> None:
    st.header("МОДУЛЬ 4 · Ответы на замечания ПМООС")
    cfg = _cfg()
    C.module_ai_selector(cfg, "module4")

    try:
        from pmoos.memory import kb_size
        n_kb = kb_size()
        if n_kb:
            st.caption(f"🧠 Память экспертизы: {n_kb} принятых ответов из прошлых проектов "
                       f"будут использованы как примеры (few-shot) при поиске ответов.")
    except Exception:  # noqa: BLE001
        pass

    rfile = st.file_uploader("Файл замечаний (docx/xlsx/pdf) — со словом «замечания» в имени удобнее",
                             type=["docx", "xlsx", "pdf", "txt"], key="remarks_up")
    remarks_path = None
    if rfile is not None:
        up = project_paths(project)["uploads"]
        up.mkdir(parents=True, exist_ok=True)
        remarks_path = up / rfile.name
        remarks_path.write_bytes(rfile.getbuffer())

    c1, c2, c3 = st.columns(3)
    run1 = c1.button("① Найти ответы", width='stretch')
    run2 = c2.button("② Проверить правки", width='stretch')
    run3 = c3.button("③ Финальная проверка", width='stretch')

    if run1:
        from pmoos.pipeline.block1_answers import run_block1
        with st.spinner("Поиск ответов (retrieval + ИИ)…"):
            try:
                out = run_block1(project, cfg, remarks_path=remarks_path, object_type=object_type)
                st.success(f"Готово: {out.get('count', 0)} ответов.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Ошибка блока 1: {e}")

    if run2:
        from pmoos.pipeline.block2_review import run_block2
        with st.spinner("Проверка расчётов/ссылок/нормативов…"):
            try:
                run_block2(project, cfg)
                st.success("Блок 2 завершён.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Ошибка блока 2: {e}")

    if run3:
        from pmoos.pipeline.block3_final import run_block3
        with st.spinner("Финальная проверка раздела…"):
            try:
                out = run_block3(project, cfg, object_type=object_type)
                st.success(f"Готовность к экспертизе: {out.get('ready', '?')}")
                if out.get("summary"):
                    st.write(out["summary"])
            except Exception as e:  # noqa: BLE001
                st.error(f"Ошибка блока 3: {e}")

    _render_answers(project)


def _render_answers(project: str) -> None:
    from pmoos.pipeline.block1_answers import load_answers, set_decision
    data = load_answers(project)
    answers = data.get("answers", [])
    if not answers:
        st.info("Ответы ещё не сформированы.")
        return

    st.subheader(f"Предложенные ответы ({len(answers)}) — приём/правка/отклонение")
    # пагинация, чтобы не тормозило на 75 замечаниях
    per = 10
    pages = (len(answers) + per - 1) // per
    page = st.number_input("Страница", 1, max(1, pages), 1) if pages > 1 else 1
    chunk = answers[(page - 1) * per: page * per]

    for a in chunk:
        num = a.get("number", "?")
        status = a.get("status", "proposed")
        icon = {"accepted": "✅", "edited": "✎", "rejected": "✗", "proposed": "·"}.get(status, "·")
        with st.expander(f"{icon} Замечание №{num} — {a.get('remark','')[:80]}"):
            st.markdown(f"**Замечание:** {a.get('remark','')}")
            cons = a.get("consistency", {})
            if not cons.get("ok", True):
                st.warning("⚠ Возможные расхождения: " + "; ".join(cons.get("issues", [])))
            txt = st.text_area("Ответ (можно отредактировать):",
                               value=a.get("user_answer") or a.get("answer", ""),
                               key=f"ans_{num}", height=120)
            if a.get("correction"):
                st.caption(f"Правка в ПМООС: {a['correction']}")
            srcs = a.get("sources", [])
            if srcs:
                st.markdown("**Источники:**")
                st.dataframe([{"Раздел": s.get("section", ""), "Файл": s.get("file", ""),
                               "Место": s.get("loc", ""), "Релевантность": s.get("score", "")}
                              for s in srcs], width='stretch', hide_index=True)
            if a.get("cascade_text"):
                st.caption("Каскад: " + a["cascade_text"])
            b1, b2, b3 = st.columns(3)
            if b1.button("✅ Принять", key=f"acc_{num}"):
                set_decision(project, num, status="accepted", user_answer=txt or None)
                st.rerun()
            if b2.button("✎ Сохранить правку", key=f"edt_{num}"):
                set_decision(project, num, status="edited", user_answer=txt)
                st.rerun()
            if b3.button("✗ Отклонить", key=f"rej_{num}"):
                set_decision(project, num, status="rejected")
                st.rerun()


def tab_m5(project: str, object_type: str) -> None:
    st.header("МОДУЛЬ 5 · Корректировка ПМООС и таблица ответов")
    st.caption("Формирует откорректированный раздел ПМООС (.docx) и таблицу ответов "
               "со столбцами «ОТВЕТ» и «ИСТОЧНИК» (.docx/.xlsx).")
    oos = st.file_uploader("Необязательно: исходный раздел ПМООС (для приложения-сверки)",
                           type=["docx", "pdf"], key="oos_up")
    oos_path = None
    if oos is not None:
        up = project_paths(project)["uploads"]
        up.mkdir(parents=True, exist_ok=True)
        oos_path = up / oos.name
        oos_path.write_bytes(oos.getbuffer())

    if st.button("📝 Сформировать документы", width='stretch'):
        from pmoos.output.docx_writer import build_corrected_oos_docx
        from pmoos.output.answers_table import build_answers_table_docx, build_answers_table_xlsx
        with st.spinner("Формирование .docx/.xlsx…"):
            p1 = build_corrected_oos_docx(project, original_oos_path=oos_path)
            p2 = build_answers_table_docx(project)
            p3 = build_answers_table_xlsx(project)
        st.success("Документы сформированы.")
        for p in (p1, p2, p3):
            _download(p)

    if st.button("🧪 Выгрузить обучающие тройки (anchor/positive/negative)",
                 help="Для будущего дообучения эмбеддера. Берёт принятые ответы."):
        from pmoos.output.training_export import export_triples
        r = export_triples(project)
        st.success(f"Сформировано троек: {r['count']}.")
        _download(r["path"])

    _list_outputs(project)


def tab_m6(project: str, object_type: str) -> None:
    st.header("МОДУЛЬ 6 · Выгрузка для УПРЗА «Эколог» / ИНТЕГРАЛ")
    st.caption("Источники выбросов + перечень ЗВ (коды) + задание на ввод. "
               "Геометрию источников и привязку значений заполняет инженер.")
    if st.button("📤 Сформировать выгрузку", width='stretch'):
        from pmoos.output.uprza_export import build_uprza_export, collect_emissions
        rows, extra = collect_emissions(project)
        paths = build_uprza_export(project)
        st.success(f"Готово. Распознано ЗВ: {len(rows)}.")
        if rows:
            st.dataframe([{"Код ЗВ": r["code"], "Наименование": r["name"]} for r in rows],
                         width='stretch', hide_index=True)
        for p in paths.values():
            _download(p)

    _list_outputs(project)


# ─────────────────────────────── утилиты вывода ───────────────────────────────
def _uid() -> int:
    """Монотонный счётчик за один рендер — гарантирует уникальные ключи виджетов."""
    n = st.session_state.get("_uid", 0)
    st.session_state["_uid"] = n + 1
    return n


def _download(path) -> None:
    path = Path(path)
    if not path.exists():
        return
    with path.open("rb") as f:
        st.download_button(f"⬇️ {path.name}", f.read(), file_name=path.name,
                           key=f"dl_{_uid()}_{path.name}")


def _list_outputs(project: str) -> None:
    out = project_paths(project)["out"]
    if out.exists():
        files = sorted(out.iterdir())
        if files:
            with st.expander(f"📁 Файлы проекта в out/ ({len(files)})"):
                for p in files:
                    _download(p)


# ─────────────────────────────── main ───────────────────────────────
def main() -> None:
    st.session_state["_uid"] = 0  # сброс счётчика ключей на каждый рендер
    # увеличенный шрифт (замечание «не видно»)
    st.markdown(
        """
        <style>
          html, body, [class*="css"], .stMarkdown, .stText, p, li, label,
          .stTabs [data-baseweb="tab"] { font-size: 17px !important; }
          .stDataFrame, .stTable { font-size: 16px !important; }
          h1 { font-size: 30px !important; }
          h2 { font-size: 24px !important; }
          h3 { font-size: 20px !important; }
          .stButton button, .stDownloadButton button { font-size: 16px !important; }
          section[data-testid="stSidebar"] * { font-size: 16px !important; }
          div[data-testid="stMetricValue"] { font-size: 22px !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    project, object_type = sidebar()
    if not project:
        st.info("Создайте или выберите проект слева, чтобы начать.")
        return

    st.title(f"Проект: {project}")
    tabs = st.tabs([
        "М1 · Систематизация", "М2 · Индексация", "М3 · Граф связей",
        "М4 · Ответы", "М5 · Корректировка", "М6 · УПРЗА",
    ])
    with tabs[0]:
        tab_m1(project, object_type)
    with tabs[1]:
        tab_m2(project, object_type)
    with tabs[2]:
        tab_m3(project, object_type)
    with tabs[3]:
        tab_m4(project, object_type)
    with tabs[4]:
        tab_m5(project, object_type)
    with tabs[5]:
        tab_m6(project, object_type)


if __name__ == "__main__":
    main()
