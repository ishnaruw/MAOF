from __future__ import annotations

import math
import sys
from html import escape
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.runtime_bootstrap import harden_scientific_runtime

harden_scientific_runtime()

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ui import live_demo_catalog as catalog
from src.ui import live_demo_loader as loader


SCORE_COL = "QoS_Adjusted_Composition_Score"
DEFENSE_CORE_TAKEAWAY = (
    "Takeaway: QoS-Pure-LLM improves over No-QoS by preserving functional suitability while improving "
    "Normalized QoS. QoS-Hybrid is the bonus refinement."
)
COMPACT_SCORE_COLUMNS = [
    "Mode",
    "QoS_Adjusted_Composition_Score",
    "Composition_Validity",
    "Composition_Completeness",
    "Functional_Coverage",
    "Normalized_QoS_Score",
    "Total_Response_Time_s",
    "Bottleneck_Throughput_kbps",
    "Average_Workflow_Availability",
]
MODE_COLORS = {
    "no_qos": "#6B7280",
    "qos_pure_llm": "#2563EB",
    "qos_topsis": "#7C3AED",
    "qos_hybrid": "#D97706",
}
RANK_COLUMNS = {
    "rag": "RAG retrieved_rank",
    "no_qos": "No-QoS mode_rank",
    "qos_pure_llm": "QoS-Pure-LLM mode_rank",
    "qos_topsis": "QoS-TOPSIS mode_rank",
    "qos_hybrid": "QoS-Hybrid mode_rank",
}
FORMULA_PROOF_MODES = ["no_qos", "qos_pure_llm", "qos_hybrid", "qos_topsis"]


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stMetric"] {
            border: 1px solid #E5E7EB;
            border-radius: 8px;
            padding: 0.65rem 0.75rem;
            background: #FFFFFF;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.55rem;
            line-height: 1.15;
        }
        .live-demo-status {
            border: 1px solid #D1D5DB;
            border-left: 6px solid #2563EB;
            border-radius: 8px;
            padding: 0.65rem 0.8rem;
            background: #F8FAFC;
            margin: 0.25rem 0 0.75rem 0;
            font-weight: 650;
        }
        .live-demo-status-green {
            border-color: #86EFAC;
            border-left-color: #16A34A;
            background: #FFFFFF;
            color: #064E3B;
        }
        .live-demo-bonus {
            border-color: #A7F3D0;
            border-left-color: #047857;
            background: #FFFFFF;
            color: #064E3B;
        }
        .live-demo-warning {
            border-color: #FCA5A5;
            border-left-color: #DC2626;
            background: #FEF2F2;
            color: #7F1D1D;
        }
        .live-demo-context-card {
            border: 1px solid #E5E7EB;
            border-radius: 8px;
            padding: 0.65rem 0.75rem;
            background: #FFFFFF;
            min-height: 82px;
        }
        .live-demo-context-label {
            color: #4B5563;
            font-size: 0.86rem;
            line-height: 1.2;
            margin-bottom: 0.25rem;
        }
        .live-demo-context-value {
            color: #111827;
            font-size: 1.05rem;
            line-height: 1.3;
            font-weight: 650;
            overflow-wrap: anywhere;
            white-space: normal;
        }
        .live-demo-goal-label {
            color: #4B5563;
            font-size: 0.95rem;
            line-height: 1.2;
            font-weight: 700;
            margin: 0.2rem 0 0.25rem 0;
        }
        .live-demo-goal {
            color: #111827;
            font-size: 1.35rem;
            line-height: 1.35;
            font-weight: 800;
            overflow-wrap: anywhere;
            margin: 0 0 0.9rem 0;
        }
        .live-demo-card-grid {
            display: grid;
            gap: 0.55rem;
            margin: 0.35rem 0 0.7rem 0;
        }
        .live-demo-step {
            display: grid;
            grid-template-columns: 4.5rem minmax(0, 1fr) 9rem;
            gap: 0.8rem;
            align-items: start;
            border: 1px solid #E5E7EB;
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            background: #FFFFFF;
        }
        .live-demo-step-num {
            font-weight: 800;
            color: #2563EB;
        }
        .live-demo-step-text {
            overflow-wrap: anywhere;
            color: #111827;
        }
        .live-demo-step-count {
            color: #4B5563;
            font-size: 0.88rem;
            text-align: right;
        }
        .live-demo-path-line {
            overflow-wrap: anywhere;
            border: 1px solid #E5E7EB;
            border-radius: 8px;
            padding: 0.65rem 0.8rem;
            background: #FFFFFF;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
            font-size: 0.88rem;
        }
        .live-demo-small-note {
            color: #4B5563;
            font-size: 0.9rem;
        }
        .live-demo-delta-card {
            border: 1px solid #E5E7EB;
            border-radius: 8px;
            padding: 0.65rem 0.75rem;
            background: #FFFFFF;
            min-height: 78px;
        }
        .live-demo-delta-card-positive {
            border-color: #86EFAC;
            background: #DCFCE7;
        }
        .live-demo-delta-label {
            color: #4B5563;
            font-size: 0.86rem;
            line-height: 1.2;
            margin-bottom: 0.25rem;
        }
        .live-demo-delta-value {
            color: #111827;
            font-size: 1.55rem;
            line-height: 1.15;
            font-weight: 700;
        }
        .live-demo-delta-card-positive .live-demo-delta-value {
            color: #064E3B;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    return None if math.isnan(parsed) else parsed


def _fmt(value: Any, *, exact: bool, decimals: int = 3) -> str:
    parsed = _to_float(value)
    if parsed is None:
        text = str(value or "").strip()
        return text if text else "n/a"
    if exact:
        return f"{parsed:.9f}".rstrip("0").rstrip(".")
    return f"{parsed:.{decimals}f}"


def _fmt_delta(value: float | None, *, exact: bool) -> str:
    if value is None:
        return "n/a"
    sign = "+" if value > 0 else ""
    return f"{sign}{_fmt(value, exact=exact)}"


def _delta_card(label: str, value: float | None, *, exact: bool) -> str:
    positive_class = " live-demo-delta-card-positive" if value is not None and value > 0 else ""
    return (
        f"<div class='live-demo-delta-card{positive_class}'>"
        f"<div class='live-demo-delta-label'>{escape(label)}</div>"
        f"<div class='live-demo-delta-value'>{escape(_fmt_delta(value, exact=exact))}</div>"
        "</div>"
    )


def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = [
        "Composition_Validity",
        "Composition_Completeness",
        "Functional_Coverage",
        "Normalized_QoS_Score",
        SCORE_COL,
        "Total_Response_Time_s",
        "Bottleneck_Throughput_kbps",
        "Average_Workflow_Availability",
        "qos_rt_s",
        "qos_tp_kbps",
        "qos_availability",
        "functional_match",
        "retrieved_rank",
        "mode_rank",
    ]
    for col in numeric_cols:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _display_frame(df: pd.DataFrame, *, exact: bool, decimals: int = 3) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        numeric = pd.to_numeric(out[col], errors="coerce")
        if numeric.notna().any() and numeric.notna().sum() == out[col].notna().sum():
            out[col] = numeric if exact else numeric.round(decimals)
    return out.astype(object).where(pd.notna(out), "n/a")


def _pinned_columns(*columns: str) -> dict[str, Any]:
    return {column: st.column_config.Column(pinned=True) for column in columns}


def _mode_df(eval_rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = _numeric_frame(pd.DataFrame(eval_rows))
    if df.empty or "Mode" not in df:
        return pd.DataFrame()
    df["Mode"] = df["Mode"].astype(str)
    df["Mode_Label"] = df["Mode"].map(loader.MODE_LABELS).fillna(df["Mode"])
    df["_mode_order"] = df["Mode"].map(lambda mode: loader.mode_sort_key(mode)[0])
    return df.sort_values(["_mode_order", "Mode"]).drop(columns=["_mode_order"])


def _row_for_mode(eval_df: pd.DataFrame, mode: str) -> dict[str, Any]:
    if eval_df.empty or "Mode" not in eval_df:
        return {}
    rows = eval_df[eval_df["Mode"].astype(str) == mode]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def _score(row: dict[str, Any]) -> float | None:
    return _to_float(row.get(SCORE_COL))


def _metric_delta(left: dict[str, Any], right: dict[str, Any], metric: str) -> float | None:
    left_value = _to_float(left.get(metric))
    right_value = _to_float(right.get(metric))
    if left_value is None or right_value is None:
        return None
    return left_value - right_value


def _query_label(row: dict[str, Any]) -> str:
    title = str(row.get("title") or "").strip()
    return f"{row['query_id']} - {title}" if title else str(row["query_id"])


def _visible_modes(modes: list[str], selected: list[str]) -> list[str]:
    selected_set = {str(mode) for mode in selected}
    return [mode for mode in modes if mode in selected_set]


def _subtask_sort_key(value: Any) -> tuple[int, str]:
    text = str(value or "")
    return (int(text), text) if text.isdigit() else (999, text)


def _mode_sort_df(df: pd.DataFrame, mode_col: str = "mode") -> pd.DataFrame:
    if df.empty or mode_col not in df.columns:
        return df
    out = df.copy()
    out["_mode_order"] = out[mode_col].astype(str).map(lambda mode: loader.mode_sort_key(mode)[0])
    return out.sort_values(["_mode_order", mode_col]).drop(columns=["_mode_order"])


def _context_card(label: str, lines: list[str]) -> str:
    value = "<br>".join(escape(str(line)) for line in lines if str(line).strip()) or "n/a"
    return (
        "<div class='live-demo-context-card'>"
        f"<div class='live-demo-context-label'>{escape(label)}</div>"
        f"<div class='live-demo-context-value'>{value}</div>"
        "</div>"
    )


def _render_sidebar(
    directory_selector: Callable[..., str] | None = None,
    default_run_dir: Path | None = None,
) -> dict[str, Any] | None:
    discovered = loader.discover_run_folders(str(loader.DEFAULT_RESULTS_ROOT))
    default_run = default_run_dir or loader.PROJECT_ROOT / "results/logs/RUNS_MAY_31_NEW_5/fireworks_gpt-oss-120b"
    if not default_run.exists() and discovered:
        default_run = Path(discovered[0]["run_dir"])

    with st.sidebar:
        st.header("Live Demo Controls")
        if directory_selector is not None:
            run_dir_text = directory_selector(
                "Run folder",
                default_path=default_run,
                key="live_demo_dir",
                root=loader.DEFAULT_RESULTS_ROOT,
            )
        else:
            if "live_demo_run_path" not in st.session_state:
                st.session_state["live_demo_run_path"] = str(default_run)
            run_dir_text = st.text_input("Run folder", key="live_demo_run_path")

        if st.button("Reload live demo artifacts"):
            for func in [
                loader.read_json_file,
                loader.discover_run_folders,
                loader.scan_run_folder,
                loader.load_live_demo_query,
                loader.load_official_query_aggregate,
                catalog.load_api_catalog,
            ]:
                if hasattr(func, "clear"):
                    func.clear()
            st.rerun()

        defense_core_mode = st.toggle(
            "Defense Core Mode",
            value=False,
            help="Condense optional diagnostics into collapsed sections for a 6-7 minute defense walkthrough.",
        )

        scan = loader.scan_run_folder(run_dir_text)
        queries = list(scan.get("queries") or [])
        for warning in scan.get("warnings") or []:
            st.warning(warning)
        if not queries:
            st.warning("Select a run folder that contains qXX timestamped query folders.")
            return {
                "view_mode": "Selected query deep dive",
                "run_dir_text": run_dir_text,
                "query_id": "",
                "selected_modes": [],
                "focused_subtask_id": "",
                "top_k": 5,
                "show_all": True,
                "show_exact": False,
                "show_raw": False,
                "defense_core_mode": defense_core_mode,
                "scan": scan,
            }

        view_mode = st.radio(
            "View",
            ["Selected query deep dive", "All queries average results"],
            index=0,
        )

        if view_mode == "All queries average results":
            show_exact = st.toggle("Show exact values", value=False)
            return {
                "view_mode": view_mode,
                "run_dir_text": run_dir_text,
                "query_id": "",
                "selected_modes": [],
                "focused_subtask_id": "",
                "top_k": 5,
                "show_all": False,
                "show_exact": show_exact,
                "show_raw": False,
                "defense_core_mode": defense_core_mode,
                "scan": scan,
            }

        query_ids = [str(row["query_id"]) for row in queries]
        default_query = query_ids[0]
        if st.session_state.get("live_demo_query_id") not in query_ids:
            st.session_state["live_demo_query_id"] = default_query
        selected_query = st.selectbox(
            "Query selector",
            query_ids,
            format_func=lambda query_id: _query_label(next(row for row in queries if row["query_id"] == query_id)),
            key="live_demo_query_id",
        )

        query_row = next(row for row in queries if row["query_id"] == selected_query)
        available_modes = sorted(list(query_row.get("available_modes") or loader.MODE_ORDER), key=loader.mode_sort_key)
        selected_modes = st.multiselect(
            "Mode visibility",
            available_modes,
            default=available_modes,
            format_func=loader.mode_label,
        )

        subtask_count = int(_to_float(query_row.get("num_subtasks")) or 1)
        subtask_options = [str(index) for index in range(1, max(subtask_count, 1) + 1)]
        if st.session_state.get("live_demo_subtask_id") not in subtask_options:
            st.session_state["live_demo_subtask_id"] = subtask_options[0]
        focused_subtask_id = st.selectbox(
            "Subtask selector",
            subtask_options,
            format_func=lambda subtask_id: f"Subtask {subtask_id}",
            key="live_demo_subtask_id",
        )

        top_k = int(st.number_input("Top-K selector", min_value=1, max_value=40, value=5, step=1))
        show_all = st.toggle("Show all 40 candidates", value=True)
        show_exact = st.toggle("Show exact values", value=False)
        show_raw = st.toggle("Show raw artifacts", value=False)

    return {
        "view_mode": view_mode,
        "run_dir_text": run_dir_text,
        "query_id": selected_query,
        "selected_modes": selected_modes,
        "focused_subtask_id": focused_subtask_id,
        "top_k": top_k,
        "show_all": show_all,
        "show_exact": show_exact,
        "show_raw": show_raw,
        "defense_core_mode": defense_core_mode,
        "scan": scan,
    }


def _selected_modes_for_api(paths: dict[str, list[dict[str, Any]]], subtask_id: str, api_id: str) -> list[str]:
    modes = []
    for mode in loader.MODE_ORDER:
        for row in paths.get(mode, []):
            if str(row.get("subtask_id") or "") == str(subtask_id) and str(row.get("api_id") or "") == str(api_id):
                modes.append(loader.mode_label(mode))
                break
    return modes


def _subtask_descriptions(bundle: dict[str, Any]) -> dict[str, str]:
    return {str(row.get("subtask_id") or ""): str(row.get("description") or "") for row in bundle.get("subtasks") or []}


def _render_query_context(bundle: dict[str, Any]) -> None:
    st.header("1. Query Context")
    st.subheader(f"{bundle['query_id']} - {bundle.get('query_title') or bundle['query_id']}")
    if bundle.get("user_goal"):
        st.markdown(
            "<div class='live-demo-goal-label'>Actual user goal</div>"
            f"<div class='live-demo-goal'>{escape(str(bundle['user_goal']))}</div>",
            unsafe_allow_html=True,
        )

    category = str(bundle.get("query_category") or "").strip()
    domain = str(bundle.get("query_domain") or "").strip()
    category_lines = [part for part in [category, domain] if part] or ["Category not recorded in artifact"]
    subtask_count = bundle.get("num_subtasks") or len(bundle.get("subtasks") or [])
    modes = [loader.mode_label(mode) for mode in bundle.get("available_modes") or []]
    cols = st.columns([1, 1.2, 1, 1.6])
    cols[0].metric("Query ID", str(bundle.get("query_id") or "n/a"))
    cols[1].markdown(_context_card("Category/domain", category_lines), unsafe_allow_html=True)
    cols[2].metric("Subtasks", str(subtask_count or "n/a"))
    cols[3].markdown(_context_card("Available modes", modes), unsafe_allow_html=True)
    st.markdown(
        "<div class='live-demo-status'>This query requires a sequential multi-API workflow, so API selection affects the final composition quality.</div>",
        unsafe_allow_html=True,
    )


def _retrieval_counts(retrieval_df: pd.DataFrame) -> dict[str, int]:
    if retrieval_df.empty or "subtask_id" not in retrieval_df:
        return {}
    return retrieval_df.groupby("subtask_id", dropna=False).size().astype(int).to_dict()


def _render_subtasks(bundle: dict[str, Any], retrieval_df: pd.DataFrame, selected_modes: list[str], *, exact: bool) -> None:
    st.header("2. Decomposed Subtasks")
    subtasks = list(bundle.get("subtasks") or [])
    if not subtasks:
        st.warning("No decomposed subtasks were loaded from 0_decomposer.json.")
        return

    counts = _retrieval_counts(retrieval_df)
    cards = []
    for row in subtasks:
        subtask_id = str(row.get("subtask_id") or "")
        description = str(row.get("description") or "")
        candidate_count = counts.get(subtask_id, 0)
        cards.append(
            "<div class='live-demo-step'>"
            f"<div class='live-demo-step-num'>Step {escape(subtask_id)}</div>"
            f"<div class='live-demo-step-text'>{escape(description)}</div>"
            f"<div class='live-demo-step-count'>{escape(str(candidate_count))} retrieved</div>"
            "</div>"
        )
    st.markdown(f"<div class='live-demo-card-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)

    rows = []
    paths = bundle.get("selected_paths") or {}
    for subtask in subtasks:
        subtask_id = str(subtask.get("subtask_id") or "")
        row: dict[str, Any] = {"subtask_id": subtask_id}
        for mode in selected_modes:
            step = next((item for item in paths.get(mode, []) if str(item.get("subtask_id") or "") == subtask_id), {})
            row[loader.mode_label(mode)] = step.get("api_id") or ""
        rows.append(row)
    with st.expander("Selected API per mode", expanded=False):
        st.dataframe(_display_frame(pd.DataFrame(rows), exact=exact), use_container_width=True, hide_index=True, height=170)


def _render_retrieval_snapshot(
    retrieval_df: pd.DataFrame,
    top_k: int,
    *,
    show_all: bool,
    exact: bool,
    defense_core_mode: bool = False,
) -> None:
    st.header("3. RAG Retrieval Snapshot")
    if defense_core_mode:
        st.write("Each subtask retrieves 40 candidates using RAG over API catalog metadata.")
    else:
        st.write(
            "The RAG retriever ranks APIs using semantic similarity to the subtask over API catalog metadata. "
            "This is the shared candidate pool used by all modes before mode-specific re-ranking."
        )
    if retrieval_df.empty:
        st.warning("Retrieval functional-match rows are unavailable; continuing with the remaining artifacts.")
        return

    if defense_core_mode:
        with st.expander("Optional: Show RAG retrieved candidates.", expanded=False):
            _render_retrieval_tables(retrieval_df, top_k, show_all=True, exact=exact)
        return

    _render_retrieval_tables(retrieval_df, top_k, show_all=show_all, exact=exact)


def _render_retrieval_tables(
    retrieval_df: pd.DataFrame,
    top_k: int,
    *,
    show_all: bool,
    exact: bool,
) -> None:
    view_limit = 40 if show_all else top_k
    display_cols = [
        "retrieved_rank",
        "api_id",
        "display_name",
        "category",
        "functional_match",
        "qos_rt_s",
        "qos_tp_kbps",
        "qos_availability",
    ]
    grouped = list(retrieval_df.sort_values(["subtask_id", "retrieved_rank"]).groupby("subtask_id", sort=True))
    tabs = st.tabs([f"Subtask {subtask_id}" for subtask_id, _ in grouped])
    for tab, (subtask_id, group) in zip(tabs, grouped):
        with tab:
            st.caption(f"Showing {'all 40' if show_all else f'top-{top_k}'} retrieved candidates for subtask {subtask_id}.")
            table = group.sort_values("retrieved_rank").head(view_limit).reset_index(drop=True)
            view = table[[col for col in display_cols if col in table.columns]].copy()
            st.dataframe(
                _display_frame(view, exact=exact),
                use_container_width=True,
                hide_index=True,
                height=360 if show_all else 260,
                column_config=_pinned_columns("retrieved_rank", "api_id"),
            )


def _fallback_value(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return ""


def _info_value(value: Any, *, exact: bool) -> str:
    parsed = _to_float(value)
    if parsed is not None:
        return _fmt(parsed, exact=exact)
    text = str(value or "").strip()
    return text if text else "n/a"


def _candidate_row(retrieval_df: pd.DataFrame, ranking_df: pd.DataFrame, subtask_id: str, api_id: str) -> dict[str, Any]:
    retrieval_rows = retrieval_df[
        (retrieval_df.get("subtask_id", pd.Series(dtype=str)).astype(str) == str(subtask_id))
        & (retrieval_df.get("api_id", pd.Series(dtype=str)).astype(str) == str(api_id))
    ]
    ranking_rows = ranking_df[
        (ranking_df.get("subtask_id", pd.Series(dtype=str)).astype(str) == str(subtask_id))
        & (ranking_df.get("api_id", pd.Series(dtype=str)).astype(str) == str(api_id))
    ]
    row: dict[str, Any] = {}
    if not retrieval_rows.empty:
        row.update(retrieval_rows.iloc[0].to_dict())
    if not ranking_rows.empty:
        for key, value in ranking_rows.iloc[0].to_dict().items():
            row.setdefault(key, value)
    return row


def _mode_rank_rows(ranking_df: pd.DataFrame, subtask_id: str, api_id: str) -> pd.DataFrame:
    rows = []
    if ranking_df.empty:
        return pd.DataFrame()
    subset = ranking_df[
        (ranking_df["subtask_id"].astype(str) == str(subtask_id)) & (ranking_df["api_id"].astype(str) == str(api_id))
    ]
    for mode in loader.MODE_ORDER:
        mode_rows = subset[subset["mode"].astype(str) == mode]
        if mode_rows.empty:
            rows.append({"Mode": loader.mode_label(mode), "mode_rank": None, "selected_by_planner": "No"})
            continue
        item = mode_rows.iloc[0]
        rows.append(
            {
                "Mode": loader.mode_label(mode),
                "mode_rank": item.get("mode_rank"),
                "selected_by_planner": item.get("selected_for_planner") or "No",
            }
        )
    return pd.DataFrame(rows)


def _render_params(title: str, value: Any, *, use_expander: bool = True) -> None:
    params = value if isinstance(value, list) else []
    if not use_expander:
        st.markdown(f"**{title}**")
        if params:
            st.dataframe(pd.DataFrame(params), use_container_width=True, hide_index=True, height=180)
        else:
            st.caption("Not available in the loaded catalog metadata.")
        return

    with st.expander(title, expanded=False):
        if params:
            st.dataframe(pd.DataFrame(params), use_container_width=True, hide_index=True, height=180)
        else:
            st.caption("Not available in the loaded catalog metadata.")


def _candidate_option_label(api_id: str, subtask_rows: pd.DataFrame) -> str:
    rows = subtask_rows[subtask_rows["api_id"].astype(str) == str(api_id)]
    if rows.empty:
        return loader.short_api_name(api_id, max_parts=5)
    rank = _to_float(rows.iloc[0].get("retrieved_rank"))
    rank_text = f"RAG #{int(rank)}" if rank is not None else "RAG rank n/a"
    return f"{rank_text}: {loader.short_api_name(api_id, max_parts=5)}"


def _render_candidate_inspector(
    bundle: dict[str, Any],
    retrieval_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    focused_subtask_id: str,
    *,
    exact: bool,
    defense_core_mode: bool = False,
) -> str:
    if defense_core_mode:
        with st.expander("Optional: Inspect API catalog metadata.", expanded=False):
            inspected_subtask_id = _render_candidate_inspector_body(
                bundle,
                retrieval_df,
                ranking_df,
                focused_subtask_id,
                exact=exact,
                use_parameter_expanders=False,
            )
        return inspected_subtask_id

    return _render_candidate_inspector_body(
        bundle,
        retrieval_df,
        ranking_df,
        focused_subtask_id,
        exact=exact,
        use_parameter_expanders=True,
    )


def _render_candidate_inspector_body(
    bundle: dict[str, Any],
    retrieval_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    focused_subtask_id: str,
    *,
    exact: bool,
    use_parameter_expanders: bool,
) -> str:
    st.header("4. Candidate API Catalog Inspector")
    subtask_ids = sorted(retrieval_df["subtask_id"].astype(str).unique().tolist(), key=_subtask_sort_key) if not retrieval_df.empty else []
    if not subtask_ids:
        st.warning("No RAG candidates are available for candidate inspection.")
        return focused_subtask_id
    default_index = subtask_ids.index(str(focused_subtask_id)) if str(focused_subtask_id) in subtask_ids else 0
    inspected_subtask_id = st.selectbox(
        "Inspector subtask",
        subtask_ids,
        index=default_index,
        format_func=lambda subtask_id: f"Subtask {subtask_id}",
        key=f"candidate_inspector_subtask_{bundle['query_id']}",
    )

    descriptions = _subtask_descriptions(bundle)
    if descriptions.get(str(inspected_subtask_id)):
        st.caption(descriptions[str(inspected_subtask_id)])

    subtask_rows = retrieval_df[retrieval_df["subtask_id"].astype(str) == str(inspected_subtask_id)].copy() if not retrieval_df.empty else pd.DataFrame()
    if subtask_rows.empty:
        st.warning("No RAG candidates are available for the selected subtask.")
        return str(inspected_subtask_id)

    subtask_rows = subtask_rows.sort_values("retrieved_rank")
    api_options = subtask_rows["api_id"].astype(str).tolist()
    stored = str(st.session_state.get("live_demo_inspect_api_id") or "")
    default_index = api_options.index(stored) if stored in api_options else 0
    selected_api = st.selectbox(
        "Inspect candidate API",
        api_options,
        index=default_index,
        format_func=lambda api_id: _candidate_option_label(api_id, subtask_rows),
        key=f"candidate_inspector_{bundle['query_id']}_{inspected_subtask_id}",
    )

    catalog_bundle = catalog.load_api_catalog()
    metadata = catalog.find_api_metadata(catalog_bundle, selected_api)
    artifact_row = _candidate_row(retrieval_df, ranking_df, inspected_subtask_id, selected_api)
    if metadata is None:
        st.warning("Catalog metadata not found for this API ID")
        metadata = {}

    mode_ranks = _mode_rank_rows(ranking_df, inspected_subtask_id, selected_api)
    left, right = st.columns([1.1, 1])
    with left:
        info_rows = [
            ("API ID", selected_api),
            ("Tool name", _fallback_value(metadata.get("tool_name"), artifact_row.get("tool_name"))),
            ("API name or endpoint name", _fallback_value(metadata.get("api_name"), metadata.get("endpoint_name"), artifact_row.get("api_name"))),
            ("Category", _fallback_value(metadata.get("category"), artifact_row.get("category"))),
            ("HTTP method", _fallback_value(metadata.get("http_method"), artifact_row.get("http_method"))),
            ("Endpoint path", _fallback_value(metadata.get("endpoint_path"), artifact_row.get("endpoint_path"), artifact_row.get("endpoint_url"))),
            ("Short description", _fallback_value(metadata.get("short_description"), artifact_row.get("comments"))),
            ("Full description", metadata.get("full_description") or ""),
            ("Functional Match Label for selected subtask", artifact_row.get("functional_match")),
            ("Retrieved rank", artifact_row.get("retrieved_rank")),
            ("QoS response time", artifact_row.get("qos_rt_s")),
            ("QoS throughput", artifact_row.get("qos_tp_kbps")),
            ("QoS availability", artifact_row.get("qos_availability")),
        ]
        info_df = pd.DataFrame(info_rows, columns=["Field", "Value"])
        info_df["Value"] = info_df["Value"].map(lambda value: _info_value(value, exact=exact))
        st.dataframe(
            info_df,
            use_container_width=True,
            hide_index=True,
            height=420,
        )
    with right:
        st.caption("Mode ranks and planner-selection status for the selected query/subtask.")
        st.dataframe(_display_frame(mode_ranks, exact=exact), use_container_width=True, hide_index=True, height=220)

    _render_params("Required parameters", metadata.get("required_parameters"), use_expander=use_parameter_expanders)
    _render_params("Optional parameters", metadata.get("optional_parameters"), use_expander=use_parameter_expanders)
    st.caption(
        "Catalog detail note: empty parameter sections mean the loaded project catalog does not record those fields for this API. "
        "This is a source-data limitation, not a candidate-to-catalog mapping error."
    )
    return str(inspected_subtask_id)


def _not_available(value: Any, *, exact: bool = False) -> str:
    text = _fmt(value, exact=exact)
    return "Not available in artifact" if text == "n/a" else text


SECTION_5_IRRELEVANT_COLUMNS = ["API", "RAG rank", "Functional match", "RT", "Throughput", "Availability"]
SECTION_5_DOMINANCE_COLUMNS = ["API", "RAG rank", "RT", "Throughput", "Availability", "Dominates top-5 candidates"]
SECTION_5_DOMINANCE_DETAIL_COLUMNS = [
    "Lower-ranked API",
    "Lower-ranked RAG rank",
    "Compared top-5 API",
    "Compared RAG rank",
    "RT comparison",
    "Throughput comparison",
    "Availability comparison",
]


def _has_complete_qos(row: pd.Series) -> bool:
    return all(
        _to_float(row.get(column)) is not None
        for column in ["qos_rt_s", "qos_tp_kbps", "qos_availability"]
    )


def _strict_qos_dominates(lower: pd.Series, compared: pd.Series) -> bool:
    lower_rt = _to_float(lower.get("qos_rt_s"))
    lower_tp = _to_float(lower.get("qos_tp_kbps"))
    lower_av = _to_float(lower.get("qos_availability"))
    compared_rt = _to_float(compared.get("qos_rt_s"))
    compared_tp = _to_float(compared.get("qos_tp_kbps"))
    compared_av = _to_float(compared.get("qos_availability"))
    return (
        lower_rt is not None
        and compared_rt is not None
        and lower_tp is not None
        and compared_tp is not None
        and lower_av is not None
        and compared_av is not None
        and lower_rt < compared_rt
        and lower_tp > compared_tp
        and lower_av > compared_av
    )


def _comparison_text(lower_value: Any, compared_value: Any, operator: str) -> str:
    return f"{_fmt(lower_value, exact=False)} {operator} {_fmt(compared_value, exact=False)}"


def _section_5_dominance_detail_row(lower: pd.Series, compared: pd.Series) -> dict[str, Any]:
    return {
        "Lower-ranked API": lower.get("api_id"),
        "Lower-ranked RAG rank": lower.get("retrieved_rank"),
        "Compared top-5 API": compared.get("api_id"),
        "Compared RAG rank": compared.get("retrieved_rank"),
        "RT comparison": _comparison_text(lower.get("qos_rt_s"), compared.get("qos_rt_s"), "<"),
        "Throughput comparison": _comparison_text(lower.get("qos_tp_kbps"), compared.get("qos_tp_kbps"), ">"),
        "Availability comparison": _comparison_text(lower.get("qos_availability"), compared.get("qos_availability"), ">"),
    }


def _numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(float("nan"), index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def _section_5_ranked_candidates(group: pd.DataFrame) -> pd.DataFrame:
    if group.empty:
        return group.copy()
    ranked = group.copy()
    if "api_id" not in ranked.columns:
        ranked["api_id"] = ""
    ranked["_rank"] = _numeric_column(ranked, "retrieved_rank")
    ranked["_functional"] = _numeric_column(ranked, "functional_match")
    return ranked.sort_values(["_rank", "api_id"], na_position="last")


def _top_ranked_irrelevant_evidence(group: pd.DataFrame) -> pd.DataFrame:
    ranked = _section_5_ranked_candidates(group)
    if ranked.empty:
        return pd.DataFrame(columns=SECTION_5_IRRELEVANT_COLUMNS)

    top5 = ranked[(ranked["_rank"] <= 5) & (ranked["_functional"] == 0)].copy()
    selected = top5 if not top5.empty else ranked[(ranked["_rank"] <= 10) & (ranked["_functional"] == 0)].copy()
    if selected.empty:
        return pd.DataFrame(columns=SECTION_5_IRRELEVANT_COLUMNS)

    return pd.DataFrame(
        {
            "API": selected.get("api_id"),
            "RAG rank": selected.get("retrieved_rank"),
            "Functional match": "Mismatch",
            "RT": selected.get("qos_rt_s"),
            "Throughput": selected.get("qos_tp_kbps"),
            "Availability": selected.get("qos_availability"),
        },
        columns=SECTION_5_IRRELEVANT_COLUMNS,
    )


def _lower_ranked_strict_qos_evidence(group: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    ranked = _section_5_ranked_candidates(group)
    if ranked.empty:
        return pd.DataFrame(columns=SECTION_5_DOMINANCE_COLUMNS), pd.DataFrame(columns=SECTION_5_DOMINANCE_DETAIL_COLUMNS), False

    functional = ranked[ranked["_functional"] == 1].copy()
    top5_functional = functional[functional["_rank"] <= 5].copy()
    lower_functional = functional[functional["_rank"] > 10].copy()

    comparison_pool = pd.concat([top5_functional, lower_functional], axis=0)
    missing_qos_excluded = bool(not comparison_pool.empty and ~comparison_pool.apply(_has_complete_qos, axis=1).all())
    top5_complete = top5_functional[top5_functional.apply(_has_complete_qos, axis=1)].copy()
    lower_complete = lower_functional[lower_functional.apply(_has_complete_qos, axis=1)].copy()

    dominance_rows: dict[str, dict[str, Any]] = {}
    detail_rows: list[dict[str, Any]] = []
    for _, lower in lower_complete.sort_values(["_rank", "api_id"]).iterrows():
        for _, compared in top5_complete.sort_values(["_rank", "api_id"]).iterrows():
            if _strict_qos_dominates(lower, compared):
                api_id = str(lower.get("api_id") or "")
                if api_id not in dominance_rows:
                    dominance_rows[api_id] = {
                        "API": lower.get("api_id"),
                        "RAG rank": lower.get("retrieved_rank"),
                        "RT": lower.get("qos_rt_s"),
                        "Throughput": lower.get("qos_tp_kbps"),
                        "Availability": lower.get("qos_availability"),
                        "Dominates top-5 candidates": 0,
                    }
                dominance_rows[api_id]["Dominates top-5 candidates"] += 1
                detail_rows.append(_section_5_dominance_detail_row(lower, compared))
    if dominance_rows:
        main = pd.DataFrame(dominance_rows.values(), columns=SECTION_5_DOMINANCE_COLUMNS)
        main["_rank_sort"] = pd.to_numeric(main["RAG rank"], errors="coerce")
        main = main.sort_values(["Dominates top-5 candidates", "_rank_sort", "API"], ascending=[False, True, True])

        details = pd.DataFrame(detail_rows, columns=SECTION_5_DOMINANCE_DETAIL_COLUMNS)
        details["_lower_rank_sort"] = pd.to_numeric(details["Lower-ranked RAG rank"], errors="coerce")
        details["_compared_rank_sort"] = pd.to_numeric(details["Compared RAG rank"], errors="coerce")
        details = details.sort_values(
            ["_lower_rank_sort", "_compared_rank_sort", "Lower-ranked API", "Compared top-5 API"],
            ascending=[True, True, True, True],
        )
        return main[SECTION_5_DOMINANCE_COLUMNS], details[SECTION_5_DOMINANCE_DETAIL_COLUMNS], missing_qos_excluded

    return pd.DataFrame(columns=SECTION_5_DOMINANCE_COLUMNS), pd.DataFrame(columns=SECTION_5_DOMINANCE_DETAIL_COLUMNS), missing_qos_excluded


def _render_reranking_motivation(
    retrieval_df: pd.DataFrame,
    focused_subtask_id: str,
    *,
    exact: bool,
    query_id: str | None = None,
    defense_core_mode: bool = False,
) -> str:
    st.header("5. Re-ranking Motivation")
    if retrieval_df.empty or "subtask_id" not in retrieval_df.columns:
        st.warning("No retrieval rows are available for the focused subtask.")
        return str(focused_subtask_id)

    subtask_ids = sorted(retrieval_df["subtask_id"].dropna().astype(str).unique().tolist(), key=_subtask_sort_key)
    if not subtask_ids:
        st.warning("No retrieval rows are available for the focused subtask.")
        return str(focused_subtask_id)

    default_index = subtask_ids.index(str(focused_subtask_id)) if str(focused_subtask_id) in subtask_ids else 0
    selected_subtask_id = st.selectbox(
        "Re-ranking subtask",
        subtask_ids,
        index=default_index,
        format_func=lambda subtask_id: f"Subtask {subtask_id}",
        key=f"reranking_motivation_subtask_{query_id or 'current'}",
    )

    group = retrieval_df[retrieval_df["subtask_id"].astype(str) == str(selected_subtask_id)].copy()
    if group.empty:
        st.warning("No retrieval rows are available for the focused subtask.")
        return str(selected_subtask_id)

    st.write(
        "RAG retrieval gives a semantically relevant candidate pool, but RAG rank alone is not enough for final API "
        "selection. Some top-ranked APIs can be functionally unsuitable, while some candidates beyond the top 10 can "
        "still be functional and have stronger QoS. This motivates re-ranking before composition planning."
    )

    fm = _numeric_column(group, "functional_match")
    retrieved_ranks = _numeric_column(group, "retrieved_rank")
    functional_count = int((fm == 1).sum())
    nonfunctional_count = int((fm == 0).sum())
    nonfunctional_top5 = int(((fm == 0) & (retrieved_ranks <= 5)).sum())
    functional_below_top10 = int(((fm == 1) & (retrieved_ranks > 10)).sum())
    has_functional_data = bool(fm.notna().any())
    has_rank_data = bool(retrieved_ranks.notna().any())

    metrics = [
        ("Functional candidates", str(functional_count) if has_functional_data else "Not available in artifact"),
        ("Nonfunctional candidates", str(nonfunctional_count) if has_functional_data else "Not available in artifact"),
        (
            "Nonfunctional APIs in RAG top-5",
            str(nonfunctional_top5) if has_functional_data and has_rank_data else "Not available in artifact",
        ),
        (
            "Functional candidates below RAG top-10",
            str(functional_below_top10) if has_functional_data and has_rank_data else "Not available in artifact",
        ),
    ]
    cols = st.columns(4)
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)

    st.subheader("Top-ranked but functionally irrelevant")
    st.caption("High RAG rank does not guarantee functional suitability.")
    irrelevant_evidence = _top_ranked_irrelevant_evidence(group)
    if irrelevant_evidence.empty:
        st.info("No top-ranked functionally irrelevant candidate was found for this selected subtask.")
    else:
        display = _display_frame(irrelevant_evidence, exact=exact).replace("n/a", "Not available in artifact")
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            height=220,
            column_config=_pinned_columns("API", "RAG rank"),
        )

    st.subheader("Beyond top-10 but stronger QoS")
    st.caption("Useful APIs may appear deeper in the RAG candidate pool.")
    evidence, details, missing_qos_excluded = _lower_ranked_strict_qos_evidence(group)
    if evidence.empty:
        st.info("No functional candidate beyond RAG top-10 with stronger QoS was found for this selected subtask.")
    else:
        display = _display_frame(evidence.head(5), exact=exact).replace("n/a", "Not available in artifact")
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            height=220,
            column_config=_pinned_columns("API", "RAG rank"),
        )
        with st.expander("Optional: Show detailed dominance comparisons", expanded=False):
            st.dataframe(
                _display_frame(details, exact=exact).replace("n/a", "Not available in artifact"),
                use_container_width=True,
                hide_index=True,
                height=320,
                column_config=_pinned_columns("Lower-ranked API", "Lower-ranked RAG rank"),
            )
    if missing_qos_excluded:
        st.caption("Some candidates were excluded because QoS values were unavailable.")
    if not defense_core_mode:
        st.write("The next section shows how each selection mode re-ranks this same candidate pool.")
    return str(selected_subtask_id)


def _rank_style(row: pd.Series, mode: str, *, selected: bool = False) -> list[str]:
    fm = _to_float(row.get("functional_match"))
    if mode == "qos_topsis" and selected and fm == 0:
        return ["background-color: #FEE2E2; color: #111827; font-weight: 700;" for _ in row]
    if selected:
        return ["background-color: #FEF3C7; color: #111827; font-weight: 700;" for _ in row]
    if fm == 1:
        return ["background-color: #ECFDF5; color: #111827;" for _ in row]
    if fm == 0:
        return ["background-color: #F3F4F6; color: #111827;" for _ in row]
    return ["" for _ in row]


def _render_ranking_by_mode(
    ranking_df: pd.DataFrame,
    selected_modes: list[str],
    top_k: int,
    *,
    show_all: bool,
    exact: bool,
    defense_core_mode: bool = False,
) -> None:
    st.header("6. Ranking by Mode")
    if defense_core_mode:
        st.write("Each mode re-ranks the same retrieved candidate pool before planning.")
        with st.expander("Optional: View ranking tables by mode.", expanded=False):
            _render_ranking_by_mode_tables(ranking_df, selected_modes, top_k, show_all=show_all, exact=exact)
        return

    _render_ranking_by_mode_tables(ranking_df, selected_modes, top_k, show_all=show_all, exact=exact)


def _render_ranking_by_mode_tables(
    ranking_df: pd.DataFrame,
    selected_modes: list[str],
    top_k: int,
    *,
    show_all: bool,
    exact: bool,
) -> None:
    if ranking_df.empty:
        st.warning("Candidate API ranking rows are unavailable.")
        return
    modes = [mode for mode in selected_modes if mode in set(ranking_df["mode"].astype(str))]
    if not modes:
        st.warning("No ranking rows match the visible modes.")
        return

    view_limit = 40 if show_all else top_k
    tabs = st.tabs([loader.mode_label(mode) for mode in modes])
    display_cols = [
        "subtask_id",
        "mode_rank",
        "retrieved_rank",
        "api_id",
        "display_name",
        "functional_match",
        "qos_rt_s",
        "qos_tp_kbps",
        "qos_availability",
    ]
    for tab, mode in zip(tabs, modes):
        with tab:
            mode_rows = ranking_df[ranking_df["mode"] == mode].copy()
            mode_rows = mode_rows.sort_values(["subtask_id", "mode_rank", "retrieved_rank"])
            mode_rows = mode_rows.groupby("subtask_id", group_keys=False).head(view_limit)
            selected_flags = pd.Series(False, index=mode_rows.index)
            if "selected_for_planner" in mode_rows.columns:
                selected_flags = mode_rows["selected_for_planner"].astype(str).str.lower().eq("yes")
            view = mode_rows[[col for col in display_cols if col in mode_rows.columns]].copy()
            display = _display_frame(view, exact=exact)
            styled = display.style.apply(
                lambda row: _rank_style(row, mode, selected=bool(selected_flags.get(row.name, False))),
                axis=1,
            )
            st.dataframe(
                styled,
                use_container_width=True,
                hide_index=True,
                height=430 if show_all else 330,
                column_config=_pinned_columns("subtask_id", "mode_rank", "retrieved_rank", "api_id"),
            )
            if mode == "qos_topsis":
                st.caption("Red rows mark TOPSIS planner-selected APIs with functional_match = 0. Amber rows mark planner selections; green rows mark functional matches.")
            else:
                st.caption("Amber rows mark APIs selected in the final planner path; green rows mark functional matches; gray rows mark nonfunctional candidates.")


def _candidate_rank_rows(
    retrieval_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    paths: dict[str, list[dict[str, Any]]],
    subtask_id: str,
    top_k: int,
) -> pd.DataFrame:
    focused_retrieval = retrieval_df[retrieval_df["subtask_id"].astype(str) == str(subtask_id)].copy() if not retrieval_df.empty else pd.DataFrame()
    focused_ranking = ranking_df[ranking_df["subtask_id"].astype(str) == str(subtask_id)].copy() if not ranking_df.empty else pd.DataFrame()
    candidate_ids: list[str] = []

    for api_id in focused_retrieval.sort_values("retrieved_rank").head(top_k).get("api_id", []):
        candidate_ids.append(str(api_id))
    for mode in loader.MODE_ORDER:
        mode_rows = focused_ranking[focused_ranking["mode"].astype(str) == mode].sort_values("mode_rank")
        for api_id in mode_rows.head(top_k).get("api_id", []):
            candidate_ids.append(str(api_id))
    for mode_path in paths.values():
        for step in mode_path:
            if str(step.get("subtask_id") or "") == str(subtask_id):
                candidate_ids.append(str(step.get("api_id") or ""))

    ordered_ids = []
    seen: set[str] = set()
    for api_id in candidate_ids:
        if api_id and api_id not in seen:
            seen.add(api_id)
            ordered_ids.append(api_id)

    rows = []
    for api_id in ordered_ids:
        retrieval_rows = focused_retrieval[focused_retrieval["api_id"].astype(str) == api_id]
        base = retrieval_rows.iloc[0].to_dict() if not retrieval_rows.empty else {}
        row = {
            "candidate": loader.short_api_name(api_id, max_parts=5),
            "api_id": api_id,
            RANK_COLUMNS["rag"]: base.get("retrieved_rank"),
            "functional_match": base.get("functional_match"),
            "selected_by_modes": ", ".join(_selected_modes_for_api(paths, subtask_id, api_id)) or base.get("selected_by_modes", ""),
        }
        for mode in loader.MODE_ORDER:
            mode_rows = focused_ranking[
                (focused_ranking["mode"].astype(str) == mode) & (focused_ranking["api_id"].astype(str) == api_id)
            ]
            row[RANK_COLUMNS[mode]] = mode_rows.iloc[0].get("mode_rank") if not mode_rows.empty else None
        rows.append(row)
    return _numeric_frame(pd.DataFrame(rows))


def _render_rank_heatmap(rank_rows: pd.DataFrame) -> None:
    st.markdown("**Candidate Rank Heatmap**")
    if rank_rows.empty:
        st.warning("No focused candidate set could be built for the selected subtask.")
        return
    rank_cols = list(RANK_COLUMNS.values())
    plot_df = rank_rows.set_index("candidate")[[col for col in rank_cols if col in rank_rows.columns]]
    numeric = plot_df.apply(pd.to_numeric, errors="coerce")
    max_rank = int(numeric.max().max()) if numeric.notna().any().any() else 40
    z = numeric.fillna(max_rank + 1)
    try:
        text = numeric.map(lambda value: "" if pd.isna(value) else str(int(value)))
    except AttributeError:  # pragma: no cover - compatibility with older pandas.
        text = numeric.applymap(lambda value: "" if pd.isna(value) else str(int(value)))
    fig = go.Figure(
        data=go.Heatmap(
            z=z.values,
            x=list(z.columns),
            y=list(z.index),
            text=text.values,
            texttemplate="%{text}",
            colorscale="Blues_r",
            colorbar=dict(title="Rank"),
        )
    )
    fig.update_layout(height=max(300, 32 * len(rank_rows)), margin=dict(l=10, r=10, t=20, b=55), xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    badge_cols = ["api_id", "functional_match", "selected_by_modes"] + [col for col in rank_cols if col in rank_rows.columns]
    st.dataframe(_display_frame(rank_rows[badge_cols], exact=False), use_container_width=True, hide_index=True, height=230)
    st.caption("Rows include top-K RAG candidates, top-K candidates from each mode, and all APIs selected by any final composition path.")


def _render_slopegraph(rank_rows: pd.DataFrame, selected_modes: list[str]) -> None:
    st.markdown("**Rank Movement / Slopegraph**")
    if rank_rows.empty:
        st.info("Rank movement is unavailable because no focused candidate set was built.")
        return
    modes = [mode for mode in ["no_qos", "qos_pure_llm", "qos_hybrid"] if mode in selected_modes and RANK_COLUMNS[mode] in rank_rows.columns]
    if not modes:
        st.info("Rank movement needs No-QoS, QoS-Pure-LLM, or QoS-Hybrid ranking rows.")
        return

    plot_rows = []
    for _, row in rank_rows.head(12).iterrows():
        rag_rank = _to_float(row.get(RANK_COLUMNS["rag"]))
        if rag_rank is None:
            continue
        for mode in modes:
            mode_rank = _to_float(row.get(RANK_COLUMNS[mode]))
            if mode_rank is None:
                continue
            plot_rows.append({"candidate": row["candidate"], "stage": "RAG", "rank": rag_rank, "target": loader.mode_label(mode)})
            plot_rows.append({"candidate": row["candidate"], "stage": loader.mode_label(mode), "rank": mode_rank, "target": loader.mode_label(mode)})
    if not plot_rows:
        st.info("No overlapping RAG and mode ranks were available for the slopegraph.")
        return

    fig = go.Figure()
    for (candidate, target), group in pd.DataFrame(plot_rows).groupby(["candidate", "target"]):
        fig.add_trace(
            go.Scatter(
                x=group["stage"],
                y=group["rank"],
                mode="lines+markers",
                name=f"{candidate} -> {target}",
                hovertemplate="%{text}<br>%{x}: rank %{y}<extra></extra>",
                text=[candidate] * len(group),
                showlegend=False,
            )
        )
    fig.update_yaxes(autorange="reversed", title="Rank, lower is better")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=45), xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The slopegraph compares RAG rank to mode-specific rank for selected or top candidates.")


def _ranked_list(ranking_df: pd.DataFrame, subtask_id: str, mode: str) -> list[str]:
    if ranking_df.empty:
        return []
    rows = ranking_df[
        (ranking_df["subtask_id"].astype(str) == str(subtask_id)) & (ranking_df["mode"].astype(str) == mode)
    ].sort_values("mode_rank")
    return rows["api_id"].astype(str).tolist()


def _spearman_lists(left: list[str], right: list[str]) -> float | None:
    if not left or not right:
        return None
    union = list(dict.fromkeys(left + right))
    if len(union) < 2:
        return 1.0
    missing_rank = len(union) + 1
    left_ranks = {api_id: rank for rank, api_id in enumerate(left, start=1)}
    right_ranks = {api_id: rank for rank, api_id in enumerate(right, start=1)}
    left_series = pd.Series([left_ranks.get(api_id, missing_rank) for api_id in union])
    right_series = pd.Series([right_ranks.get(api_id, missing_rank) for api_id in union])
    value = left_series.corr(right_series, method="spearman")
    return None if pd.isna(value) else float(value)


def _jaccard_at_k(left: list[str], right: list[str], k: int) -> float | None:
    left_set = set(left[:k])
    right_set = set(right[:k])
    union = left_set | right_set
    if not union:
        return None
    return len(left_set & right_set) / len(union)


def _pairwise_matrix(lists: dict[str, list[str]], metric: str, k: int) -> pd.DataFrame:
    labels = [loader.mode_label(mode) for mode in lists]
    matrix = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for left_mode, left_items in lists.items():
        for right_mode, right_items in lists.items():
            value = _spearman_lists(left_items, right_items) if metric == "spearman" else _jaccard_at_k(left_items, right_items, k)
            matrix.loc[loader.mode_label(left_mode), loader.mode_label(right_mode)] = value
    return matrix


def _render_similarity_heatmap(ranking_df: pd.DataFrame, selected_modes: list[str], focused_subtask_id: str, top_k: int) -> None:
    st.markdown("**Pairwise Mode Similarity Heatmap**")
    lists = {
        mode: _ranked_list(ranking_df, focused_subtask_id, mode)
        for mode in selected_modes
        if mode in loader.MODE_ORDER and _ranked_list(ranking_df, focused_subtask_id, mode)
    }
    if len(lists) < 2:
        st.info("Pairwise mode similarity requires at least two visible modes with ranking rows.")
        return

    st.caption("Derived diagnostic for selected query/subtask, not an official aggregate result.")
    tabs = st.tabs(["Spearman", f"Jaccard@{top_k}"])
    for tab, metric in zip(tabs, ["spearman", "jaccard"]):
        with tab:
            if metric == "spearman":
                st.caption(
                    "Spearman: 1.000 = identical or almost identical ranking order; "
                    "0.000 = little or no ranking agreement; -1.000 = opposite ranking order."
                )
            else:
                st.caption(f"Jaccard@{top_k}: 1.000 = same top-{top_k} candidate set; 0.000 = no overlap in top-{top_k} candidates.")
            matrix = _pairwise_matrix(lists, metric, top_k)
            fig = px.imshow(matrix, text_auto=".3f", zmin=-1 if metric == "spearman" else 0, zmax=1, color_continuous_scale="RdYlGn")
            fig.update_traces(hovertemplate="%{y} vs %{x}<br>Value: %{z:.3f}<extra></extra>")
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=45))
            st.plotly_chart(fig, use_container_width=True)
            matrix_display = matrix.reset_index().rename(columns={"index": "Mode"})
            st.dataframe(_display_frame(matrix_display, exact=False), use_container_width=True, hide_index=True, height=180)
    st.caption("Average Overlap@K and RBO are not shown on this page unless a project implementation is wired in; Spearman and Jaccard@K are computed locally.")


def _render_ranking_difference(
    retrieval_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    paths: dict[str, list[dict[str, Any]]],
    selected_modes: list[str],
    focused_subtask_id: str,
    top_k: int,
    *,
    defense_core_mode: bool = False,
) -> None:
    st.header("7. Ranking Difference Visualization")
    if defense_core_mode:
        with st.expander("Optional: Pairwise ranking similarity heatmap.", expanded=False):
            _render_ranking_difference_body(
                retrieval_df,
                ranking_df,
                paths,
                selected_modes,
                focused_subtask_id,
                top_k,
            )
        return

    _render_ranking_difference_body(
        retrieval_df,
        ranking_df,
        paths,
        selected_modes,
        focused_subtask_id,
        top_k,
    )


def _render_ranking_difference_body(
    retrieval_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    paths: dict[str, list[dict[str, Any]]],
    selected_modes: list[str],
    focused_subtask_id: str,
    top_k: int,
) -> None:
    subtask_values: set[str] = set()
    if not ranking_df.empty and "subtask_id" in ranking_df.columns:
        subtask_values.update(ranking_df["subtask_id"].dropna().astype(str).tolist())
    if not retrieval_df.empty and "subtask_id" in retrieval_df.columns:
        subtask_values.update(retrieval_df["subtask_id"].dropna().astype(str).tolist())
    subtask_options = sorted(subtask_values, key=_subtask_sort_key)
    if not subtask_options:
        st.info("Ranking difference visualization requires subtask-level ranking rows.")
        return

    default_index = subtask_options.index(str(focused_subtask_id)) if str(focused_subtask_id) in subtask_options else 0
    selected_subtask_id = st.selectbox(
        "Ranking difference subtask",
        subtask_options,
        index=default_index,
        format_func=lambda subtask_id: f"Subtask {subtask_id}",
    )
    st.caption(f"Focused diagnostic for Subtask {selected_subtask_id}. Only pairwise ranking similarity is shown here.")
    _render_similarity_heatmap(ranking_df, selected_modes, str(selected_subtask_id), top_k)


def _path_rows(path: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for step in path:
        rows.append(
            {
                "subtask_id": step.get("subtask_id"),
                "subtask description": step.get("subtask"),
                "api_id": step.get("api_id"),
                "display_name": step.get("display_name"),
                "functional_match": step.get("functional_match"),
                "qos_rt_s": step.get("qos_rt_s"),
                "qos_tp_kbps": step.get("qos_tp_kbps"),
                "qos_availability": step.get("qos_availability"),
                "mode_rank": step.get("mode_rank"),
                "retrieved_rank": step.get("retrieved_rank"),
            }
        )
    return _numeric_frame(pd.DataFrame(rows))


def _render_single_path(mode: str, path: list[dict[str, Any]], *, exact: bool) -> None:
    if not path:
        st.info(f"No selected composition path was loaded for {loader.mode_label(mode)}.")
        return
    path_text = " -> ".join(f"Subtask {row.get('subtask_id')}: {row.get('api_id')}" for row in path)
    st.markdown(f"<div class='live-demo-path-line'>{escape(path_text)}</div>", unsafe_allow_html=True)
    view = _path_rows(path)
    st.dataframe(_display_frame(view, exact=exact), use_container_width=True, hide_index=True, height=260)


def _compare_paths(left_mode: str, right_mode: str, paths: dict[str, list[dict[str, Any]]], *, exact: bool) -> pd.DataFrame:
    left = {str(row.get("subtask_id")): row for row in paths.get(left_mode, [])}
    right = {str(row.get("subtask_id")): row for row in paths.get(right_mode, [])}
    subtask_ids = sorted(set(left) | set(right), key=_subtask_sort_key)
    rows = []
    for subtask_id in subtask_ids:
        left_row = left.get(subtask_id, {})
        right_row = right.get(subtask_id, {})
        rows.append(
            {
                "subtask_id": subtask_id,
                f"{loader.mode_label(left_mode)} api": left_row.get("api_id", ""),
                f"{loader.mode_label(left_mode)} FM": left_row.get("functional_match"),
                f"{loader.mode_label(left_mode)} RT_s": left_row.get("qos_rt_s"),
                f"{loader.mode_label(right_mode)} api": right_row.get("api_id", ""),
                f"{loader.mode_label(right_mode)} FM": right_row.get("functional_match"),
                f"{loader.mode_label(right_mode)} RT_s": right_row.get("qos_rt_s"),
                "same_api": "Yes" if left_row.get("api_id") and left_row.get("api_id") == right_row.get("api_id") else "No",
            }
        )
    return _display_frame(_numeric_frame(pd.DataFrame(rows)), exact=exact)


def _selected_path_frame(paths: dict[str, list[dict[str, Any]]], modes: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for mode in modes:
        for step in paths.get(mode, []):
            api_id = str(step.get("api_id") or "")
            rows.append(
                {
                    "mode": mode,
                    "Mode": loader.mode_label(mode),
                    "subtask_id": str(step.get("subtask_id") or ""),
                    "api_id": api_id,
                    "api_short": loader.short_api_name(api_id, max_parts=4),
                    "functional_match": _to_float(step.get("functional_match")),
                    "functional_status": "Functional match" if _to_float(step.get("functional_match")) == 1 else "Functional mismatch",
                    "qos_rt_s": _to_float(step.get("qos_rt_s")),
                    "qos_tp_kbps": _to_float(step.get("qos_tp_kbps")),
                    "qos_availability": _to_float(step.get("qos_availability")),
                    "mode_rank": _to_float(step.get("mode_rank")),
                    "retrieved_rank": _to_float(step.get("retrieved_rank")),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["_subtask_order"] = out["subtask_id"].map(lambda value: _subtask_sort_key(value)[0])
        out["_mode_order"] = out["mode"].map(lambda mode: loader.mode_sort_key(mode)[0])
        out = out.sort_values(["_mode_order", "_subtask_order"]).drop(columns=["_mode_order", "_subtask_order"])
    return out


def _path_rollup(path_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode, group in path_df.groupby("mode", sort=False):
        rt = pd.to_numeric(group["qos_rt_s"], errors="coerce").dropna()
        tp = pd.to_numeric(group["qos_tp_kbps"], errors="coerce").dropna()
        av = pd.to_numeric(group["qos_availability"], errors="coerce").dropna()
        fm = pd.to_numeric(group["functional_match"], errors="coerce").dropna()
        rows.append(
            {
                "Mode": loader.mode_label(mode),
                "Functional_Coverage_from_path": float((fm == 1).sum() / len(group)) if len(group) else None,
                "Total_Response_Time_s_from_path": float(rt.sum()) if len(rt) == len(group) else None,
                "Bottleneck_Throughput_kbps_from_path": float(tp.min()) if len(tp) == len(group) else None,
                "Average_Workflow_Availability_from_path": float(av.mean()) if len(av) == len(group) else None,
            }
        )
    return pd.DataFrame(rows)


def _render_path_figures(
    path_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    modes: list[str],
    *,
    exact: bool,
    include_rollup: bool = True,
) -> None:
    if path_df.empty:
        st.info("No selected path rows are available for path figures.")
        return

    st.markdown("**Selected API matrix**")
    fig = px.scatter(
        path_df,
        x="subtask_id",
        y="Mode",
        color="functional_status",
        text="api_short",
        symbol="functional_status",
        color_discrete_map={"Functional match": "#059669", "Functional mismatch": "#DC2626"},
        category_orders={"Mode": [loader.mode_label(mode) for mode in loader.MODE_ORDER]},
        hover_data={
            "api_id": True,
            "qos_rt_s": ":.6f",
            "qos_tp_kbps": ":.6f",
            "qos_availability": ":.6f",
            "mode_rank": True,
            "retrieved_rank": True,
            "functional_status": False,
            "api_short": False,
        },
        labels={"subtask_id": "Subtask", "Mode": "Mode"},
    )
    fig.update_traces(marker=dict(size=28, line=dict(width=1, color="#111827")), textposition="middle right")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=45), legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Each marker is the API fixed into the planner path for one mode/subtask. Red marks functional mismatches; hover for all QoS values.")

    st.markdown("**Normalized QoS by selected path**")
    qos_rows = eval_df[eval_df["Mode"].astype(str).isin(modes)].copy() if not eval_df.empty and "Mode" in eval_df else pd.DataFrame()
    if qos_rows.empty or "Normalized_QoS_Score" not in qos_rows:
        st.info("Normalized QoS rows are unavailable for the visible modes.")
    else:
        qos_rows["Mode_Label"] = qos_rows["Mode"].map(loader.MODE_LABELS).fillna(qos_rows["Mode"])
        qos_rows["_mode_order"] = qos_rows["Mode"].map(lambda mode: loader.mode_sort_key(str(mode))[0])
        qos_rows = qos_rows.sort_values(["_mode_order", "Mode"])
        fig = px.bar(
            qos_rows,
            x="Mode_Label",
            y="Normalized_QoS_Score",
            color="Mode",
            text="Normalized_QoS_Score",
            color_discrete_map=MODE_COLORS,
            labels={"Mode_Label": "Mode", "Normalized_QoS_Score": "Normalized QoS"},
            hover_data=[
                col
                for col in ["Total_Response_Time_s", "Bottleneck_Throughput_kbps", "Average_Workflow_Availability"]
                if col in qos_rows
            ],
        )
        fig.update_traces(texttemplate="%{y:.3f}", textposition="outside", cliponaxis=False)
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=25, b=45), showlegend=False, yaxis_range=[0, 1.08], xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Normalized QoS is the official artifact value for each mode's selected composition path.")

    if include_rollup:
        _render_path_rollup(path_df, exact=exact)


def _render_path_rollup(path_df: pd.DataFrame, *, exact: bool) -> None:
    rollup = _path_rollup(path_df)
    st.dataframe(_display_frame(rollup, exact=exact), use_container_width=True, hide_index=True, height=180)
    st.caption("Rollups are derived from selected API rows as an explanatory check; official score rows are shown in the next section.")


def _render_all_mode_paths(paths: dict[str, list[dict[str, Any]]], modes: list[str], *, exact: bool) -> None:
    st.markdown("**All mode paths**")
    tabs = st.tabs([loader.mode_label(mode) for mode in modes])
    for tab, mode in zip(tabs, modes):
        with tab:
            _render_single_path(mode, paths.get(mode, []), exact=exact)


def _render_selected_paths(
    bundle: dict[str, Any],
    selected_modes: list[str],
    eval_df: pd.DataFrame,
    *,
    exact: bool,
    defense_core_mode: bool = False,
) -> None:
    st.header("8. Selected Composition Path")
    paths = bundle.get("selected_paths") or {}
    modes = [mode for mode in selected_modes if mode in paths]
    if not modes:
        st.warning("No planner paths are available for the visible modes.")
        return

    path_df = _selected_path_frame(paths, modes)
    _render_path_figures(path_df, eval_df, modes, exact=exact, include_rollup=not defense_core_mode)

    if defense_core_mode:
        with st.expander("Optional: Selected path details.", expanded=False):
            if not path_df.empty:
                _render_path_rollup(path_df, exact=exact)
            _render_all_mode_paths(paths, modes, exact=exact)
    else:
        _render_all_mode_paths(paths, modes, exact=exact)

    if "qos_topsis" in paths:
        topsis_path = paths.get("qos_topsis", [])
        if any(_to_float(row.get("functional_match")) == 0 for row in topsis_path):
            st.markdown(
                "<div class='live-demo-status live-demo-warning'>QoS-TOPSIS is diagnostic here: the selected-path figures show where QoS-oriented choices lose functional suitability.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("QoS-TOPSIS has no selected functional mismatch for this query; it remains a diagnostic comparison mode rather than the primary thesis claim.")


def _render_score_components(visible: pd.DataFrame, *, exact: bool) -> None:
    component_rows = []
    for _, row in visible.iterrows():
        mode_label = loader.mode_label(str(row.get("Mode")))
        completeness = _to_float(row.get("Composition_Completeness"))
        functional = _to_float(row.get("Functional_Coverage"))
        qos = _to_float(row.get("Normalized_QoS_Score"))
        official = _to_float(row.get(SCORE_COL))
        if completeness is None or functional is None or qos is None:
            continue
        functional_contribution = 0.70 * functional
        qos_contribution = 0.30 * qos
        combined = functional_contribution + qos_contribution
        computed = completeness * combined
        component_rows.extend(
            [
                {
                    "Mode": mode_label,
                    "Component": "0.70 x Functional Coverage",
                    "Value": functional_contribution,
                    "Composition Completeness": completeness,
                    "Derived formula value": computed,
                    "Official artifact score": official,
                },
                {
                    "Mode": mode_label,
                    "Component": "0.30 x Normalized QoS",
                    "Value": qos_contribution,
                    "Composition Completeness": completeness,
                    "Derived formula value": computed,
                    "Official artifact score": official,
                },
            ]
        )

    if not component_rows:
        st.info("Score component values are unavailable for the visible modes.")
        return
    st.caption("Derived visual explanation only, not an official experiment artifact. Official scores remain the loaded artifact values.")
    fig = px.bar(
        pd.DataFrame(component_rows),
        x="Mode",
        y="Value",
        color="Component",
        barmode="stack",
        color_discrete_map={
            "0.70 x Functional Coverage": "#059669",
            "0.30 x Normalized QoS": "#2563EB",
        },
        labels={"Value": "Weighted contribution"},
        hover_data=["Composition Completeness", "Derived formula value", "Official artifact score"],
    )
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=25, b=45), yaxis_range=[0, 1.05])
    st.plotly_chart(fig, use_container_width=True)


def _render_score_comparison(eval_df: pd.DataFrame, selected_modes: list[str], *, exact: bool) -> None:
    st.header("9. Score Comparison")
    visible = eval_df[eval_df["Mode"].isin(selected_modes)].copy() if not eval_df.empty else pd.DataFrame()
    if visible.empty or SCORE_COL not in visible:
        st.warning("Composition score rows are unavailable for the selected modes.")
        return
    visible["Mode_Label"] = visible["Mode"].map(loader.MODE_LABELS).fillna(visible["Mode"])
    fig = px.bar(
        visible,
        x="Mode_Label",
        y=SCORE_COL,
        color="Mode",
        text=SCORE_COL,
        color_discrete_map=MODE_COLORS,
        category_orders={"Mode_Label": [loader.mode_label(mode) for mode in loader.MODE_ORDER]},
        range_y=[0, 1.08],
        labels={SCORE_COL: "QoS-Adjusted Composition Score", "Mode_Label": "Mode"},
        hover_data=[
            col
            for col in ["Functional_Coverage", "Normalized_QoS_Score", "Total_Response_Time_s", "Bottleneck_Throughput_kbps"]
            if col in visible
        ],
    )
    fig.update_traces(texttemplate="%{y:.3f}", textposition="outside", cliponaxis=False)
    fig.update_layout(height=390, margin=dict(l=10, r=10, t=25, b=45), showlegend=False, yaxis_title="QoS-Adjusted Composition Score", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Primary comparison: QoS-Pure-LLM versus No-QoS. QoS-Hybrid is a bonus refinement; QoS-TOPSIS is diagnostic.")

    table_cols = [col for col in COMPACT_SCORE_COLUMNS if col in visible.columns]
    display = visible[table_cols].copy()
    display["Mode"] = display["Mode"].map(loader.MODE_LABELS).fillna(display["Mode"])
    st.dataframe(
        _display_frame(display, exact=exact),
        use_container_width=True,
        hide_index=True,
        height=210,
        column_config=_pinned_columns("Mode"),
    )


def _valid_qos_row(row: dict[str, Any]) -> bool:
    return all(
        _to_float(row.get(key)) is not None
        for key in ["Total_Response_Time_s", "Bottleneck_Throughput_kbps", "Average_Workflow_Availability"]
    )


def _qos_reference_values(eval_df: pd.DataFrame) -> dict[str, float | None]:
    rows = [row.to_dict() for _, row in eval_df.iterrows()]
    complete_rows = [
        row
        for row in rows
        if (_to_float(row.get("Composition_Completeness")) or 0.0) >= 1.0 and _valid_qos_row(row)
    ]
    if complete_rows:
        max_fc = max((_to_float(row.get("Functional_Coverage")) or 0.0) for row in complete_rows)
        reference_rows = [row for row in complete_rows if (_to_float(row.get("Functional_Coverage")) or 0.0) == max_fc]
    else:
        reference_rows = [row for row in rows if _valid_qos_row(row)]
    return {
        "best_response_time": min((_to_float(row.get("Total_Response_Time_s")) for row in reference_rows), default=None),
        "best_throughput": max((_to_float(row.get("Bottleneck_Throughput_kbps")) for row in reference_rows), default=None),
        "best_availability": max((_to_float(row.get("Average_Workflow_Availability")) for row in reference_rows), default=None),
    }


def _path_metric_components(path: list[dict[str, Any]]) -> dict[str, Any]:
    rt_values = [_to_float(row.get("qos_rt_s")) for row in path]
    tp_values = [_to_float(row.get("qos_tp_kbps")) for row in path]
    av_values = [_to_float(row.get("qos_availability")) for row in path]
    fm_values = [_to_float(row.get("functional_match")) for row in path]
    rt = [value for value in rt_values if value is not None]
    tp = [value for value in tp_values if value is not None]
    av = [value for value in av_values if value is not None]
    fm = [value for value in fm_values if value is not None]
    return {
        "planned_count": len(path),
        "functional_count": int(sum(1 for value in fm if value == 1)),
        "functional_coverage": (sum(1 for value in fm if value == 1) / len(path)) if path else None,
        "rt_values": rt,
        "tp_values": tp,
        "av_values": av,
        "total_rt": sum(rt) if len(rt) == len(path) and path else None,
        "bottleneck_tp": min(tp) if len(tp) == len(path) and path else None,
        "avg_availability": (sum(av) / len(av)) if len(av) == len(path) and path else None,
    }


def _formula_list(values: list[float], *, exact: bool) -> str:
    return " + ".join(_fmt(value, exact=exact) for value in values) if values else "n/a"


def _formula_substitution(row: dict[str, Any], *, exact: bool) -> str:
    completeness = _to_float(row.get("Composition_Completeness"))
    fc = _to_float(row.get("Functional_Coverage"))
    qos = _to_float(row.get("Normalized_QoS_Score"))
    if completeness is None or fc is None or qos is None:
        return "n/a"
    computed = completeness * (0.70 * fc + 0.30 * qos)
    return (
        f"{_fmt(completeness, exact=exact)} x "
        f"(0.70 x {_fmt(fc, exact=exact)} + 0.30 x {_fmt(qos, exact=exact)}) "
        f"= {_fmt(computed, exact=exact)}"
    )


def _formula_proof_table(eval_df: pd.DataFrame, *, exact: bool) -> pd.DataFrame:
    rows = []
    for mode in FORMULA_PROOF_MODES:
        row = _row_for_mode(eval_df, mode)
        if not row:
            continue
        rows.append(
            {
                "Mode": loader.mode_label(mode),
                "Composition Completeness": row.get("Composition_Completeness"),
                "Functional Coverage": row.get("Functional_Coverage"),
                "Normalized QoS": row.get("Normalized_QoS_Score"),
                "Formula substitution": _formula_substitution(row, exact=exact),
                "Reported experiment score": row.get(SCORE_COL),
            }
        )
    return _display_frame(pd.DataFrame(rows), exact=exact)


def _formula_interpretation(eval_df: pd.DataFrame) -> str:
    no_qos = _row_for_mode(eval_df, "no_qos")
    pure = _row_for_mode(eval_df, "qos_pure_llm")
    pure_fc = _to_float(pure.get("Functional_Coverage"))
    no_fc = _to_float(no_qos.get("Functional_Coverage"))
    pure_qos = _to_float(pure.get("Normalized_QoS_Score"))
    no_qos_score = _to_float(no_qos.get("Normalized_QoS_Score"))
    pure_score = _score(pure)
    no_score = _score(no_qos)
    if (
        pure_fc is not None
        and no_fc is not None
        and pure_qos is not None
        and no_qos_score is not None
        and pure_score is not None
        and no_score is not None
        and pure_fc >= no_fc
        and pure_qos > no_qos_score
        and pure_score > no_score
    ):
        return "QoS-Pure-LLM preserves Functional Coverage and improves Normalized QoS, which raises the final score."
    return "QoS-Pure-LLM versus No-QoS is interpreted from the loaded Functional Coverage, Normalized QoS, and reported score values above."


def _render_formula_line(
    mode: str,
    row: dict[str, Any],
    *,
    path: list[dict[str, Any]],
    references: dict[str, float | None],
    exact: bool,
) -> None:
    completeness = _to_float(row.get("Composition_Completeness"))
    fc = _to_float(row.get("Functional_Coverage"))
    qos = _to_float(row.get("Normalized_QoS_Score"))
    reported = _to_float(row.get(SCORE_COL))
    if completeness is None or fc is None or qos is None:
        st.warning(f"{loader.mode_label(mode)}: formula proof unavailable because one or more components are missing.")
        return
    computed = completeness * (0.70 * fc + 0.30 * qos)
    path_metrics = _path_metric_components(path)
    artifact_rt = _to_float(row.get("Total_Response_Time_s"))
    artifact_tp = _to_float(row.get("Bottleneck_Throughput_kbps"))
    artifact_av = _to_float(row.get("Average_Workflow_Availability"))
    norm_rt = _to_float(row.get("Normalized_Response_Time_Score"))
    norm_tp = _to_float(row.get("Normalized_Throughput_Score"))
    norm_av = _to_float(row.get("Normalized_Availability_Score"))
    best_rt = references.get("best_response_time")
    best_tp = references.get("best_throughput")
    best_av = references.get("best_availability")
    st.markdown(f"**{loader.mode_label(mode)}**")
    st.caption("Path-level values below are explanatory recomputations from selected API rows; reported experiment metrics come from artifact rows.")
    st.code(
        (
            f"Functional Coverage = matched planned APIs / planned APIs\n"
            f"                    = {path_metrics['functional_count']} / {path_metrics['planned_count']}\n"
            f"                    = {_fmt(path_metrics['functional_coverage'], exact=exact)}; artifact = {_fmt(fc, exact=exact)}\n\n"
            f"Total Response Time = sum(selected API response times)\n"
            f"                    = {_formula_list(path_metrics['rt_values'], exact=exact)}\n"
            f"                    = {_fmt(path_metrics['total_rt'], exact=exact)}; artifact = {_fmt(artifact_rt, exact=exact)}\n\n"
            f"Bottleneck Throughput = min(selected API throughputs)\n"
            f"                      = min({', '.join(_fmt(value, exact=exact) for value in path_metrics['tp_values']) or 'n/a'})\n"
            f"                      = {_fmt(path_metrics['bottleneck_tp'], exact=exact)}; artifact = {_fmt(artifact_tp, exact=exact)}\n\n"
            f"Average Availability = mean(selected API availabilities)\n"
            f"                     = ({_formula_list(path_metrics['av_values'], exact=exact)}) / {path_metrics['planned_count']}\n"
            f"                     = {_fmt(path_metrics['avg_availability'], exact=exact)}; artifact = {_fmt(artifact_av, exact=exact)}\n\n"
            f"QoS normalization references from artifact rows:\n"
            f"  best response time = {_fmt(best_rt, exact=exact)}\n"
            f"  best bottleneck throughput = {_fmt(best_tp, exact=exact)}\n"
            f"  best average availability = {_fmt(best_av, exact=exact)}\n\n"
            f"Normalized Response Time = best response time / total response time = {_fmt(best_rt, exact=exact)} / {_fmt(artifact_rt, exact=exact)} = {_fmt(norm_rt, exact=exact)}\n"
            f"Normalized Throughput = bottleneck throughput / best bottleneck throughput = {_fmt(artifact_tp, exact=exact)} / {_fmt(best_tp, exact=exact)} = {_fmt(norm_tp, exact=exact)}\n"
            f"Normalized Availability = average availability / best average availability = {_fmt(artifact_av, exact=exact)} / {_fmt(best_av, exact=exact)} = {_fmt(norm_av, exact=exact)}\n"
            f"Normalized QoS = ({_fmt(norm_rt, exact=exact)} + {_fmt(norm_tp, exact=exact)} + {_fmt(norm_av, exact=exact)}) / 3 = {_fmt(qos, exact=exact)}\n\n"
            f"Score = {_fmt(completeness, exact=exact)} x "
            f"(0.70 x {_fmt(fc, exact=exact)} + 0.30 x {_fmt(qos, exact=exact)})\n"
            f"      = {_fmt(computed, exact=exact)}\n"
            f"Reported experiment score = {_fmt(reported, exact=exact)}"
        ),
        language="text",
    )


def _render_formula_proof(
    eval_df: pd.DataFrame,
    paths: dict[str, list[dict[str, Any]]],
    selected_modes: list[str],
    *,
    exact: bool,
) -> None:
    st.header("10. Formula Proof")
    st.markdown("**Score = Composition Completeness × (0.70 × Functional Coverage + 0.30 × Normalized QoS)**")

    proof_table = _formula_proof_table(eval_df, exact=exact)
    if proof_table.empty:
        st.warning("Formula proof values are unavailable because composition evaluation rows were not loaded.")
    else:
        st.dataframe(proof_table, use_container_width=True, hide_index=True, height=190)
        st.caption("Formula substitution is an explanatory recomputation from loaded artifact components. Reported experiment score is the official artifact value.")
        st.write(_formula_interpretation(eval_df))

    if not eval_df.empty and "Mode" in eval_df:
        visible = eval_df[eval_df["Mode"].astype(str).isin(selected_modes)].copy()
        if not visible.empty:
            with st.expander("Score component breakdown", expanded=False):
                _render_score_components(visible, exact=exact)

    references = _qos_reference_values(eval_df)
    detail_titles = {
        "no_qos": "Detailed calculation: No-QoS",
        "qos_pure_llm": "Detailed calculation: QoS-Pure-LLM",
        "qos_hybrid": "Detailed calculation: QoS-Hybrid",
        "qos_topsis": "QoS-TOPSIS diagnostic calculation",
    }
    for mode in FORMULA_PROOF_MODES:
        row = _row_for_mode(eval_df, mode)
        if row:
            with st.expander(detail_titles[mode], expanded=False):
                _render_formula_line(mode, row, path=paths.get(mode, []), references=references, exact=exact)


def _score_status(left: dict[str, Any], right: dict[str, Any], greater: str, equal: str, lower: str) -> str:
    left_score = _score(left)
    right_score = _score(right)
    if left_score is None or right_score is None:
        return "Score comparison unavailable"
    if left_score > right_score:
        return greater
    if left_score == right_score:
        return equal
    return lower


def _status_box(text: str, css_class: str = "") -> str:
    class_attr = f"live-demo-status {css_class}".strip()
    return f"<div class='{escape(class_attr)}'>{escape(text)}</div>"


def _render_hypothesis_cards(eval_df: pd.DataFrame, *, exact: bool) -> None:
    st.header("11. Hypothesis Proof")
    no_qos = _row_for_mode(eval_df, "no_qos")
    pure = _row_for_mode(eval_df, "qos_pure_llm")
    hybrid = _row_for_mode(eval_df, "qos_hybrid")
    pure_delta = _metric_delta(pure, no_qos, SCORE_COL)
    hybrid_delta = _metric_delta(hybrid, pure, SCORE_COL)
    qos_delta = _metric_delta(pure, no_qos, "Normalized_QoS_Score")

    cols = st.columns(4)
    cols[0].metric("No-QoS final score", _fmt(no_qos.get(SCORE_COL), exact=exact))
    cols[1].metric("QoS-Pure-LLM final score", _fmt(pure.get(SCORE_COL), exact=exact))
    cols[2].markdown(_delta_card("Pure minus No-QoS", pure_delta, exact=exact), unsafe_allow_html=True)
    cols[3].markdown(_delta_card("Normalized QoS delta", qos_delta, exact=exact), unsafe_allow_html=True)

    pure_status = _score_status(
        pure,
        no_qos,
        "QoS-Pure-LLM beats No-QoS",
        "QoS-Pure-LLM ties No-QoS",
        "QoS-Pure-LLM does not beat No-QoS",
    )
    pure_fc = _to_float(pure.get("Functional_Coverage"))
    no_fc = _to_float(no_qos.get("Functional_Coverage"))
    if pure_fc is None or no_fc is None:
        fc_status = "Functional Coverage preservation unavailable"
    else:
        fc_status = "Functional Coverage preserved" if pure_fc >= no_fc else "Functional Coverage not preserved"
    qos_status = "Normalized QoS improved" if qos_delta is not None and qos_delta > 0 else "Normalized QoS did not improve"

    pure_class = "live-demo-status-green" if pure_status == "QoS-Pure-LLM beats No-QoS" else "live-demo-warning"
    fc_class = "live-demo-status-green" if fc_status == "Functional Coverage preserved" else "live-demo-warning"
    qos_class = "live-demo-status-green" if qos_status == "Normalized QoS improved" else "live-demo-warning"
    st.markdown(_status_box(pure_status, pure_class), unsafe_allow_html=True)
    st.markdown(_status_box(fc_status, fc_class), unsafe_allow_html=True)
    st.markdown(_status_box(qos_status, qos_class), unsafe_allow_html=True)

    bonus_cols = st.columns(3)
    bonus_cols[0].metric("QoS-Hybrid final score", _fmt(hybrid.get(SCORE_COL), exact=exact))
    bonus_cols[1].markdown(_delta_card("Hybrid minus Pure", hybrid_delta, exact=exact), unsafe_allow_html=True)
    hybrid_status = _score_status(
        hybrid,
        pure,
        "QoS-Hybrid improves further",
        "QoS-Hybrid ties QoS-Pure-LLM",
        "QoS-Hybrid does not improve further",
    )
    hybrid_class = "live-demo-bonus" if hybrid_status == "QoS-Hybrid improves further" else "live-demo-warning"
    bonus_cols[2].markdown(_status_box(hybrid_status, hybrid_class), unsafe_allow_html=True)


def _render_takeaway(eval_df: pd.DataFrame, *, exact: bool) -> None:
    st.header("12. Dynamic Query Takeaway")
    no_qos = _row_for_mode(eval_df, "no_qos")
    pure = _row_for_mode(eval_df, "qos_pure_llm")
    hybrid = _row_for_mode(eval_df, "qos_hybrid")
    topsis = _row_for_mode(eval_df, "qos_topsis")
    lines: list[str] = []

    pure_delta = _metric_delta(pure, no_qos, SCORE_COL)
    if pure_delta is not None and pure_delta > 0:
        lines.append("QoS-Pure-LLM improves over No-QoS.")

    pure_fc = _to_float(pure.get("Functional_Coverage"))
    no_fc = _to_float(no_qos.get("Functional_Coverage"))
    if pure_fc is not None and no_fc is not None and pure_fc >= no_fc:
        lines.append("The improvement preserves functional suitability.")

    qos_delta = _metric_delta(pure, no_qos, "Normalized_QoS_Score")
    if qos_delta is not None and qos_delta > 0:
        lines.append(f"Normalized QoS increases by {_fmt_delta(qos_delta, exact=exact)}.")

    hybrid_delta = _metric_delta(hybrid, pure, SCORE_COL)
    if hybrid_delta is not None and hybrid_delta > 0:
        lines.append("QoS-Hybrid improves further as a bonus functional-first refinement.")
    elif hybrid_delta == 0:
        lines.append("QoS-Hybrid ties QoS-Pure-LLM for this query.")

    topsis_fc = _to_float(topsis.get("Functional_Coverage"))
    topsis_qos = _to_float(topsis.get("Normalized_QoS_Score"))
    pure_qos = _to_float(pure.get("Normalized_QoS_Score"))
    no_qos_score = _to_float(no_qos.get("Normalized_QoS_Score"))
    baseline_fcs = [value for value in [no_fc, pure_fc] if value is not None]
    qos_high = topsis_qos is not None and (topsis_qos >= 0.70 or topsis_qos > max([value for value in [pure_qos, no_qos_score] if value is not None], default=-1))
    if topsis_fc is not None and baseline_fcs and topsis_fc < max(baseline_fcs) and qos_high:
        lines.append("QoS-TOPSIS is diagnostic: QoS alone can be insufficient when functional suitability is not preserved.")

    if not lines:
        lines.append("The selected query loaded successfully, but the available metrics do not support any predefined takeaway rule.")
    for line in lines:
        st.markdown(f"- {line}")


def _defense_relation(delta: float | None, *, metric_name: str) -> str:
    if delta is None:
        return f"{metric_name} is unavailable"
    if delta > 0:
        return f"{metric_name} improves"
    if delta == 0:
        return f"{metric_name} ties"
    return f"{metric_name} is lower"


def _render_defense_takeaway(eval_df: pd.DataFrame) -> None:
    no_qos = _row_for_mode(eval_df, "no_qos")
    pure = _row_for_mode(eval_df, "qos_pure_llm")
    hybrid = _row_for_mode(eval_df, "qos_hybrid")

    pure_delta = _metric_delta(pure, no_qos, SCORE_COL)
    qos_delta = _metric_delta(pure, no_qos, "Normalized_QoS_Score")
    hybrid_delta = _metric_delta(hybrid, pure, SCORE_COL)
    pure_fc = _to_float(pure.get("Functional_Coverage"))
    no_fc = _to_float(no_qos.get("Functional_Coverage"))
    preserves_fc = pure_fc is not None and no_fc is not None and pure_fc >= no_fc

    if pure_delta is not None and pure_delta > 0 and qos_delta is not None and qos_delta > 0 and preserves_fc:
        st.markdown(_status_box(DEFENSE_CORE_TAKEAWAY, "live-demo-status-green"), unsafe_allow_html=True)
        return

    fc_text = "Functional Coverage preservation unavailable"
    if pure_fc is not None and no_fc is not None:
        fc_text = "Functional Coverage is preserved" if preserves_fc else "Functional Coverage is lower"
    hybrid_text = _defense_relation(hybrid_delta, metric_name="QoS-Hybrid score")
    text = (
        "Takeaway: For this loaded query, "
        f"{_defense_relation(pure_delta, metric_name='QoS-Pure-LLM score')} versus No-QoS, "
        f"{fc_text}, and {_defense_relation(qos_delta, metric_name='Normalized QoS')}. "
        f"{hybrid_text} versus QoS-Pure-LLM."
    )
    st.markdown(_status_box(text, "live-demo-warning"), unsafe_allow_html=True)


def _aggregate_mode_average(score_rows: list[dict[str, Any]], mode: str) -> float | None:
    row = next((row for row in score_rows if str(row.get("Mode") or "") == mode), {})
    return _to_float(row.get("Average_QoS_Adjusted_Composition_Score"))


def _comparison_state(left: float, right: float) -> str:
    if math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
        return "equal"
    return "greater" if left > right else "lower"


def _aggregate_conclusion_boxes(score_rows: list[dict[str, Any]], *, exact: bool) -> None:
    no_qos = _aggregate_mode_average(score_rows, "no_qos")
    pure = _aggregate_mode_average(score_rows, "qos_pure_llm")
    hybrid = _aggregate_mode_average(score_rows, "qos_hybrid")

    if no_qos is None or pure is None:
        pure_text = "QoS-Pure-LLM versus No-QoS could not be compared because one or both averages are unavailable."
        pure_class = "live-demo-warning"
    else:
        pure_state = _comparison_state(pure, no_qos)
        if pure_state == "greater":
            pure_text = "Across the full query set, QoS-Pure-LLM improves the average QoS-adjusted composition score over No-QoS."
            pure_class = "live-demo-status-green"
        elif pure_state == "equal":
            pure_text = "Across the full query set, QoS-Pure-LLM ties No-QoS on average for this run."
            pure_class = "live-demo-warning"
        else:
            pure_text = (
                "Across the full query set, QoS-Pure-LLM is lower than No-QoS on average for this run "
                f"({_fmt(pure, exact=exact)} vs {_fmt(no_qos, exact=exact)})."
            )
            pure_class = "live-demo-warning"
    st.markdown(_status_box(pure_text, pure_class), unsafe_allow_html=True)

    if pure is None or hybrid is None:
        hybrid_text = "QoS-Hybrid versus QoS-Pure-LLM could not be compared because one or both averages are unavailable."
        hybrid_class = "live-demo-warning"
    else:
        hybrid_state = _comparison_state(hybrid, pure)
        if hybrid_state == "greater":
            hybrid_text = "QoS-Hybrid improves further as the bonus functional-first QoS refinement."
            hybrid_class = "live-demo-bonus"
        elif hybrid_state == "equal":
            hybrid_text = "QoS-Hybrid ties QoS-Pure-LLM on average for this run."
            hybrid_class = "live-demo-bonus"
        else:
            hybrid_text = (
                "QoS-Hybrid is lower than QoS-Pure-LLM on average for this run "
                f"({_fmt(hybrid, exact=exact)} vs {_fmt(pure, exact=exact)})."
            )
            hybrid_class = "live-demo-warning"
    st.markdown(_status_box(hybrid_text, hybrid_class), unsafe_allow_html=True)


def _aggregate_score_table(score_rows: list[dict[str, Any]], *, exact: bool, compact: bool = False) -> pd.DataFrame:
    rows = []
    for row in score_rows:
        table_row = {
            "Mode": row.get("Mode_Label") or loader.mode_label(str(row.get("Mode") or "")),
            "Average QoS-Adjusted Composition Score": _fmt(
                row.get("Average_QoS_Adjusted_Composition_Score"),
                exact=exact,
            ),
            "Number of queries included": row.get("Number_of_queries_included"),
        }
        if not compact:
            table_row["Missing query count"] = row.get("Missing_query_count")
        rows.append(table_row)
    return pd.DataFrame(rows)


def _aggregate_component_table(component_rows: list[dict[str, Any]], *, exact: bool) -> pd.DataFrame:
    rows = []
    for row in component_rows:
        rows.append(
            {
                "Mode": row.get("Mode_Label") or loader.mode_label(str(row.get("Mode") or "")),
                "Average Functional Coverage": _fmt(row.get("Average_Functional_Coverage"), exact=exact),
                "Average Normalized QoS Score": _fmt(row.get("Average_Normalized_QoS_Score"), exact=exact),
                "Average Total Response Time": _fmt(row.get("Average_Total_Response_Time"), exact=exact),
                "Average Bottleneck Throughput": _fmt(row.get("Average_Bottleneck_Throughput"), exact=exact),
                "Average Workflow Availability": _fmt(row.get("Average_Workflow_Availability"), exact=exact),
            }
        )
    return pd.DataFrame(rows)


def _render_final_average_results(run_dir_text: str, *, exact: bool, defense_core_mode: bool = False) -> None:
    st.header("13. Final Average Results Across All 15 Queries")
    if not defense_core_mode:
        st.write(
            "This aggregate view summarizes the same composition metric across the full official q01–q15 evaluation set. "
            "The selected query above provides a detailed trace, while this chart shows the average QoS-adjusted score across all 15 queries."
        )

    aggregate = loader.load_official_query_aggregate(run_dir_text)
    official_count = int(aggregate.get("official_query_count") or len(loader.OFFICIAL_QUERY_IDS))
    found_count = len(aggregate.get("found_query_ids") or [])
    if found_count == official_count:
        if not defense_core_mode:
            st.success("All 15 official queries included.")
    else:
        st.warning(
            f"Only {found_count} of 15 official queries were found in the selected run folder. "
            "Average is computed over available loaded queries."
        )

    score_rows = list(aggregate.get("score_rows") or [])
    plot_df = pd.DataFrame(score_rows)
    if plot_df.empty or "Average_QoS_Adjusted_Composition_Score" not in plot_df:
        st.warning("Aggregate composition score rows were not loaded for the selected run folder.")
        return
    plot_df["Average_QoS_Adjusted_Composition_Score"] = pd.to_numeric(
        plot_df["Average_QoS_Adjusted_Composition_Score"],
        errors="coerce",
    )
    plot_df = plot_df[plot_df["Average_QoS_Adjusted_Composition_Score"].notna()].copy()
    if plot_df.empty:
        st.warning("No aggregate QoS-adjusted composition scores were available for the expected modes.")
        return

    fig = px.bar(
        plot_df,
        x="Mode_Label",
        y="Average_QoS_Adjusted_Composition_Score",
        color="Mode",
        text="Average_QoS_Adjusted_Composition_Score",
        color_discrete_map=MODE_COLORS,
        category_orders={"Mode_Label": [loader.mode_label(mode) for mode in loader.MODE_ORDER]},
        labels={
            "Mode_Label": "Mode",
            "Average_QoS_Adjusted_Composition_Score": "Average QoS-Adjusted Composition Score",
        },
        hover_data=["Number_of_queries_included", "Missing_query_count"],
    )
    fig.update_traces(texttemplate="%{y:.3f}", textposition="outside", cliponaxis=False)
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=25, b=45),
        showlegend=False,
        yaxis_title="Average QoS-Adjusted Composition Score",
        xaxis_title="Mode",
        yaxis_range=[0, 1.08],
    )
    st.plotly_chart(fig, use_container_width=True)
    if not defense_core_mode:
        st.caption("Average QoS-Adjusted Composition Score across the official q01–q15 query set.")

    if not defense_core_mode:
        _aggregate_conclusion_boxes(score_rows, exact=exact)

    st.dataframe(
        _aggregate_score_table(score_rows, exact=exact, compact=defense_core_mode),
        use_container_width=True,
        hide_index=True,
        height=180,
        column_config=_pinned_columns("Mode"),
    )

    component_rows = list(aggregate.get("component_rows") or [])
    if component_rows:
        label = "Optional: Detailed aggregate component metrics." if defense_core_mode else "Optional aggregate component metrics"
        with st.expander(label, expanded=False):
            st.dataframe(
                _aggregate_component_table(component_rows, exact=exact),
                use_container_width=True,
                hide_index=True,
                height=210,
                column_config=_pinned_columns("Mode"),
            )

    if aggregate.get("warnings"):
        with st.expander("Aggregate loading notes", expanded=False):
            for warning in aggregate["warnings"]:
                st.write(f"- {warning}")


def _render_raw_artifacts(bundle: dict[str, Any], *, defense_core_mode: bool = False) -> None:
    st.header("14. Raw Artifacts")
    if defense_core_mode:
        with st.expander("Optional: Raw artifacts.", expanded=False):
            st.markdown("**artifact_sources**")
            st.json(bundle.get("artifact_sources") or {})
            raw = bundle.get("raw_artifacts") or {}
            preferred = [
                "meta.json",
                "0_decomposer.json",
                "composition_qos_eval_rows",
                "candidate_api_rankings_rows",
                "retrieval_functional_match_rows",
                "planner files by mode",
            ]
            for name in preferred:
                if name in raw:
                    st.markdown(f"**{name}**")
                    st.json(raw[name])
        return

    with st.expander("artifact_sources", expanded=False):
        st.json(bundle.get("artifact_sources") or {})
    raw = bundle.get("raw_artifacts") or {}
    preferred = [
        "meta.json",
        "0_decomposer.json",
        "composition_qos_eval_rows",
        "candidate_api_rankings_rows",
        "retrieval_functional_match_rows",
        "planner files by mode",
    ]
    for name in preferred:
        if name in raw:
            with st.expander(name, expanded=False):
                st.json(raw[name])


def render_live_demo_deep_dive(
    directory_selector: Callable[..., str] | None = None,
    default_run_dir: Path | None = None,
) -> None:
    _inject_css()
    controls = _render_sidebar(directory_selector=directory_selector, default_run_dir=default_run_dir)

    st.title("Live Demo Deep Dive")
    st.caption("Select a run folder to inspect query-level result artifacts or all-query average results.")

    if controls is None:
        st.info("Choose a valid run folder from the sidebar.")
        return

    if controls.get("view_mode") == "All queries average results":
        _render_final_average_results(
            controls["run_dir_text"],
            exact=bool(controls["show_exact"]),
            defense_core_mode=bool(controls.get("defense_core_mode")),
        )
        return

    if not controls.get("query_id"):
        st.info("Choose a valid run folder and query from the sidebar.")
        return

    selected_modes = controls["selected_modes"]
    if not selected_modes:
        st.warning("Select at least one visible mode.")
        return

    bundle = loader.load_live_demo_query(controls["run_dir_text"], controls["query_id"])
    for warning in bundle.get("warnings") or []:
        st.warning(warning)
    if not bundle.get("available"):
        st.error("The selected query could not be loaded from the chosen run folder.")
        return

    selected_modes = _visible_modes(list(bundle.get("available_modes") or []), selected_modes)
    exact = bool(controls["show_exact"])
    top_k = int(controls["top_k"])
    show_all = bool(controls["show_all"])
    defense_core_mode = bool(controls.get("defense_core_mode"))

    eval_df = _mode_df(bundle.get("composition_rows") or [])
    ranking_df = _numeric_frame(pd.DataFrame(bundle.get("ranking_rows") or []))
    retrieval_df = _numeric_frame(pd.DataFrame(bundle.get("retrieval_rows") or []))

    subtask_ids = sorted({str(row.get("subtask_id") or "") for row in bundle.get("subtasks") or [] if row.get("subtask_id")}, key=_subtask_sort_key)
    focused_subtask_id = str(controls["focused_subtask_id"] or (subtask_ids[0] if subtask_ids else "1"))
    if subtask_ids and focused_subtask_id not in subtask_ids:
        focused_subtask_id = subtask_ids[0]

    _render_query_context(bundle)
    _render_subtasks(bundle, retrieval_df, selected_modes, exact=exact)
    _render_retrieval_snapshot(
        retrieval_df,
        top_k,
        show_all=show_all,
        exact=exact,
        defense_core_mode=defense_core_mode,
    )
    focused_subtask_id = _render_candidate_inspector(
        bundle,
        retrieval_df,
        ranking_df,
        focused_subtask_id,
        exact=exact,
        defense_core_mode=defense_core_mode,
    )
    focused_subtask_id = _render_reranking_motivation(
        retrieval_df,
        focused_subtask_id,
        exact=exact,
        query_id=str(bundle.get("query_id") or controls["query_id"]),
        defense_core_mode=defense_core_mode,
    )
    _render_ranking_by_mode(
        ranking_df,
        selected_modes,
        top_k,
        show_all=show_all,
        exact=exact,
        defense_core_mode=defense_core_mode,
    )
    _render_ranking_difference(
        retrieval_df,
        ranking_df,
        bundle.get("selected_paths") or {},
        selected_modes,
        focused_subtask_id,
        top_k,
        defense_core_mode=defense_core_mode,
    )
    _render_selected_paths(bundle, selected_modes, eval_df, exact=exact, defense_core_mode=defense_core_mode)

    if eval_df.empty:
        st.error("Composition evaluation rows were not loaded, so score comparisons cannot be shown.")
    else:
        _render_score_comparison(eval_df, selected_modes, exact=exact)
        _render_formula_proof(eval_df, bundle.get("selected_paths") or {}, selected_modes, exact=exact)
        _render_hypothesis_cards(eval_df, exact=exact)
        if defense_core_mode:
            _render_defense_takeaway(eval_df)
        else:
            _render_takeaway(eval_df, exact=exact)

    if defense_core_mode:
        _render_final_average_results(controls["run_dir_text"], exact=exact, defense_core_mode=True)

    if controls["show_raw"]:
        _render_raw_artifacts(bundle, defense_core_mode=defense_core_mode)


def main() -> None:
    st.set_page_config(page_title="Live Demo Deep Dive", layout="wide")
    render_live_demo_deep_dive()


if __name__ == "__main__":
    main()
