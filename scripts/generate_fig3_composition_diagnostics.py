#!/usr/bin/env python3
"""Generate publication-ready Figure 3 composition diagnostics.

The script discovers workflow-level metrics and ranking diagnostics under a
run directory, writes auditable source tables, and saves an IEEE-style figure
as PDF, PNG, and SVG.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Avoid noisy optional binary extensions in environments where pandas is usable
# but numexpr/bottleneck were compiled against an older NumPy ABI.
sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)

import numpy as np
import pandas as pd


SCRIPT_NAME = Path(__file__).name
DEFAULT_RUN_DIR = Path("results/logs/RUNS_MAY_31_NEW_5/fireworks_gpt-oss-120b")
DEFAULT_OUT_DIR = DEFAULT_RUN_DIR / "figures" / "paper"
SUPPORTED_EXTENSIONS = {".csv", ".json", ".jsonl", ".parquet"}
EXPECTED_QUERY_COUNT = 15
EXPECTED_MODE_COUNT = 4
EXPECTED_CASE_COUNT = 45
MODE_ORDER = ("no_qos", "qos_pure_llm", "qos_topsis", "qos_hybrid")
MODE_LABELS = {
    "no_qos": "No-QoS",
    "qos_pure_llm": "QoS-Pure-LLM",
    "qos_topsis": "QoS-TOPSIS",
    "qos_hybrid": "QoS-Hybrid",
}
MODE_TICK_LABELS = {
    "no_qos": "No-\nQoS",
    "qos_pure_llm": "QoS-\nPure-LLM",
    "qos_topsis": "QoS-\nTOPSIS",
    "qos_hybrid": "QoS-\nHybrid",
}
MODE_ALIASES = {
    "baseline": "no_qos",
    "functional_only": "no_qos",
    "no_qos": "no_qos",
    "noqos": "no_qos",
    "no_qos_baseline": "no_qos",
    "pure_llm": "qos_pure_llm",
    "qos_llm": "qos_pure_llm",
    "qos_pure_llm": "qos_pure_llm",
    "qospurellm": "qos_pure_llm",
    "topsis": "qos_topsis",
    "qos_topsis": "qos_topsis",
    "qostopsis": "qos_topsis",
    "functional_topsis": "qos_hybrid",
    "hybrid": "qos_hybrid",
    "qos_hybrid": "qos_hybrid",
    "qoshybrid": "qos_hybrid",
}
QUERY_ALIASES = ("query_id", "qid", "query", "query_name", "query_idx", "query_number")
MODE_ALIASES_COLUMNS = ("mode", "ranking_mode", "selection_mode", "method", "strategy")
QACS_ALIASES = (
    "qacs",
    "qos_adjusted_composition_score",
    "final_score",
    "composition_score",
    "score",
)
FC_ALIASES = ("fc", "functional_coverage", "function_coverage")
NQS_ALIASES = ("nqs", "normalized_qos_score", "normalized_quality_score", "qos_score")
COMPLETENESS_ALIASES = ("completeness", "composition_completeness", "complete")
SUBTASK_ALIASES = ("subtask_id", "sub_task", "sub_task_id", "step_id", "step", "subtask", "subtask_index", "workflow_step")
RANK_ALIASES = (
    "mode_rank",
    "rank",
    "ranking_position",
    "position",
    "selected_rank",
    "top_rank",
)
SELECTED_ALIASES = ("selected", "is_selected", "chosen", "is_chosen", "planner_selected", "selected_for_planner")
API_ALIASES = ("api_id", "endpoint_id", "candidate_id", "selected_api_id", "api_name", "selected_api")
FUNCTIONAL_ALIASES = (
    "functional_match",
    "functional_match_0_1",
    "match_label",
    "functional_label",
    "is_functional_match",
    "is_match",
    "label",
    "llm_label",
    "functionally_matching",
)
TOPSIS_ALIASES = ("topsis_score", "closeness", "closeness_score", "ci", "qos_rank_score")
QUERY_RE = re.compile(r"q?(\d{1,3})", re.IGNORECASE)
QUERY_DIR_RE = re.compile(r"(q\d{2})(?:_|$)", re.IGNORECASE)
SUBTASK_RE = re.compile(r"(?:^|[_\-/])s(?:ubtask)?_?(\d+)(?:\D|$)", re.IGNORECASE)


@dataclass(frozen=True)
class TableCandidate:
    path: Path
    rows_loaded: int
    score: float
    query_count: int
    mode_count: int
    case_count: int
    data: pd.DataFrame


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Figure 3 composition diagnostics from an AutoLLMCompose run directory."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="Input run directory.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for figure files.")
    inset_group = parser.add_mutually_exclusive_group()
    inset_group.add_argument("--include-inset", dest="include_inset", action="store_true", help="Include inset if data exists.")
    inset_group.add_argument("--no-inset", dest="include_inset", action="store_false", help="Omit representative path inset.")
    parser.set_defaults(include_inset=True)
    parser.add_argument(
        "--allow-missing-ranking",
        action="store_true",
        help="Generate Panel A and aggregate metrics if ranking diagnostics cannot be computed.",
    )
    parser.add_argument("--debug", action="store_true", help="Print additional discovery details.")
    return parser.parse_args()


def normalize_key(value: Any) -> str:
    """Normalize column names and aliases to lower snake case."""
    text = str(value).strip()
    for acronym in ("QoS", "API", "LLM", "QACS", "NQS"):
        text = re.sub(acronym, acronym.upper(), text, flags=re.IGNORECASE)
    text = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text.lower()).strip("_")
    return text


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with lower-snake-case, unique column names."""
    used: dict[str, int] = {}
    columns: list[str] = []
    for column in df.columns:
        base = normalize_key(column) or "column"
        count = used.get(base, 0)
        used[base] = count + 1
        columns.append(base if count == 0 else f"{base}_{count + 1}")
    out = df.copy()
    out.columns = columns
    return out


def first_existing(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    """Return the first alias present in a collection of columns."""
    available = set(columns)
    for alias in aliases:
        if alias in available:
            return alias
    return None


def canonicalize_mode(value: Any) -> str | None:
    """Map common mode-name variants to canonical mode keys."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = normalize_key(value)
    compact = text.replace("_", "")
    if text in MODE_ALIASES:
        return MODE_ALIASES[text]
    if compact in MODE_ALIASES:
        return MODE_ALIASES[compact]
    return text if text in MODE_ORDER else None


def normalize_query_id(value: Any) -> str | None:
    """Normalize query identifiers to qXX."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    full = re.fullmatch(r"q?(\d{1,3})", text, re.IGNORECASE)
    if not full:
        return None
    number = int(full.group(1))
    return f"q{number:02d}" if number > 0 else None


def natural_query_sort(query_id: Any) -> tuple[int, str]:
    """Sort query ids naturally: q01, q02, ..., q15."""
    normalized = normalize_query_id(query_id)
    if normalized:
        return (int(normalized[1:]), normalized)
    return (10_000, str(query_id))


def normalize_subtask_id(value: Any) -> str | None:
    """Normalize subtask identifiers to sN."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"(\d+)", text)
    if not match:
        return normalize_key(text) or None
    number = int(match.group(1))
    return f"s{number}" if number > 0 else None


def subtask_sort_key(subtask_id: Any) -> tuple[int, str]:
    """Sort subtask ids naturally."""
    normalized = normalize_subtask_id(subtask_id)
    if normalized:
        return (int(normalized[1:]), normalized)
    return (10_000, str(subtask_id))


def infer_query_from_path(path: Path) -> str | None:
    """Infer qXX from a query run folder or filename."""
    for part in path.parts:
        match = QUERY_DIR_RE.search(part)
        if match:
            return match.group(1).lower()
    match = QUERY_RE.search(path.name)
    if match:
        return normalize_query_id(match.group(1))
    return None


def infer_mode_from_path(path: Path) -> str | None:
    """Infer canonical mode from path components."""
    for part in reversed(path.parts):
        mode = canonicalize_mode(part)
        if mode:
            return mode
    return None


def infer_subtask_from_path(path: Path) -> str | None:
    """Infer subtask id from filenames such as 2_ranked_s1.json."""
    text = path.as_posix()
    match = SUBTASK_RE.search(text)
    if match:
        return normalize_subtask_id(match.group(1))
    return None


def sha256_file(path: Path) -> str:
    """Compute SHA256 for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_records(payload: Any) -> list[Any]:
    """Convert supported JSON payload shapes into records for DataFrame creation."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("rows", "records", "data", "results", "items", "evaluations"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return [payload]
    return [{"value": payload}]


def load_table(path: Path) -> pd.DataFrame:
    """Load .csv, .json, .jsonl, or .parquet into a DataFrame."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return pd.json_normalize(as_records(payload), sep="_")
    if suffix == ".jsonl":
        rows: list[Any] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                rows.extend(as_records(payload))
        return pd.json_normalize(rows, sep="_")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file extension: {path.suffix}")


def safe_to_numeric(series: pd.Series | None) -> pd.Series:
    """Convert a series to numeric values, preserving missing values."""
    if series is None:
        return pd.Series(dtype="float64")
    return pd.to_numeric(series, errors="coerce")


def normalize_truthy(value: Any) -> bool | None:
    """Normalize common truthy/falsey values."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(int(value))
    text = normalize_key(value)
    if text in {"1", "true", "yes", "y", "selected", "chosen", "match", "positive"}:
        return True
    if text in {"0", "false", "no", "n", "not_selected", "not_chosen", "non_match", "negative"}:
        return False
    return None


def normalize_functional_match(value: Any) -> float:
    """Normalize functional-match labels to 1.0, 0.0, or NaN."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        return 1.0 if int(value) != 0 else 0.0
    text = normalize_key(value)
    false_values = {"0", "false", "no", "n", "non_match", "nonmatching", "not_match", "negative", "not_functional"}
    true_values = {"1", "true", "yes", "y", "match", "matching", "positive", "functional", "functionally_matching"}
    if text in false_values:
        return 0.0
    if text in true_values:
        return 1.0
    return np.nan


def is_relative_to(path: Path, parent: Path) -> bool:
    """Compatibility helper for Path.is_relative_to."""
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def discover_files(run_dir: Path, out_dir: Path) -> list[Path]:
    """Find candidate data files while excluding generated figure outputs."""
    files: list[Path] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if is_relative_to(path, out_dir):
            continue
        if "figures" in {part.lower() for part in path.parts}:
            continue
        files.append(path)
    return files


def standardize_metrics_table(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Extract workflow metric columns from a loaded table."""
    if df.empty:
        return pd.DataFrame()
    normalized = normalize_columns(df)
    q_col = first_existing(normalized.columns, QUERY_ALIASES)
    mode_col = first_existing(normalized.columns, MODE_ALIASES_COLUMNS)
    qacs_col = first_existing(normalized.columns, QACS_ALIASES)
    fc_col = first_existing(normalized.columns, FC_ALIASES)
    nqs_col = first_existing(normalized.columns, NQS_ALIASES)
    complete_col = first_existing(normalized.columns, COMPLETENESS_ALIASES)
    if not all((qacs_col, fc_col, nqs_col)):
        return pd.DataFrame()

    query_values = normalized[q_col].map(normalize_query_id) if q_col else pd.Series([infer_query_from_path(path)] * len(normalized))
    mode_values = normalized[mode_col].map(canonicalize_mode) if mode_col else pd.Series([infer_mode_from_path(path)] * len(normalized))
    out = pd.DataFrame(
        {
            "query_id": query_values,
            "mode": mode_values,
            "qacs": safe_to_numeric(normalized[qacs_col]),
            "fc": safe_to_numeric(normalized[fc_col]),
            "nqs": safe_to_numeric(normalized[nqs_col]),
            "source_path": path.as_posix(),
        }
    )
    if complete_col:
        out["completeness"] = safe_to_numeric(normalized[complete_col])
    else:
        out["completeness"] = np.nan
    out = out.dropna(subset=["query_id", "mode", "qacs", "fc", "nqs"])
    out = out[out["mode"].isin(MODE_ORDER)]
    return out


def score_metric_candidate(path: Path, data: pd.DataFrame, rows_loaded: int) -> TableCandidate:
    """Score workflow metric candidates for deterministic selection."""
    pair_count = data.drop_duplicates(["query_id", "mode"]).shape[0]
    query_count = data["query_id"].nunique()
    mode_count = data["mode"].nunique()
    lower = path.as_posix().lower()
    keywords = ("workflow", "evaluation", "metrics", "summary", "scores", "results")
    score = 100.0 + min(query_count, EXPECTED_QUERY_COUNT) * 4 + min(mode_count, EXPECTED_MODE_COUNT) * 8
    if pair_count == EXPECTED_QUERY_COUNT * EXPECTED_MODE_COUNT:
        score += 100
    elif pair_count >= EXPECTED_QUERY_COUNT * EXPECTED_MODE_COUNT * 0.8:
        score += 60
    elif mode_count == EXPECTED_MODE_COUNT:
        score += 20
    if any(keyword in lower for keyword in keywords):
        score += 20
    if "/summary/" in lower:
        score += 20
    score -= abs(pair_count - EXPECTED_QUERY_COUNT * EXPECTED_MODE_COUNT) * 0.2
    return TableCandidate(path, rows_loaded, score, query_count, mode_count, pair_count, data)


def find_workflow_metrics(files: list[Path], warnings: list[str], debug: bool = False) -> tuple[pd.DataFrame, list[Path], list[dict[str, Any]]]:
    """Discover and select workflow-level QACS/FC/NQS data."""
    candidates: list[TableCandidate] = []
    for path in files:
        try:
            raw = load_table(path)
        except Exception as exc:
            if debug:
                warnings.append(f"Could not load {path}: {exc}")
            continue
        data = standardize_metrics_table(raw, path)
        if data.empty:
            continue
        candidates.append(score_metric_candidate(path, data, len(raw)))

    if not candidates:
        raise RuntimeError("No workflow metrics table with query_id, mode, qacs, fc, and nqs was found.")

    candidates = sorted(candidates, key=lambda c: (-c.score, -c.case_count, c.path.as_posix()))
    best = candidates[0]
    data = best.data.copy()
    selected_paths = [best.path]
    if best.case_count < EXPECTED_QUERY_COUNT * EXPECTED_MODE_COUNT:
        combined_parts: list[pd.DataFrame] = []
        seen_pairs: set[tuple[str, str]] = set()
        selected_paths = []
        for candidate in candidates:
            part = candidate.data.sort_values(["query_id", "mode"]).copy()
            new_part = part[~part[["query_id", "mode"]].apply(tuple, axis=1).isin(seen_pairs)]
            if new_part.empty:
                continue
            combined_parts.append(new_part)
            selected_paths.append(candidate.path)
            seen_pairs.update(new_part[["query_id", "mode"]].apply(tuple, axis=1))
        data = pd.concat(combined_parts, ignore_index=True) if combined_parts else best.data.copy()
        warnings.append("Workflow metrics were assembled from multiple candidate files.")

    data["mode_order"] = data["mode"].map({mode: idx for idx, mode in enumerate(MODE_ORDER)})
    data = data.sort_values(["query_id", "mode_order"], key=lambda s: s.map(natural_query_sort) if s.name == "query_id" else s)
    data = data.drop_duplicates(["query_id", "mode"], keep="first").drop(columns=["mode_order"])

    modes = set(data["mode"])
    missing_modes = [mode for mode in MODE_ORDER if mode not in modes]
    if missing_modes:
        raise RuntimeError(f"Fewer than four canonical modes detected for Panel A; missing: {', '.join(missing_modes)}")

    query_count = data["query_id"].nunique()
    if query_count != EXPECTED_QUERY_COUNT:
        warnings.append(f"Expected {EXPECTED_QUERY_COUNT} queries, detected {query_count}.")

    manifest_candidates = [
        {
            "path": c.path.as_posix(),
            "rows_loaded": c.rows_loaded,
            "score": round(c.score, 3),
            "query_count": c.query_count,
            "mode_count": c.mode_count,
            "query_mode_rows": c.case_count,
        }
        for c in candidates
    ]
    return data, selected_paths, manifest_candidates


def standardize_ranking_table(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Extract ranking/selection diagnostic columns from a loaded table."""
    if df.empty:
        return pd.DataFrame()
    normalized = normalize_columns(df)
    q_col = first_existing(normalized.columns, QUERY_ALIASES)
    mode_col = first_existing(normalized.columns, MODE_ALIASES_COLUMNS)
    subtask_col = first_existing(normalized.columns, SUBTASK_ALIASES)
    rank_col = first_existing(normalized.columns, RANK_ALIASES)
    selected_col = first_existing(normalized.columns, SELECTED_ALIASES)
    functional_col = first_existing(normalized.columns, FUNCTIONAL_ALIASES)
    topsis_col = first_existing(normalized.columns, TOPSIS_ALIASES)

    if not (rank_col or selected_col):
        return pd.DataFrame()

    query_values = normalized[q_col].map(normalize_query_id) if q_col else pd.Series([infer_query_from_path(path)] * len(normalized))
    mode_values = normalized[mode_col].map(canonicalize_mode) if mode_col else pd.Series([infer_mode_from_path(path)] * len(normalized))
    subtask_values = normalized[subtask_col].map(normalize_subtask_id) if subtask_col else pd.Series([infer_subtask_from_path(path)] * len(normalized))
    out = pd.DataFrame(
        {
            "query_id": query_values,
            "subtask_id": subtask_values,
            "mode": mode_values,
            "rank": safe_to_numeric(normalized[rank_col]) if rank_col else np.nan,
            "selected": normalized[selected_col].map(normalize_truthy) if selected_col else False,
            "functional_match": normalized[functional_col].map(normalize_functional_match) if functional_col else np.nan,
            "topsis_score": safe_to_numeric(normalized[topsis_col]) if topsis_col else np.nan,
            "source_path": path.as_posix(),
        }
    )
    for alias in API_ALIASES:
        if alias in normalized.columns:
            out[alias] = normalized[alias]
    label_columns = (
        "tool_name",
        "endpoint_name",
        "service_tool_name",
        "service_toolbench_tool_name",
        "service_toolbench_enrichment_tool_name",
        "service_name",
        "service_toolbench_enrichment_endpoint_name",
        "service_toolbench_endpoint_description",
        "name",
    )
    for column in label_columns:
        if column in normalized.columns:
            out[column] = normalized[column]
    out = out.dropna(subset=["query_id", "subtask_id", "mode"])
    out = out[out["mode"].isin(MODE_ORDER)]
    return out


def score_ranking_candidate(path: Path, data: pd.DataFrame, rows_loaded: int) -> TableCandidate:
    """Score ranking candidates for deterministic selection."""
    group_count = data.drop_duplicates(["query_id", "subtask_id", "mode"]).shape[0]
    query_count = data["query_id"].nunique()
    mode_count = data["mode"].nunique()
    case_count = data.drop_duplicates(["query_id", "subtask_id"]).shape[0]
    lower = path.as_posix().lower()
    functional_rate = float(data["functional_match"].notna().mean()) if "functional_match" in data else 0.0
    score = 50.0 + min(query_count, EXPECTED_QUERY_COUNT) * 3 + min(mode_count, EXPECTED_MODE_COUNT) * 10
    score += min(case_count, EXPECTED_CASE_COUNT) * 1.5
    score += functional_rate * 80
    if data["rank"].notna().any():
        score += 40
    if "candidate_api_rankings_rows" in lower:
        score += 120
    elif "ranking" in lower or "ranked" in lower:
        score += 40
    if "evaluation" in lower:
        score += 20
    return TableCandidate(path, rows_loaded, score, query_count, mode_count, group_count, data)


def select_top1_rows(ranking: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    """Select one top-ranked/planner-facing row per query-subtask-mode."""
    data = ranking.copy()
    if data.empty:
        return data
    if data["rank"].notna().any():
        ranked = data[data["rank"].notna()].copy()
        min_rank = ranked.groupby(["query_id", "subtask_id", "mode"])["rank"].transform("min")
        top = ranked[ranked["rank"] == min_rank].copy()
    elif "selected" in data:
        top = data[data["selected"] == True].copy()  # noqa: E712
    else:
        return pd.DataFrame()

    top["mode_order"] = top["mode"].map({mode: idx for idx, mode in enumerate(MODE_ORDER)})
    for column in ("api_id", "selected_api", "candidate_id", "api_name"):
        if column not in top.columns:
            top[column] = ""
    sort_columns = ["query_id", "subtask_id", "mode_order", "selected", "rank", "api_id", "selected_api", "candidate_id", "source_path"]
    top = top.sort_values(sort_columns, ascending=[True, True, True, False, True, True, True, True, True])
    duplicate_mask = top.duplicated(["query_id", "subtask_id", "mode"], keep=False)
    if duplicate_mask.any():
        duplicate_groups = top.loc[duplicate_mask, ["query_id", "subtask_id", "mode"]].drop_duplicates().shape[0]
        warnings.append(f"Ranking top-1 selection had {duplicate_groups} duplicate groups; kept the deterministic first row.")
    top = top.drop_duplicates(["query_id", "subtask_id", "mode"], keep="first")
    return top.drop(columns=["mode_order"])


def api_lookup_key(row: pd.Series) -> str:
    """Build a generic API identifier key for joining selected rows to labels."""
    return first_text(row, API_ALIASES)


def fill_missing_functional_labels(top1: pd.DataFrame, candidates: list[TableCandidate], warnings: list[str]) -> pd.DataFrame:
    """Join functional labels from candidate/label tables when selected rows omit them."""
    if top1.empty or "functional_match" not in top1 or not top1["functional_match"].isna().any():
        return top1
    label_frames = [candidate.data for candidate in candidates if "functional_match" in candidate.data and candidate.data["functional_match"].notna().any()]
    if not label_frames:
        return top1

    labels = pd.concat(label_frames, ignore_index=True)
    labels = labels[labels["functional_match"].notna()].copy()
    labels["api_lookup_key"] = labels.apply(api_lookup_key, axis=1)
    labels = labels[labels["api_lookup_key"] != ""]
    if labels.empty:
        return top1

    labels = labels.sort_values(["query_id", "subtask_id", "mode", "api_lookup_key", "rank", "source_path"])
    lookup = (
        labels.drop_duplicates(["query_id", "subtask_id", "mode", "api_lookup_key"], keep="first")
        .set_index(["query_id", "subtask_id", "mode", "api_lookup_key"])["functional_match"]
        .to_dict()
    )
    out = top1.copy()
    out["api_lookup_key"] = out.apply(api_lookup_key, axis=1)
    missing_mask = out["functional_match"].isna()
    filled = 0
    for idx, row in out.loc[missing_mask].iterrows():
        key = (row["query_id"], row["subtask_id"], row["mode"], row["api_lookup_key"])
        if key in lookup:
            out.at[idx, "functional_match"] = lookup[key]
            filled += 1
    if filled:
        warnings.append(f"Filled {filled} missing selected-row functional labels from candidate/label tables.")
    return out.drop(columns=["api_lookup_key"])


def find_ranking_diagnostics(
    files: list[Path],
    warnings: list[str],
    allow_missing: bool,
    debug: bool = False,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, list[Path], list[dict[str, Any]]]:
    """Discover ranking diagnostics and compute top-1 rows."""
    candidates: list[TableCandidate] = []
    for path in files:
        try:
            raw = load_table(path)
        except Exception as exc:
            if debug:
                warnings.append(f"Could not load {path}: {exc}")
            continue
        data = standardize_ranking_table(raw, path)
        if data.empty:
            continue
        if not (data["rank"].notna().any() or data["selected"].fillna(False).any()):
            continue
        candidates.append(score_ranking_candidate(path, data, len(raw)))

    manifest_candidates = [
        {
            "path": c.path.as_posix(),
            "rows_loaded": c.rows_loaded,
            "score": round(c.score, 3),
            "query_count": c.query_count,
            "mode_count": c.mode_count,
            "query_subtask_mode_rows": c.case_count,
        }
        for c in sorted(candidates, key=lambda c: (-c.score, c.path.as_posix()))
    ]
    if not candidates:
        message = "No ranking/selection diagnostics table was found."
        if allow_missing:
            warnings.append(message)
            return None, None, [], manifest_candidates
        raise RuntimeError(message)

    preferred = [c for c in candidates if "candidate_api_rankings_rows" in c.path.name.lower() and c.data["functional_match"].notna().any()]
    if preferred:
        selected_candidates = sorted(preferred, key=lambda c: c.path.as_posix())
    else:
        functional = [c for c in candidates if c.data["functional_match"].notna().any()]
        selected_candidates = sorted(functional or candidates, key=lambda c: (-c.score, c.path.as_posix()))
        if selected_candidates:
            best_score = selected_candidates[0].score
            selected_candidates = [c for c in selected_candidates if c.score >= best_score - 5]

    combined = pd.concat([candidate.data for candidate in selected_candidates], ignore_index=True)
    top1 = select_top1_rows(combined, warnings)
    top1 = fill_missing_functional_labels(top1, candidates, warnings)
    if top1.empty or top1["functional_match"].isna().all():
        message = "Top-1 functional match could not be computed from the selected ranking diagnostics."
        if allow_missing:
            warnings.append(message)
            return None, None, [c.path for c in selected_candidates], manifest_candidates
        raise RuntimeError(message)
    if top1["functional_match"].isna().any():
        missing = int(top1["functional_match"].isna().sum())
        message = f"Top-1 diagnostics contain {missing} rows without functional_match labels."
        if allow_missing:
            warnings.append(message)
            top1 = top1.dropna(subset=["functional_match"])
        else:
            raise RuntimeError(message)

    top1["functional_match"] = top1["functional_match"].astype(int)
    case_count = top1.drop_duplicates(["query_id", "subtask_id"]).shape[0]
    if case_count != EXPECTED_CASE_COUNT:
        warnings.append(f"Expected {EXPECTED_CASE_COUNT} query-subtask cases, detected {case_count}.")
    for mode in MODE_ORDER:
        denominator = top1[top1["mode"] == mode].drop_duplicates(["query_id", "subtask_id"]).shape[0]
        if denominator != case_count:
            warnings.append(f"{MODE_LABELS[mode]} has denominator {denominator}, overall detected case count is {case_count}.")
    return combined, top1, [c.path for c in selected_candidates], manifest_candidates


def find_selected_paths(files: list[Path], debug: bool = False) -> tuple[pd.DataFrame, list[Path]]:
    """Load selected API rows for optional representative workflow labels."""
    frames: list[pd.DataFrame] = []
    paths: list[Path] = []
    for path in files:
        lower = path.name.lower()
        if not re.search(r"3_selected_s\d+\.json$", lower):
            continue
        try:
            raw = load_table(path)
        except Exception:
            if debug:
                print(f"Skipping unreadable selected path file: {path}", file=sys.stderr)
            continue
        data = standardize_ranking_table(raw, path)
        if data.empty:
            continue
        frames.append(data)
        paths.append(path)
    if not frames:
        return pd.DataFrame(), []
    selected = pd.concat(frames, ignore_index=True)
    selected["mode_order"] = selected["mode"].map({mode: idx for idx, mode in enumerate(MODE_ORDER)})
    selected = selected.sort_values(["query_id", "subtask_id", "mode_order", "rank", "source_path"])
    selected = selected.drop(columns=["mode_order"])
    return selected, paths


def compute_panel_a(metrics: pd.DataFrame) -> pd.DataFrame:
    """Prepare query-mode source rows and best/tied-best markers for Panel A."""
    rows = metrics.copy()
    max_by_query = rows.groupby("query_id")["qacs"].transform("max")
    rows["is_best"] = np.isclose(rows["qacs"], max_by_query, rtol=0.0, atol=1e-9)
    best_counts = rows.groupby("query_id")["is_best"].transform("sum")
    rows["best_type"] = np.where(rows["is_best"], np.where(best_counts > 1, "tied_best", "unique_best"), "not_best")
    rows["is_best"] = rows["is_best"].astype(int)
    rows["mode_order"] = rows["mode"].map({mode: idx for idx, mode in enumerate(MODE_ORDER)})
    rows = rows.sort_values(["query_id", "mode_order"], key=lambda s: s.map(natural_query_sort) if s.name == "query_id" else s)
    return rows[["query_id", "mode", "qacs", "fc", "nqs", "is_best", "best_type"]]


def compute_aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate workflow metrics by mode."""
    rows = (
        metrics.groupby("mode", observed=True)
        .agg(
            mean_qacs=("qacs", "mean"),
            std_qacs=("qacs", lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0),
            mean_fc=("fc", "mean"),
            mean_nqs=("nqs", "mean"),
            n_queries=("query_id", "nunique"),
        )
        .reset_index()
    )
    rows["mode_order"] = rows["mode"].map({mode: idx for idx, mode in enumerate(MODE_ORDER)})
    return rows.sort_values("mode_order").drop(columns=["mode_order"])


def compute_top1_summary(top1: pd.DataFrame | None) -> pd.DataFrame | None:
    """Compute top-1 functional-match counts by mode."""
    if top1 is None or top1.empty:
        return None
    rows: list[dict[str, Any]] = []
    for mode in MODE_ORDER:
        subset = top1[top1["mode"] == mode].drop_duplicates(["query_id", "subtask_id", "mode"])
        denominator = int(subset.shape[0])
        count = int(subset["functional_match"].sum())
        rows.append(
            {
                "mode": mode,
                "top1_functional_match_count": count,
                "denominator": denominator,
                "top1_functional_match_rate": count / denominator if denominator else np.nan,
            }
        )
    return pd.DataFrame(rows)


def first_text(row: pd.Series, columns: Iterable[str]) -> str:
    """Return the first non-empty text value from a row."""
    for column in columns:
        if column not in row.index:
            continue
        value = row[column]
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return ""


def compact_api_label(row: pd.Series) -> str:
    """Create a concise selected API label from service/API fields."""
    tool = first_text(row, ("tool_name", "service_tool_name", "service_toolbench_tool_name", "service_toolbench_enrichment_tool_name"))
    endpoint = first_text(row, ("endpoint_name", "service_name", "service_toolbench_enrichment_endpoint_name", "name"))
    if tool and endpoint:
        if endpoint.lower() in tool.lower():
            return tool
        return f"{tool}: {endpoint}"
    return first_text(row, ("selected_api", "api_name", "selected_api_id", "api_id", "candidate_id")) or "unknown"


def truncate_label(value: str, limit: int = 22) -> str:
    """Truncate labels deterministically with ASCII ellipsis."""
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def choose_representative_query(metrics: pd.DataFrame, top1: pd.DataFrame) -> str | None:
    """Choose a deterministic representative query for the workflow-path inset."""
    if top1.empty:
        return None
    subtask_counts = top1.drop_duplicates(["query_id", "subtask_id"]).groupby("query_id")["subtask_id"].count()
    candidates = sorted(subtask_counts[subtask_counts == 3].index.tolist(), key=natural_query_sort)
    if not candidates:
        candidates = sorted(subtask_counts.index.tolist(), key=natural_query_sort)
    pivot = metrics.pivot_table(index="query_id", columns="mode", values="qacs", aggfunc="first")
    scored: list[tuple[float, str]] = []
    for query_id in candidates:
        if query_id not in pivot.index or "qos_hybrid" not in pivot.columns:
            continue
        hybrid = pivot.loc[query_id, "qos_hybrid"]
        others = [pivot.loc[query_id, mode] for mode in MODE_ORDER if mode != "qos_hybrid" and mode in pivot.columns]
        if pd.isna(hybrid) or not others:
            continue
        gap = float(hybrid - max(others))
        if gap > 0:
            scored.append((gap, query_id))
    if scored:
        return sorted(scored, key=lambda item: (-item[0], natural_query_sort(item[1])))[0][1]

    drops: list[tuple[float, str]] = []
    for query_id in candidates:
        if query_id in pivot.index and {"qos_hybrid", "qos_topsis"}.issubset(set(pivot.columns)):
            drop = float(pivot.loc[query_id, "qos_hybrid"] - pivot.loc[query_id, "qos_topsis"])
            drops.append((drop, query_id))
    if drops:
        return sorted(drops, key=lambda item: (-item[0], natural_query_sort(item[1])))[0][1]
    return candidates[0] if candidates else None


def build_inset_source(metrics: pd.DataFrame, top1: pd.DataFrame | None, selected: pd.DataFrame) -> pd.DataFrame | None:
    """Build representative selected-path data for Panel C/inset."""
    if top1 is None or top1.empty:
        return None
    query_id = choose_representative_query(metrics, top1)
    if not query_id:
        return None
    top_subset = top1[top1["query_id"] == query_id].copy()
    if top_subset.empty:
        return None

    selected_subset = selected[selected["query_id"] == query_id].copy() if not selected.empty else pd.DataFrame()
    selected_lookup: dict[tuple[str, str, str], pd.Series] = {}
    if not selected_subset.empty:
        selected_subset = selected_subset.sort_values(["query_id", "subtask_id", "mode", "rank", "source_path"])
        for _, row in selected_subset.iterrows():
            key = (str(row["query_id"]), str(row["subtask_id"]), str(row["mode"]))
            selected_lookup.setdefault(key, row)

    rows: list[dict[str, Any]] = []
    subtasks = sorted(top_subset["subtask_id"].dropna().unique().tolist(), key=subtask_sort_key)
    for mode in MODE_ORDER:
        for subtask_id in subtasks:
            match = top_subset[(top_subset["mode"] == mode) & (top_subset["subtask_id"] == subtask_id)]
            if match.empty:
                continue
            top_row = match.sort_values(["rank", "source_path"]).iloc[0]
            key = (query_id, subtask_id, mode)
            selected_row = selected_lookup.get(key)
            label_row = selected_row if selected_row is not None else top_row
            topsis_value = np.nan
            if selected_row is not None and "topsis_score" in selected_row.index:
                topsis_value = selected_row["topsis_score"]
            if pd.isna(topsis_value) and "topsis_score" in top_row.index:
                topsis_value = top_row["topsis_score"]
            rows.append(
                {
                    "query_id": query_id,
                    "subtask_id": subtask_id,
                    "mode": mode,
                    "selected_api_label": compact_api_label(label_row),
                    "selected_rank": int(top_row["rank"]) if not pd.isna(top_row["rank"]) else "",
                    "functional_match": int(top_row["functional_match"]),
                    "topsis_score": float(topsis_value) if not pd.isna(topsis_value) else np.nan,
                }
            )
    if not rows:
        return None
    out = pd.DataFrame(rows)
    out["mode_order"] = out["mode"].map({mode: idx for idx, mode in enumerate(MODE_ORDER)})
    out = out.sort_values(["subtask_id", "mode_order"], key=lambda s: s.map(subtask_sort_key) if s.name == "subtask_id" else s)
    return out.drop(columns=["mode_order"])


def configure_matplotlib() -> None:
    """Configure deterministic, headless matplotlib output."""
    cache_root = Path(tempfile.gettempdir()) / "autollmcompose_fig3"
    mpl_dir = cache_root / "matplotlib"
    xdg_dir = cache_root / "xdg"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    xdg_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_dir.as_posix())
    os.environ.setdefault("XDG_CACHE_HOME", xdg_dir.as_posix())


def load_pyplot():
    """Import and configure matplotlib.pyplot."""
    configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    try:
        font_manager.findfont("Times New Roman", fallback_to_default=False)
        serif_font = "Times New Roman"
    except Exception:
        serif_font = "DejaVu Serif"

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [serif_font, "DejaVu Serif"],
            "font.size": 7.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "fig3_composition_diagnostics",
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
        }
    )
    return plt


def plot_panel_a(ax: Any, fig: Any, panel_a: pd.DataFrame, plt: Any) -> None:
    """Draw Panel A query-level QACS heatmap."""
    queries = sorted(panel_a["query_id"].unique(), key=natural_query_sort)
    matrix = np.full((len(queries), len(MODE_ORDER)), np.nan)
    best_types: dict[tuple[int, int], str] = {}
    for i, query_id in enumerate(queries):
        for j, mode in enumerate(MODE_ORDER):
            row = panel_a[(panel_a["query_id"] == query_id) & (panel_a["mode"] == mode)]
            if row.empty:
                continue
            matrix[i, j] = float(row.iloc[0]["qacs"])
            best_types[(i, j)] = str(row.iloc[0]["best_type"])

    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#f5f5f5")
    image = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest")
    ax.set_title("(a) Query-level QACS", loc="left", fontweight="bold", pad=5)
    ax.set_ylabel("Query")
    ax.set_xticks(np.arange(len(MODE_ORDER)))
    ax.set_xticklabels([MODE_TICK_LABELS[mode] for mode in MODE_ORDER])
    ax.set_yticks(np.arange(len(queries)))
    ax.set_yticklabels(queries)
    ax.set_xticks(np.arange(-0.5, len(MODE_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(queries), 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linestyle="-", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="x", pad=2)

    for i in range(len(queries)):
        for j in range(len(MODE_ORDER)):
            value = matrix[i, j]
            if np.isnan(value):
                continue
            text_color = "white" if value > 0.68 else "#111111"
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", color=text_color, fontsize=6.0)
            best_type = best_types.get((i, j), "not_best")
            if best_type != "not_best":
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.2)
                ax.add_patch(rect)
                marker = "T" if best_type == "tied_best" else "U"
                ax.text(j + 0.34, i - 0.31, marker, ha="center", va="center", fontsize=5.5, fontweight="bold", color="black")

    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("QACS", labelpad=2)
    cbar.ax.tick_params(labelsize=6.2, width=0.4)


def annotate_bars(ax: Any, bars: Iterable[Any], color: str = "black", fmt: str = "{:.3f}") -> None:
    """Annotate vertical bars."""
    for bar in bars:
        height = float(bar.get_height())
        y = height - 0.018 if height >= 0.16 else height + 0.012
        va = "top" if height >= 0.16 else "bottom"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            fmt.format(height),
            ha="center",
            va=va,
            fontsize=5.4,
            rotation=90,
            color=color,
            clip_on=True,
        )


def plot_panel_b1(ax: Any, aggregate: pd.DataFrame, plt: Any) -> None:
    """Draw Panel B1 aggregate workflow metrics."""
    values = aggregate.set_index("mode").loc[list(MODE_ORDER)]
    x = np.arange(len(MODE_ORDER))
    width = 0.22
    specs = (
        ("mean_qacs", "QACS", "#4C78A8", ""),
        ("mean_fc", "FC", "#9A9A9A", "///"),
        ("mean_nqs", "NQS", "#D6A43A", "\\\\\\"),
    )
    for idx, (column, label, color, hatch) in enumerate(specs):
        bars = ax.bar(
            x + (idx - 1) * width,
            values[column].to_numpy(),
            width=width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.45,
            hatch=hatch,
        )
        label_color = "white" if column == "mean_qacs" else "black"
        annotate_bars(ax, bars, color=label_color)
    ax.set_title("(b) Aggregate workflow metrics", loc="left", fontweight="bold", pad=5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean")
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_TICK_LABELS[mode] for mode in MODE_ORDER], fontsize=6.2)
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(
        frameon=False,
        ncol=1,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.015),
        borderaxespad=0.0,
        handlelength=1.0,
        labelspacing=0.25,
        prop={"size": 5.8},
    )


def plot_panel_b2(ax: Any, top1_summary: pd.DataFrame, plt: Any) -> None:
    """Draw Panel B2 top-1 functional-match counts."""
    values = top1_summary.set_index("mode").loc[list(MODE_ORDER)]
    y = np.arange(len(MODE_ORDER))
    counts = values["top1_functional_match_count"].to_numpy(dtype=float)
    denominators = values["denominator"].to_numpy(dtype=float)
    max_denominator = int(np.nanmax(denominators)) if len(denominators) else 1
    bars = ax.barh(y, counts, color="#7F7F7F", edgecolor="black", linewidth=0.5, hatch="///")
    ax.set_title("(c) Top-1 functional match", loc="left", fontweight="bold", pad=5)
    ax.set_yticks(y)
    ax.set_yticklabels([MODE_LABELS[mode] for mode in MODE_ORDER], fontsize=6.4)
    ax.set_xlim(0, max_denominator * 1.22)
    ax.set_xlabel("Count")
    ax.invert_yaxis()
    ax.grid(axis="x", color="#dddddd", linewidth=0.5)
    ax.set_axisbelow(True)
    for bar, count, denominator in zip(bars, counts, denominators):
        label = f"{int(count)}/{int(denominator)}"
        ax.text(count + max_denominator * 0.03, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=6.4, clip_on=False)


def plot_inset(ax: Any, inset: pd.DataFrame, plt: Any) -> None:
    """Draw compact representative workflow path inset."""
    query_id = str(inset["query_id"].iloc[0])
    subtasks = sorted(inset["subtask_id"].unique(), key=subtask_sort_key)
    ax.set_title(f"Representative workflow path: {query_id}", loc="left", fontweight="bold", pad=3)
    ax.set_xlim(-0.8, len(subtasks))
    ax.set_ylim(0, len(MODE_ORDER))
    ax.axis("off")
    for row_idx, mode in enumerate(MODE_ORDER):
        y = len(MODE_ORDER) - 1 - row_idx
        ax.text(-0.1, y + 0.5, MODE_LABELS[mode], ha="right", va="center", fontsize=6.3)
        for col_idx, subtask_id in enumerate(subtasks):
            row = inset[(inset["mode"] == mode) & (inset["subtask_id"] == subtask_id)]
            if row.empty:
                continue
            item = row.iloc[0]
            is_functional = int(item["functional_match"]) == 1
            face = "#ffffff" if is_functional else "#e6e6e6"
            hatch = "" if is_functional else "////"
            rect = plt.Rectangle((col_idx, y + 0.05), 0.96, 0.9, facecolor=face, edgecolor="black", linewidth=0.55, hatch=hatch)
            ax.add_patch(rect)
            label = truncate_label(str(item["selected_api_label"]), 24)
            rank = item["selected_rank"]
            details = f"r={rank} F={int(item['functional_match'])}" if rank != "" else f"F={int(item['functional_match'])}"
            if not pd.isna(item["topsis_score"]):
                details += f" s={float(item['topsis_score']):.2f}"
            ax.text(col_idx + 0.48, y + 0.61, label, ha="center", va="center", fontsize=5.6)
            ax.text(col_idx + 0.48, y + 0.29, details, ha="center", va="center", fontsize=5.3)
    for col_idx, subtask_id in enumerate(subtasks):
        ax.text(col_idx + 0.48, len(MODE_ORDER) + 0.03, subtask_id.upper(), ha="center", va="bottom", fontsize=6.5, fontweight="bold")


def save_optional_inset(inset: pd.DataFrame, out_dir: Path, plt: Any) -> Path:
    """Save a separate inset figure when the main figure omits it."""
    fig, ax = plt.subplots(figsize=(7.16, 1.6))
    plot_inset(ax, inset, plt)
    path = out_dir / "fig3_representative_workflow_inset.pdf"
    fig.savefig(path, bbox_inches="tight", metadata={"Creator": SCRIPT_NAME, "CreationDate": None, "ModDate": None})
    plt.close(fig)
    return path


def make_figure(
    panel_a: pd.DataFrame,
    aggregate: pd.DataFrame,
    top1_summary: pd.DataFrame | None,
    inset: pd.DataFrame | None,
    out_dir: Path,
    include_inset: bool,
) -> list[Path]:
    """Create and save the full Figure 3 diagnostics plot."""
    plt = load_pyplot()
    has_b2 = top1_summary is not None and not top1_summary.empty
    has_inset = include_inset and inset is not None and not inset.empty
    inset_in_main = False
    separate_inset_path: Path | None = None
    if has_inset:
        subtask_count = inset["subtask_id"].nunique()
        inset_in_main = subtask_count <= 4
        if not inset_in_main:
            separate_inset_path = save_optional_inset(inset, out_dir, plt)

    height = 5.15 if inset_in_main else 4.65
    fig = plt.figure(figsize=(7.16, height))
    if inset_in_main:
        outer = fig.add_gridspec(
            2,
            2,
            width_ratios=[1.52, 1.0],
            height_ratios=[3.25, 1.25],
            left=0.065,
            right=0.985,
            top=0.945,
            bottom=0.065,
            wspace=0.34,
            hspace=0.48,
        )
        ax_a = fig.add_subplot(outer[0, 0])
        right = outer[0, 1].subgridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.62)
        ax_b1 = fig.add_subplot(right[0, 0])
        ax_b2 = fig.add_subplot(right[1, 0]) if has_b2 else None
        ax_c = fig.add_subplot(outer[1, :])
    else:
        outer = fig.add_gridspec(1, 2, width_ratios=[1.52, 1.0], left=0.065, right=0.985, top=0.94, bottom=0.12, wspace=0.34)
        ax_a = fig.add_subplot(outer[0, 0])
        if has_b2:
            right = outer[0, 1].subgridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.55)
            ax_b1 = fig.add_subplot(right[0, 0])
            ax_b2 = fig.add_subplot(right[1, 0])
        else:
            ax_b1 = fig.add_subplot(outer[0, 1])
            ax_b2 = None
        ax_c = None

    plot_panel_a(ax_a, fig, panel_a, plt)
    plot_panel_b1(ax_b1, aggregate, plt)
    if ax_b2 is not None and top1_summary is not None:
        plot_panel_b2(ax_b2, top1_summary, plt)
    if ax_c is not None and inset is not None:
        plot_inset(ax_c, inset, plt)

    outputs = [
        out_dir / "fig3_composition_diagnostics.pdf",
        out_dir / "fig3_composition_diagnostics.png",
        out_dir / "fig3_composition_diagnostics.svg",
    ]
    metadata = {"Creator": SCRIPT_NAME, "CreationDate": None, "ModDate": None}
    fig.savefig(outputs[0], bbox_inches="tight", metadata=metadata)
    fig.savefig(outputs[1], dpi=600, bbox_inches="tight")
    fig.savefig(outputs[2], bbox_inches="tight", metadata={"Creator": SCRIPT_NAME, "Date": None})
    plt.close(fig)
    if separate_inset_path:
        outputs.append(separate_inset_path)
    return outputs


def write_csv(path: Path, df: pd.DataFrame) -> None:
    """Write deterministic CSV output."""
    df.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")


def relative_map(paths: Iterable[Path], root: Path) -> list[str]:
    """Return stable run-relative paths when possible."""
    values: list[str] = []
    for path in sorted(set(paths), key=lambda p: p.as_posix()):
        try:
            values.append(path.resolve().relative_to(root.resolve()).as_posix())
        except ValueError:
            values.append(path.resolve().as_posix())
    return values


def write_manifest(
    path: Path,
    run_dir: Path,
    out_dir: Path,
    scanned_count: int,
    metric_paths: list[Path],
    ranking_paths: list[Path],
    selected_paths: list[Path],
    metrics: pd.DataFrame,
    top1: pd.DataFrame | None,
    inset_generated: bool,
    warnings: list[str],
    metric_candidates: list[dict[str, Any]],
    ranking_candidates: list[dict[str, Any]],
) -> None:
    """Write a manifest describing data provenance and checks."""
    detected_paths = sorted(set(metric_paths + ranking_paths + selected_paths), key=lambda p: p.as_posix())
    hashes: dict[str, str] = {}
    for file_path in detected_paths:
        try:
            key = file_path.resolve().relative_to(run_dir.resolve()).as_posix()
        except ValueError:
            key = file_path.resolve().as_posix()
        hashes[key] = sha256_file(file_path)

    query_subtask_cases = 0
    if top1 is not None and not top1.empty:
        query_subtask_cases = int(top1.drop_duplicates(["query_id", "subtask_id"]).shape[0])

    manifest = {
        "script_name": SCRIPT_NAME,
        "run_directory": run_dir.resolve().as_posix(),
        "output_directory": out_dir.resolve().as_posix(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files_scanned": scanned_count,
        "detected_input_files": {
            "workflow_metrics": relative_map(metric_paths, run_dir),
            "ranking_diagnostics": relative_map(ranking_paths, run_dir),
            "selected_paths_for_inset": relative_map(selected_paths, run_dir),
        },
        "input_file_sha256": hashes,
        "number_of_queries_detected": int(metrics["query_id"].nunique()),
        "number_of_query_subtask_cases_detected": query_subtask_cases,
        "modes_detected": [mode for mode in MODE_ORDER if mode in set(metrics["mode"])],
        "inset_generated": bool(inset_generated),
        "warnings": warnings,
        "metric_candidates": metric_candidates,
        "ranking_candidates": ranking_candidates,
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_discovery_report(
    scanned_count: int,
    metric_paths: list[Path],
    ranking_paths: list[Path],
    selected_paths: list[Path],
    metrics: pd.DataFrame,
    ranking: pd.DataFrame | None,
    metric_candidates: list[dict[str, Any]],
    ranking_candidates: list[dict[str, Any]],
) -> None:
    """Print concise discovery information."""
    metric_rows = {item["path"]: item["rows_loaded"] for item in metric_candidates}
    ranking_rows = {item["path"]: item["rows_loaded"] for item in ranking_candidates}
    print(f"Discovery: scanned {scanned_count} candidate data files.")
    print("Workflow metrics:")
    for path in metric_paths:
        print(f"  - {path.as_posix()} ({metric_rows.get(path.as_posix(), len(metrics))} rows loaded)")
    print("Ranking/selection diagnostics:")
    if ranking_paths:
        for path in ranking_paths:
            print(f"  - {path.as_posix()} ({ranking_rows.get(path.as_posix(), 0)} rows loaded)")
        if ranking is not None:
            print(f"  - combined ranking rows: {len(ranking)}")
    else:
        print("  - none")
    if selected_paths:
        print(f"Inset selected-path files: {len(selected_paths)} files ({len(selected_paths)} rows loaded)")


def main() -> int:
    """Entry point."""
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser()
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not run_dir.is_dir():
        print(f"ERROR: run directory does not exist: {run_dir}", file=sys.stderr)
        return 2

    warnings: list[str] = []
    files = discover_files(run_dir, out_dir)
    metrics, metric_paths, metric_candidates = find_workflow_metrics(files, warnings, debug=args.debug)
    ranking, top1, ranking_paths, ranking_candidates = find_ranking_diagnostics(
        files,
        warnings,
        allow_missing=args.allow_missing_ranking,
        debug=args.debug,
    )
    selected = pd.DataFrame()
    selected_paths: list[Path] = []
    inset = None
    if args.include_inset and top1 is not None:
        selected, selected_paths = find_selected_paths(files, debug=args.debug)
        inset = build_inset_source(metrics, top1, selected)
        if inset is None:
            warnings.append("Inset requested but selected API path data was unavailable or incomplete.")

    panel_a = compute_panel_a(metrics)
    aggregate = compute_aggregate_metrics(metrics)
    top1_summary = compute_top1_summary(top1)

    panel_a_path = out_dir / "fig3_panel_a_qacs_by_query.csv"
    aggregate_path = out_dir / "fig3_panel_b_aggregate_metrics.csv"
    top1_path = out_dir / "fig3_panel_b_top1_functional_match.csv"
    inset_path = out_dir / "fig3_inset_selected_paths.csv"
    manifest_path = out_dir / "fig3_manifest.json"

    write_csv(panel_a_path, panel_a)
    write_csv(aggregate_path, aggregate)
    if top1_summary is not None:
        write_csv(top1_path, top1_summary)
    if inset is not None:
        write_csv(inset_path, inset)

    figure_paths = make_figure(panel_a, aggregate, top1_summary, inset, out_dir, args.include_inset)
    write_manifest(
        manifest_path,
        run_dir,
        out_dir,
        len(files),
        metric_paths,
        ranking_paths,
        selected_paths,
        metrics,
        top1,
        inset is not None,
        warnings,
        metric_candidates,
        ranking_candidates,
    )

    print_discovery_report(len(files), metric_paths, ranking_paths, selected_paths, metrics, ranking, metric_candidates, ranking_candidates)
    print("Generated files:")
    for path in figure_paths + [panel_a_path, aggregate_path]:
        print(f"  - {path.as_posix()}")
    if top1_summary is not None:
        print(f"  - {top1_path.as_posix()}")
    if inset is not None:
        print(f"  - {inset_path.as_posix()}")
    print(f"  - {manifest_path.as_posix()}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
