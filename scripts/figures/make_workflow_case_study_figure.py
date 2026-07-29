#!/usr/bin/env python3
"""Generate a representative workflow case-study figure from run logs.

This script intentionally avoids dashboard or browser exports. It discovers
logged AutoLLMCompose result files, joins selected API records with functional
labels and QoS/TOPSIS data, chooses a deterministic three-subtask case, and
writes a vector-quality IEEE-style matplotlib figure plus audit artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_RUN_DIR = Path("results/logs/RUNS_MAY_31_NEW_5/fireworks_gpt-oss-120b")
DEFAULT_OUT_DIR = DEFAULT_RUN_DIR / "figures" / "paper"

MODE_ORDER = ("no_qos", "qos_pure_llm", "qos_topsis", "qos_hybrid")
MODE_DISPLAY = {
    "no_qos": "No-QoS",
    "qos_pure_llm": "QoS-Pure-LLM",
    "qos_topsis": "QoS-TOPSIS",
    "qos_hybrid": "QoS-Hybrid",
}
MODE_LOGIC = {
    "no_qos": "Functional metadata only",
    "qos_pure_llm": "LLM uses QoS context",
    "qos_topsis": "QoS-only TOPSIS",
    "qos_hybrid": "Functional-first + TOPSIS",
}
MODE_ALIASES = {
    "noqos": "no_qos",
    "no_qos": "no_qos",
    "noqosbaseline": "no_qos",
    "baseline": "no_qos",
    "functionalonly": "no_qos",
    "qospurellm": "qos_pure_llm",
    "qos_pure_llm": "qos_pure_llm",
    "qosllm": "qos_pure_llm",
    "llmqos": "qos_pure_llm",
    "purellm": "qos_pure_llm",
    "qostopsis": "qos_topsis",
    "qos_topsis": "qos_topsis",
    "topsis": "qos_topsis",
    "qoshybrid": "qos_hybrid",
    "qos_hybrid": "qos_hybrid",
    "hybrid": "qos_hybrid",
    "functionaltopsis": "qos_hybrid",
}
AVOID_TIED_CASES = {"q06", "q08", "q11", "q14"}

SUPPORTED_EXTENSIONS = {".json", ".jsonl", ".csv"}
QUERY_DIR_RE = re.compile(r"(q\d{2})(?:_|$)", re.IGNORECASE)
QUERY_ID_RE = re.compile(r"q?(\d{1,3})", re.IGNORECASE)
SUBTASK_PATH_RE = re.compile(r"(?:^|[_\-/])s(?:ubtask)?_?(\d+)(?:\D|$)", re.IGNORECASE)
FUNCTIONAL_CACHE_KEY_RE = re.compile(r"^(q\d{2})_(\d+)_(.+)$", re.IGNORECASE)

API_ID_ALIASES = (
    "api_id",
    "selected_api",
    "selected_api_id",
    "endpoint_id",
    "api_name",
    "endpoint_name_id",
)
CANDIDATE_ID_ALIASES = ("candidate_id", "candidate", "cid")
SUBTASK_ALIASES = (
    "subtask_id",
    "sub_task",
    "sub_task_id",
    "subtask",
    "subtask_index",
    "step_id",
    "step",
)
QUERY_ALIASES = ("query_id", "qid", "query")
MODE_ALIASES_COLUMNS = ("mode", "ranking_mode", "selection_mode", "method", "strategy")
FUNCTIONAL_ALIASES = (
    "functional_match",
    "functional_match_0_1",
    "functional_label",
    "match_label",
    "is_functional_match",
    "is_match",
    "label",
)
TOPSIS_ALIASES = ("topsis_score", "closeness_score", "closeness", "ci")
NQS_ALIASES = ("normalized_qos_score", "normalized_qos", "nqs")
QOS_SCORE_ALIASES = ("qos_score", "normalized_quality_score")
RAG_SCORE_ALIASES = ("rag_score", "retrieval_score")
RT_ALIASES = ("response_time", "response_time_s", "rt_s", "latency", "latency_s", "qos_rt_s")
TP_ALIASES = ("throughput", "throughput_kbps", "tp_kbps", "qos_tp_kbps")
AVAIL_ALIASES = ("availability", "qos_availability")
QACS_ALIASES = ("qacs", "qos_adjusted_composition_score", "composition_score", "final_score")
FC_ALIASES = ("fc", "functional_coverage", "function_coverage")
COMPLETENESS_ALIASES = ("composition_completeness", "completeness")
VALIDITY_ALIASES = ("composition_validity", "validity")

OUTPUT_BASENAME = "fig_workflow_case_study"
TOPSIS_CALLOUT_TEMPLATE = "QoS-only: {fm0}/3 FM=0 despite high TOPSIS"
HYBRID_CALLOUT_TEXT = "FM=1 gate, then TOPSIS"
SCORE_NOTE_TEXT = (
    "NQS is shown for No-QoS and QoS-Pure-LLM; TOPSIS is shown for QoS-TOPSIS and QoS-Hybrid. "
    "In QoS-Hybrid, TOPSIS is applied only after the FM=1 functional gate."
)


@dataclass(frozen=True)
class LoadedFile:
    path: Path
    rel_path: str
    data: Any
    kind: str


@dataclass
class SelectedRecord:
    query_id: str
    mode: str
    raw_mode: str
    subtask_index: int
    api_id: str
    candidate_id: str | None
    tool_name: str | None
    endpoint_name: str | None
    response_time: float | None
    throughput: float | None
    availability: float | None
    topsis_score: float | None
    normalized_qos: float | None
    qos_score: float | None
    rag_score: float | None
    mode_rank: float | None
    selection_order: float | None
    source_file: str
    raw: dict[str, Any]
    functional_match: int | None = None
    functional_source_file: str | None = None
    score_label: str = "Score"
    score_value: float | None = None
    score_status: str = "unavailable"
    score_source_file: str | None = None
    original_api_name: str = ""
    display_api_name: str = ""
    display_name_rule: str = ""
    short_api_name: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a paper-quality representative workflow comparison figure."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="Run directory to inspect.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    return parser.parse_args()


def normalize_key(value: Any) -> str:
    text = str(value).strip()
    for acronym in ("QoS", "API", "LLM", "QACS", "NQS", "FC"):
        text = re.sub(acronym, acronym.upper(), text, flags=re.IGNORECASE)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text.lower()).strip("_")
    return text


def compact_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def normalize_mode(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    candidates = {compact_key(raw), normalize_key(raw)}
    for candidate in candidates:
        if candidate in MODE_ALIASES:
            return MODE_ALIASES[candidate]
    return None


def normalize_query_id(value: Any) -> str | None:
    if value is None:
        return None
    match = QUERY_ID_RE.fullmatch(str(value).strip())
    if not match:
        return None
    number = int(match.group(1))
    if number <= 0:
        return None
    return f"q{number:02d}"


def query_sort_key(query_id: str) -> tuple[int, str]:
    normalized = normalize_query_id(query_id)
    if normalized:
        return (int(normalized[1:]), normalized)
    return (10_000, str(query_id))


def relative_to(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def record_map(record: dict[str, Any]) -> dict[str, Any]:
    return {normalize_key(key): value for key, value in record.items()}


def get_any(record: dict[str, Any], aliases: Iterable[str]) -> Any:
    mapped = record_map(record)
    for alias in aliases:
        key = normalize_key(alias)
        if key in mapped:
            return mapped[key]
    return None


def get_nested(record: dict[str, Any], path: Iterable[str]) -> Any:
    current: Any = record
    for part in path:
        if not isinstance(current, dict):
            return None
        mapped = {normalize_key(key): value for key, value in current.items()}
        current = mapped.get(normalize_key(part))
    return current


def finite_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def functional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        number = int(float(value))
        if number in (0, 1):
            return number
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "match", "functional"}:
        return 1
    if text in {"0", "false", "no", "mismatch", "nonfunctional", "not_functional"}:
        return 0
    return None


def first_nonempty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def infer_query_from_path(path: Path) -> str | None:
    for part in path.parts:
        match = QUERY_DIR_RE.search(part)
        if match:
            return match.group(1).lower()
        if part.lower().startswith("query_q"):
            query = normalize_query_id(part.lower().replace("query_", ""))
            if query:
                return query
    match = re.search(r"query_(q\d{2})", path.name, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None


def infer_mode_from_path(path: Path) -> tuple[str | None, str | None]:
    for part in path.parts:
        mode = normalize_mode(part)
        if mode:
            return mode, part
    return None, None


def infer_subtask_from_path(path: Path) -> int | None:
    match = SUBTASK_PATH_RE.search(path.stem)
    if match:
        return int(match.group(1))
    return None


def infer_query_from_record(record: dict[str, Any], path: Path) -> str | None:
    query = normalize_query_id(get_any(record, QUERY_ALIASES))
    return query or infer_query_from_path(path)


def infer_mode_from_record(record: dict[str, Any], path: Path) -> tuple[str | None, str | None]:
    raw = get_any(record, MODE_ALIASES_COLUMNS)
    mode = normalize_mode(raw)
    if mode:
        return mode, str(raw)
    return infer_mode_from_path(path)


def infer_subtask_from_record(record: dict[str, Any], path: Path) -> int | None:
    raw = get_any(record, SUBTASK_ALIASES)
    if raw is not None and raw != "":
        match = re.search(r"\d+", str(raw))
        if match:
            return int(match.group(0))
    return infer_subtask_from_path(path)


def service_dict(record: dict[str, Any]) -> dict[str, Any]:
    service = record.get("service")
    if isinstance(service, dict):
        return service
    return {}


def service_qos(record: dict[str, Any]) -> dict[str, Any]:
    qos = service_dict(record).get("qos")
    if isinstance(qos, dict):
        return qos
    return {}


def extract_api_id(record: dict[str, Any]) -> str | None:
    api_id = first_nonempty(
        get_any(record, API_ID_ALIASES),
        get_nested(record, ("service", "api_id")),
        get_nested(record, ("service", "toolbench_enrichment", "api_id")),
    )
    return str(api_id).strip() if api_id is not None and str(api_id).strip() else None


def extract_candidate_id(record: dict[str, Any]) -> str | None:
    candidate_id = get_any(record, CANDIDATE_ID_ALIASES)
    return str(candidate_id).strip() if candidate_id is not None and str(candidate_id).strip() else None


def extract_tool_name(record: dict[str, Any]) -> str | None:
    value = first_nonempty(
        get_any(record, ("tool_name", "tool", "service_name")),
        get_nested(record, ("service", "tool_name")),
        get_nested(record, ("service", "toolbench_tool_name")),
        get_nested(record, ("service", "toolbench_enrichment", "tool_name")),
    )
    return str(value).strip() if value is not None and str(value).strip() else None


def extract_endpoint_name(record: dict[str, Any]) -> str | None:
    value = first_nonempty(
        get_any(record, ("endpoint_name", "name", "endpoint", "api_name")),
        get_nested(record, ("service", "name")),
        get_nested(record, ("service", "endpoint_name")),
        get_nested(record, ("service", "toolbench_enrichment", "endpoint_name")),
    )
    return str(value).strip() if value is not None and str(value).strip() else None


def extract_qos(record: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    qos = service_qos(record)
    response_time = first_nonempty(get_any(record, RT_ALIASES), get_any(qos, RT_ALIASES))
    throughput = first_nonempty(get_any(record, TP_ALIASES), get_any(qos, TP_ALIASES))
    availability = first_nonempty(get_any(record, AVAIL_ALIASES), get_any(qos, AVAIL_ALIASES))
    return finite_float(response_time), finite_float(throughput), finite_float(availability)


def extract_subtask_text(record: dict[str, Any]) -> str | None:
    value = first_nonempty(
        get_any(record, ("subtask_text", "subtask_purpose", "description", "task_description", "purpose")),
        get_nested(record, ("subtask", "description")),
    )
    return str(value).strip() if value is not None and str(value).strip() else None


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL line") from exc
    return rows


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def discover_files(run_dir: Path, warnings: list[str]) -> list[LoadedFile]:
    loaded: list[LoadedFile] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        rel_parts = path.resolve().relative_to(run_dir.resolve()).parts
        if rel_parts and rel_parts[0] == "figures":
            continue
        if path.name.startswith("."):
            continue
        if path.name.startswith(OUTPUT_BASENAME):
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        rel = relative_to(path, run_dir)
        try:
            if path.suffix.lower() == ".json":
                data = load_json(path)
                kind = "json"
            elif path.suffix.lower() == ".jsonl":
                data = load_jsonl(path)
                kind = "jsonl"
            else:
                data = load_csv(path)
                kind = "csv"
        except Exception as exc:  # noqa: BLE001 - discovery should continue with warnings.
            warnings.append(f"Could not load {rel}: {exc}")
            continue
        loaded.append(LoadedFile(path=path, rel_path=rel, data=data, kind=kind))
    return loaded


def top_level_records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("rows", "data", "records", "results", "candidates", "ranked", "subtasks"):
            value = data.get(key)
            if isinstance(value, list):
                records = [item for item in value if isinstance(item, dict)]
                if records:
                    return records
        return [data]
    return []


def iter_dicts(data: Any) -> Iterable[dict[str, Any]]:
    if isinstance(data, dict):
        yield data
        for value in data.values():
            yield from iter_dicts(value)
    elif isinstance(data, list):
        for item in data:
            yield from iter_dicts(item)


def has_any_key(data: Any, aliases: Iterable[str]) -> bool:
    normalized_aliases = {normalize_key(alias) for alias in aliases}
    for record in iter_dicts(data):
        keys = {normalize_key(key) for key in record}
        if keys & normalized_aliases:
            return True
    return False


def looks_like_decomposer(data: Any) -> bool:
    records = top_level_records(data)
    if not records:
        return False
    if any(extract_api_id(record) for record in records[:5]):
        return False
    matches = 0
    for record in records:
        mapped = record_map(record)
        if ("description" in mapped or "subtask_purpose" in mapped) and (
            "id" in mapped or "subtask_id" in mapped or "sub_task" in mapped
        ):
            matches += 1
    return matches >= 2


def detect_source_files(loaded_files: list[LoadedFile]) -> dict[str, list[str]]:
    categories: dict[str, set[str]] = {
        "decomposed_queries_or_subtasks": set(),
        "selected_apis_by_mode": set(),
        "ranked_candidate_apis_by_mode": set(),
        "functional_match_labels": set(),
        "qos_fields": set(),
        "topsis_scores": set(),
        "normalized_qos_scores": set(),
        "query_level_metrics": set(),
    }
    for loaded in loaded_files:
        lower = loaded.rel_path.lower()
        stem = normalize_key(loaded.path.stem)
        records = top_level_records(loaded.data)
        if "decomposer" in lower or looks_like_decomposer(loaded.data):
            categories["decomposed_queries_or_subtasks"].add(loaded.rel_path)
        if "selected" in stem and "trace" not in stem:
            if any(extract_api_id(record) for record in records):
                categories["selected_apis_by_mode"].add(loaded.rel_path)
        if "ranked" in stem or "candidate_api_rankings" in lower:
            if any(extract_api_id(record) for record in records):
                categories["ranked_candidate_apis_by_mode"].add(loaded.rel_path)
        if has_any_key(loaded.data, FUNCTIONAL_ALIASES) or "functional_match" in lower:
            categories["functional_match_labels"].add(loaded.rel_path)
        if has_any_key(loaded.data, RT_ALIASES) and has_any_key(loaded.data, TP_ALIASES):
            categories["qos_fields"].add(loaded.rel_path)
        if has_any_key(loaded.data, TOPSIS_ALIASES):
            categories["topsis_scores"].add(loaded.rel_path)
        if has_any_key(loaded.data, NQS_ALIASES) or has_any_key(loaded.data, QOS_SCORE_ALIASES):
            categories["normalized_qos_scores"].add(loaded.rel_path)
        if has_any_key(loaded.data, QACS_ALIASES) and has_any_key(loaded.data, FC_ALIASES):
            categories["query_level_metrics"].add(loaded.rel_path)
    return {key: sorted(values) for key, values in categories.items()}


def build_indexes(
    loaded_files: list[LoadedFile],
    run_dir: Path,
) -> tuple[
    dict[str, Path],
    dict[str, str],
    dict[str, dict[int, str]],
    list[SelectedRecord],
    list[dict[str, Any]],
    dict[tuple[str, str | None, int, str], list[dict[str, Any]]],
    dict[tuple[str, int, str], list[dict[str, Any]]],
    dict[tuple[str, str], dict[str, Any]],
]:
    query_dirs: dict[str, Path] = {}
    query_text: dict[str, str] = {}
    subtasks: dict[str, dict[int, str]] = {}
    selected_records: list[SelectedRecord] = []
    candidate_records: list[dict[str, Any]] = []
    functional_index: dict[tuple[str, str | None, int, str], list[dict[str, Any]]] = {}
    candidate_by_api: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    metrics: dict[tuple[str, str], dict[str, Any]] = {}

    for path in run_dir.iterdir():
        if path.is_dir():
            match = QUERY_DIR_RE.search(path.name)
            if match:
                query_dirs[match.group(1).lower()] = path

    for loaded in loaded_files:
        rel = loaded.rel_path
        path = loaded.path
        path_query = infer_query_from_path(path)
        path_mode, raw_path_mode = infer_mode_from_path(path)
        path_subtask = infer_subtask_from_path(path)

        if path.name == "meta.json" and isinstance(loaded.data, dict):
            query = normalize_query_id(loaded.data.get("query_id")) or path_query
            if query:
                goal = first_nonempty(loaded.data.get("user_goal"), loaded.data.get("query_text"), loaded.data.get("query_title"))
                if goal:
                    query_text.setdefault(query, str(goal).strip())

        if "summary/all_15_query_composition_results.csv" in rel and isinstance(loaded.data, list):
            for row in loaded.data:
                if not isinstance(row, dict):
                    continue
                query = normalize_query_id(get_any(row, QUERY_ALIASES))
                if query:
                    text = first_nonempty(get_any(row, ("query_text", "user_goal", "goal")), get_any(row, ("query_title",)))
                    if text:
                        query_text.setdefault(query, str(text).strip())

        if "decomposer" in path.name.lower() or looks_like_decomposer(loaded.data):
            query = path_query
            records = top_level_records(loaded.data)
            for record in records:
                subtask_index = infer_subtask_from_record(record, path)
                text = extract_subtask_text(record)
                if query and subtask_index and text:
                    subtasks.setdefault(query, {})[subtask_index] = text

        if isinstance(loaded.data, dict) and "functional_match_cache" in path.name.lower():
            for key, value in loaded.data.items():
                if not isinstance(value, dict):
                    continue
                match = FUNCTIONAL_CACHE_KEY_RE.match(str(key))
                if not match:
                    continue
                query = match.group(1).lower()
                subtask_index = int(match.group(2))
                api_id = match.group(3)
                functional_match = functional_int(get_any(value, FUNCTIONAL_ALIASES))
                if functional_match is None:
                    continue
                entry = {
                    "query_id": query,
                    "mode": None,
                    "subtask_index": subtask_index,
                    "api_id": api_id,
                    "functional_match": functional_match,
                    "candidate_id": extract_candidate_id(value),
                    "source_file": rel,
                }
                functional_index.setdefault((query, None, subtask_index, api_id), []).append(entry)

        records = top_level_records(loaded.data)
        is_selected_json = "selected" in normalize_key(path.stem) and "trace" not in normalize_key(path.stem)
        is_ranked_json = "ranked" in normalize_key(path.stem)
        is_candidate_rows = "candidate_api_rankings" in rel.lower()

        for record in records:
            api_id = extract_api_id(record)
            query = infer_query_from_record(record, path)
            mode, raw_mode = infer_mode_from_record(record, path)
            subtask_index = infer_subtask_from_record(record, path)
            response_time, throughput, availability = extract_qos(record)
            functional_match = functional_int(get_any(record, FUNCTIONAL_ALIASES))
            subtask_text = extract_subtask_text(record)

            if query and subtask_index and subtask_text:
                subtasks.setdefault(query, {}).setdefault(subtask_index, subtask_text)

            if query and mode:
                qacs = finite_float(get_any(record, QACS_ALIASES))
                fc = finite_float(get_any(record, FC_ALIASES))
                nqs = finite_float(get_any(record, NQS_ALIASES))
                completeness = finite_float(get_any(record, COMPLETENESS_ALIASES))
                validity = finite_float(get_any(record, VALIDITY_ALIASES))
                if qacs is not None or fc is not None or nqs is not None:
                    key = (query, mode)
                    current = metrics.setdefault(key, {"query_id": query, "mode": mode, "source_file": rel})
                    for field, value in (
                        ("qacs", qacs),
                        ("fc", fc),
                        ("nqs", nqs),
                        ("completeness", completeness),
                        ("validity", validity),
                    ):
                        if value is not None and current.get(field) is None:
                            current[field] = value
                    current.setdefault("source_file", rel)

            if query and mode and subtask_index and api_id:
                candidate = {
                    "query_id": query,
                    "mode": mode,
                    "raw_mode": raw_mode or raw_path_mode or mode,
                    "subtask_index": subtask_index,
                    "api_id": api_id,
                    "candidate_id": extract_candidate_id(record),
                    "tool_name": extract_tool_name(record),
                    "endpoint_name": extract_endpoint_name(record),
                    "response_time": response_time,
                    "throughput": throughput,
                    "availability": availability,
                    "topsis_score": finite_float(get_any(record, TOPSIS_ALIASES)),
                    "normalized_qos": finite_float(get_any(record, NQS_ALIASES)),
                    "qos_score": finite_float(get_any(record, QOS_SCORE_ALIASES)),
                    "rag_score": finite_float(get_any(record, RAG_SCORE_ALIASES)),
                    "mode_rank": finite_float(get_any(record, ("mode_rank", "mode rank", "rank", "rank_position"))),
                    "functional_match": functional_match,
                    "source_file": rel,
                }
                if is_ranked_json or is_candidate_rows or is_selected_json or response_time is not None or functional_match is not None:
                    candidate_records.append(candidate)
                    candidate_by_api.setdefault((query, subtask_index, api_id), []).append(candidate)
                if functional_match is not None:
                    functional_index.setdefault((query, mode, subtask_index, api_id), []).append(
                        {
                            "query_id": query,
                            "mode": mode,
                            "subtask_index": subtask_index,
                            "api_id": api_id,
                            "functional_match": functional_match,
                            "candidate_id": extract_candidate_id(record),
                            "source_file": rel,
                        }
                    )

            if is_selected_json and path_mode and path_query and path_subtask and api_id:
                selected_records.append(
                    SelectedRecord(
                        query_id=path_query,
                        mode=path_mode,
                        raw_mode=raw_path_mode or path_mode,
                        subtask_index=path_subtask,
                        api_id=api_id,
                        candidate_id=extract_candidate_id(record),
                        tool_name=extract_tool_name(record),
                        endpoint_name=extract_endpoint_name(record),
                        response_time=response_time,
                        throughput=throughput,
                        availability=availability,
                        topsis_score=finite_float(get_any(record, TOPSIS_ALIASES)),
                        normalized_qos=finite_float(get_any(record, NQS_ALIASES)),
                        qos_score=finite_float(get_any(record, QOS_SCORE_ALIASES)),
                        rag_score=finite_float(get_any(record, RAG_SCORE_ALIASES)),
                        mode_rank=finite_float(get_any(record, ("mode_rank", "mode rank", "rank", "rank_position"))),
                        selection_order=finite_float(get_any(record, ("selection_order", "selected_rank", "planner_selected_rank"))),
                        source_file=rel,
                        raw=record,
                    )
                )

    return (
        query_dirs,
        query_text,
        subtasks,
        selected_records,
        candidate_records,
        functional_index,
        candidate_by_api,
        metrics,
    )


def merge_field(selected: SelectedRecord, field: str, candidates: list[dict[str, Any]]) -> Any:
    current = getattr(selected, field)
    if current is not None and current != "":
        return current
    for candidate in candidates:
        value = candidate.get(field)
        if value is not None and value != "":
            return value
    return current


def enrich_selected_records(
    selected_records: list[SelectedRecord],
    candidate_by_api: dict[tuple[str, int, str], list[dict[str, Any]]],
    functional_index: dict[tuple[str, str | None, int, str], list[dict[str, Any]]],
) -> list[SelectedRecord]:
    by_key: dict[tuple[str, str, int], SelectedRecord] = {}
    for selected in selected_records:
        candidates = candidate_by_api.get((selected.query_id, selected.subtask_index, selected.api_id), [])
        exact_candidates = [candidate for candidate in candidates if candidate.get("mode") == selected.mode]
        ordered_candidates = exact_candidates + [candidate for candidate in candidates if candidate.get("mode") != selected.mode]

        for field in (
            "candidate_id",
            "tool_name",
            "endpoint_name",
            "response_time",
            "throughput",
            "availability",
            "topsis_score",
            "normalized_qos",
            "qos_score",
            "rag_score",
            "mode_rank",
        ):
            setattr(selected, field, merge_field(selected, field, ordered_candidates))

        functional_candidates = (
            functional_index.get((selected.query_id, selected.mode, selected.subtask_index, selected.api_id), [])
            + functional_index.get((selected.query_id, None, selected.subtask_index, selected.api_id), [])
        )
        if not functional_candidates:
            for candidate in ordered_candidates:
                if candidate.get("functional_match") is not None:
                    functional_candidates.append(candidate)
        if functional_candidates:
            selected.functional_match = functional_int(functional_candidates[0].get("functional_match"))
            selected.functional_source_file = functional_candidates[0].get("source_file")

        key = (selected.query_id, selected.mode, selected.subtask_index)
        previous = by_key.get(key)
        if previous is None:
            by_key[key] = selected
            continue
        prev_order = first_nonempty(previous.selection_order, previous.mode_rank, 10_000.0)
        next_order = first_nonempty(selected.selection_order, selected.mode_rank, 10_000.0)
        if float(next_order) < float(prev_order):
            by_key[key] = selected
    return [by_key[key] for key in sorted(by_key, key=lambda item: (query_sort_key(item[0]), MODE_ORDER.index(item[1]), item[2]))]


def candidate_pool(
    query_id: str,
    mode: str,
    subtask_index: int,
    candidate_records: list[dict[str, Any]],
    selected_records: list[SelectedRecord],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in candidate_records:
        if record["query_id"] == query_id and record["mode"] == mode and record["subtask_index"] == subtask_index:
            rows.append(dict(record))
    for selected in selected_records:
        if selected.query_id == query_id and selected.mode == mode and selected.subtask_index == subtask_index:
            rows.append(
                {
                    "query_id": selected.query_id,
                    "mode": selected.mode,
                    "subtask_index": selected.subtask_index,
                    "api_id": selected.api_id,
                    "candidate_id": selected.candidate_id,
                    "tool_name": selected.tool_name,
                    "endpoint_name": selected.endpoint_name,
                    "response_time": selected.response_time,
                    "throughput": selected.throughput,
                    "availability": selected.availability,
                    "topsis_score": selected.topsis_score,
                    "normalized_qos": selected.normalized_qos,
                    "qos_score": selected.qos_score,
                    "source_file": selected.source_file,
                }
            )

    merged: dict[str, dict[str, Any]] = {}
    for row in rows:
        api_id = row.get("api_id")
        if not api_id:
            continue
        current = merged.setdefault(str(api_id), {"api_id": str(api_id), "source_files": []})
        if row.get("source_file"):
            current.setdefault("source_files", []).append(row["source_file"])
        for key, value in row.items():
            if key == "source_files":
                continue
            if current.get(key) is None and value is not None and value != "":
                current[key] = value
    return list(merged.values())


def minmax(values: list[float], value: float, higher_is_better: bool) -> float:
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return 1.0
    scaled = (value - low) / (high - low)
    return scaled if higher_is_better else 1.0 - scaled


def compute_normalized_qos(pool: list[dict[str, Any]]) -> dict[str, float]:
    values_by_key: dict[str, list[float]] = {}
    for key in ("response_time", "throughput", "availability"):
        values = [finite_float(row.get(key)) for row in pool]
        finite_values = [value for value in values if value is not None]
        if finite_values:
            values_by_key[key] = finite_values
    scores: dict[str, float] = {}
    for row in pool:
        components: list[float] = []
        response_time = finite_float(row.get("response_time"))
        throughput = finite_float(row.get("throughput"))
        availability = finite_float(row.get("availability"))
        if response_time is not None and "response_time" in values_by_key:
            components.append(minmax(values_by_key["response_time"], response_time, higher_is_better=False))
        if throughput is not None and "throughput" in values_by_key:
            components.append(minmax(values_by_key["throughput"], throughput, higher_is_better=True))
        if availability is not None and "availability" in values_by_key:
            components.append(minmax(values_by_key["availability"], availability, higher_is_better=True))
        api_id = row.get("api_id")
        if api_id and components:
            scores[str(api_id)] = sum(components) / len(components)
    return scores


def compute_topsis(pool: list[dict[str, Any]], weights: dict[str, float] | None = None) -> dict[str, float]:
    criteria = ("response_time", "throughput", "availability")
    benefit = {"response_time": False, "throughput": True, "availability": True}
    usable: list[dict[str, Any]] = []
    for row in pool:
        if not row.get("api_id"):
            continue
        if all(finite_float(row.get(key)) is not None for key in criteria):
            usable.append(row)
    if len(usable) < 2:
        return {}

    raw_weights = weights or {"response_time": 1.0, "throughput": 1.0, "availability": 1.0}
    total_weight = sum(float(raw_weights.get(key, 0.0)) for key in criteria) or 1.0
    normalized_weights = {key: float(raw_weights.get(key, 0.0)) / total_weight for key in criteria}

    weighted: dict[str, dict[str, float]] = {}
    for key in criteria:
        denom = math.sqrt(sum(float(row[key]) ** 2 for row in usable if finite_float(row.get(key)) is not None))
        if math.isclose(denom, 0.0):
            denom = 1.0
        for row in usable:
            api_id = str(row["api_id"])
            weighted.setdefault(api_id, {})[key] = (float(row[key]) / denom) * normalized_weights[key]

    ideals: dict[str, tuple[float, float]] = {}
    for key in criteria:
        column = [weighted[str(row["api_id"])][key] for row in usable]
        if benefit[key]:
            ideals[key] = (max(column), min(column))
        else:
            ideals[key] = (min(column), max(column))

    scores: dict[str, float] = {}
    for row in usable:
        api_id = str(row["api_id"])
        d_best = math.sqrt(sum((weighted[api_id][key] - ideals[key][0]) ** 2 for key in criteria))
        d_worst = math.sqrt(sum((weighted[api_id][key] - ideals[key][1]) ** 2 for key in criteria))
        denom = d_best + d_worst
        scores[api_id] = 0.0 if math.isclose(denom, 0.0) else d_worst / denom
    return scores


def recover_scores(
    selected_rows: list[SelectedRecord],
    candidate_records: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    status: dict[str, Any] = {
        "topsis": {"logged": [], "recomputed": [], "unavailable": []},
        "normalized_qos": {"logged": [], "recomputed": [], "unavailable": []},
    }
    for selected in selected_rows:
        row_id = f"{selected.query_id}/{selected.mode}/s{selected.subtask_index}/{selected.api_id}"
        pool = candidate_pool(selected.query_id, selected.mode, selected.subtask_index, candidate_records, selected_rows)
        if selected.mode in {"qos_topsis", "qos_hybrid"}:
            if selected.topsis_score is not None:
                selected.score_label = "TOPSIS"
                selected.score_value = selected.topsis_score
                selected.score_status = "logged"
                selected.score_source_file = selected.source_file
                status["topsis"]["logged"].append(row_id)
            else:
                scores = compute_topsis(pool)
                if selected.api_id in scores:
                    selected.score_label = "TOPSIS"
                    selected.score_value = scores[selected.api_id]
                    selected.score_status = "recomputed"
                    status["topsis"]["recomputed"].append(row_id)
                    warnings.append(f"Recomputed TOPSIS for {row_id} from QoS components.")
                else:
                    selected.score_label = "Score"
                    selected.score_value = None
                    selected.score_status = "unavailable"
                    status["topsis"]["unavailable"].append(row_id)
                    warnings.append(f"TOPSIS score unavailable for {row_id}; card shows Score=N/A.")
        else:
            if selected.normalized_qos is not None:
                selected.score_label = "NQS"
                selected.score_value = selected.normalized_qos
                selected.score_status = "logged"
                selected.score_source_file = selected.source_file
                status["normalized_qos"]["logged"].append(row_id)
            elif selected.qos_score is not None:
                selected.score_label = "QoS"
                selected.score_value = selected.qos_score
                selected.score_status = "logged"
                selected.score_source_file = selected.source_file
                status["normalized_qos"]["logged"].append(row_id)
            else:
                scores = compute_normalized_qos(pool)
                if selected.api_id in scores:
                    selected.score_label = "NQS"
                    selected.score_value = scores[selected.api_id]
                    selected.score_status = "recomputed"
                    status["normalized_qos"]["recomputed"].append(row_id)
                    warnings.append(f"Recomputed normalized QoS for {row_id} from QoS components.")
                else:
                    selected.score_label = "Score"
                    selected.score_value = None
                    selected.score_status = "unavailable"
                    status["normalized_qos"]["unavailable"].append(row_id)
                    warnings.append(f"Normalized QoS unavailable for {row_id}; card shows Score=N/A.")
    return status


def build_selected_lookup(selected_rows: list[SelectedRecord]) -> dict[tuple[str, str, int], SelectedRecord]:
    return {(row.query_id, row.mode, row.subtask_index): row for row in selected_rows}


def select_representative_query(
    selected_rows: list[SelectedRecord],
    subtasks: dict[str, dict[int, str]],
    metrics: dict[tuple[str, str], dict[str, Any]],
) -> tuple[str, str, list[dict[str, Any]]]:
    selected_lookup = build_selected_lookup(selected_rows)
    query_ids = sorted({row.query_id for row in selected_rows} | set(subtasks), key=query_sort_key)
    candidate_scores: list[dict[str, Any]] = []

    for query_id in query_ids:
        subtask_map = subtasks.get(query_id, {})
        inferred_subtasks = sorted({row.subtask_index for row in selected_rows if row.query_id == query_id})
        subtask_count = len(subtask_map) if subtask_map else len(inferred_subtasks)
        exactly_three = subtask_count == 3
        required_subtasks = (1, 2, 3)
        all_selected = all((query_id, mode, subtask) in selected_lookup for mode in MODE_ORDER for subtask in required_subtasks)
        fm_available = all(
            selected_lookup.get((query_id, mode, subtask)) is not None
            and selected_lookup[(query_id, mode, subtask)].functional_match is not None
            for mode in MODE_ORDER
            for subtask in required_subtasks
        )
        hybrid_all_fm1 = all(
            selected_lookup.get((query_id, "qos_hybrid", subtask)) is not None
            and selected_lookup[(query_id, "qos_hybrid", subtask)].functional_match == 1
            for subtask in required_subtasks
        )
        topsis_has_fm0 = any(
            selected_lookup.get((query_id, "qos_topsis", subtask)) is not None
            and selected_lookup[(query_id, "qos_topsis", subtask)].functional_match == 0
            for subtask in required_subtasks
        )
        qacs_by_mode = {mode: metrics.get((query_id, mode), {}).get("qacs") for mode in MODE_ORDER}
        finite_qacs = [value for value in qacs_by_mode.values() if value is not None]
        hybrid_qacs = qacs_by_mode.get("qos_hybrid")
        pure_qacs = qacs_by_mode.get("qos_pure_llm")
        topsis_qacs = qacs_by_mode.get("qos_topsis")
        hybrid_highest_or_tied = bool(
            finite_qacs and hybrid_qacs is not None and hybrid_qacs >= max(finite_qacs) - 1e-12
        )
        hybrid_unique_best_vs_pure = bool(hybrid_qacs is not None and pure_qacs is not None and hybrid_qacs > pure_qacs + 1e-12)
        improvement = (
            float(hybrid_qacs) - float(topsis_qacs)
            if hybrid_qacs is not None and topsis_qacs is not None
            else float("-inf")
        )
        satisfies_core = bool(exactly_three and all_selected and fm_available)
        satisfies_all_preferences = bool(
            satisfies_core
            and hybrid_all_fm1
            and topsis_has_fm0
            and hybrid_highest_or_tied
            and hybrid_unique_best_vs_pure
        )
        candidate_scores.append(
            {
                "query_id": query_id,
                "subtask_count": subtask_count,
                "exactly_three_subtasks": exactly_three,
                "all_required_mode_subtask_selections": all_selected,
                "functional_labels_available": fm_available,
                "hybrid_all_functional_match_1": hybrid_all_fm1,
                "topsis_has_functional_match_0": topsis_has_fm0,
                "hybrid_highest_or_tied_qacs": hybrid_highest_or_tied,
                "hybrid_unique_best_over_qos_pure_llm": hybrid_unique_best_vs_pure,
                "avoid_tied_case": query_id in AVOID_TIED_CASES,
                "hybrid_minus_topsis_qacs": None if improvement == float("-inf") else improvement,
                "hybrid_qacs": hybrid_qacs,
                "qos_pure_llm_qacs": pure_qacs,
                "qos_topsis_qacs": topsis_qacs,
                "satisfies_core_requirements": satisfies_core,
                "satisfies_all_preferences": satisfies_all_preferences,
                "q02_preferred": query_id == "q02" and satisfies_all_preferences,
            }
        )

    q02 = next((item for item in candidate_scores if item["query_id"] == "q02" and item["q02_preferred"]), None)
    if q02:
        return "q02", "q02 satisfies the three-subtask, complete-selection, Hybrid-functional, TOPSIS-failure, and Hybrid-best QACS preferences.", candidate_scores

    eligible = [item for item in candidate_scores if item["satisfies_core_requirements"]]
    if not eligible:
        fallback = [item for item in candidate_scores if item["exactly_three_subtasks"]]
        if not fallback:
            raise ValueError("No query with exactly three subtasks was found.")
        best = sorted(fallback, key=lambda item: query_sort_key(item["query_id"]))[0]
        return best["query_id"], "Fallback: no three-subtask query had all required mode/subtask selections and functional labels.", candidate_scores

    def sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
        improvement = item["hybrid_minus_topsis_qacs"]
        return (
            not item["hybrid_all_functional_match_1"],
            not item["topsis_has_functional_match_0"],
            not item["hybrid_highest_or_tied_qacs"],
            not item["hybrid_unique_best_over_qos_pure_llm"],
            item["avoid_tied_case"],
            -(improvement if improvement is not None else float("-inf")),
            query_sort_key(item["query_id"]),
        )

    best = sorted(eligible, key=sort_key)[0]
    reason = "Selected by deterministic preference ranking over eligible three-subtask queries."
    if not best["satisfies_all_preferences"]:
        reason = "Fallback: selected the best available three-subtask query because no query satisfied all preferences."
    return best["query_id"], reason, candidate_scores


def clean_name_piece(value: str) -> str:
    text = str(value)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\{[^}]*\}", " ", text)
    text = text.replace("...", " ").replace("\u2026", " ")
    text = re.sub(r"[_/\-]+", " ", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"\b(v|api)\s*\d+\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(get|post|put|delete)\b", lambda match: match.group(1).title(), text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" :;,-")
    if not text:
        return ""
    words = text.split()
    deduped: list[str] = []
    for word in words:
        if deduped and deduped[-1].lower() == word.lower():
            continue
        deduped.append(word)
    return " ".join(deduped)


def title_name(text: str) -> str:
    keep_upper = {"API", "USA", "SMS", "SMTP", "URL", "IP", "ID", "ML"}
    words = []
    for word in text.split():
        cleaned = word.strip()
        if cleaned.upper() in keep_upper:
            words.append(cleaned.upper())
        elif cleaned.isupper() and len(cleaned) <= 4:
            words.append(cleaned)
        elif any(ch.isdigit() for ch in cleaned):
            words.append(cleaned[:1].upper() + cleaned[1:])
        else:
            words.append(cleaned[:1].upper() + cleaned[1:].lower())
    return " ".join(words)


def readable_api_name(tool_name: str | None, endpoint_name: str | None, api_id: str) -> str:
    tool = clean_name_piece(tool_name or "")
    endpoint = clean_name_piece(endpoint_name or "")
    if endpoint.lower() in {"default", "get", "post", "api", "endpoint"}:
        endpoint = ""
    if not tool and not endpoint:
        endpoint = clean_name_piece(api_id)
    if tool and endpoint and endpoint.lower() not in tool.lower():
        name = f"{tool} {endpoint}"
    else:
        name = tool or endpoint
    return title_name(name)


DISPLAY_NAME_ALIASES = {
    "yelpapigetbusinesses": ("Yelp Businesses", "q02_alias_yelp_businesses"),
    "cannafindershopsinradiusaroundcoordinates": ("Canna Finder Nearby Shops", "q02_alias_canna_finder_nearby_shops"),
    "emailer420": ("E-Mailer 420", "q02_alias_e_mailer_420"),
    "restaurantsreviews": ("Restaurant Reviews", "q02_alias_restaurant_reviews"),
    "veggiemerestaurants": ("Veggie Me Restaurants", "q02_alias_veggie_me_restaurants"),
}


def clean_api_display_name(tool_name: str | None, endpoint_name: str | None, raw_name: str) -> tuple[str, str]:
    """Return a concise deterministic display name and the cleanup rule used."""
    original = clean_name_piece(raw_name)
    if not original:
        original = readable_api_name(tool_name, endpoint_name, raw_name)
    original = title_name(original)
    alias = DISPLAY_NAME_ALIASES.get(compact_key(original))
    if alias:
        return alias

    display = original
    display = re.sub(r"\bAPI\b", "", display).strip()
    display = re.sub(r"\bGet\s+([A-Z])", r"\1", display)
    display = re.sub(r"\bIn Radius Around Coordinates\b", "Nearby", display)
    display = re.sub(r"\bReviews\b", "Reviews", display)
    display = re.sub(r"\bE Mailer\b", "E-Mailer", display)
    display = re.sub(r"\s+", " ", display).strip()
    if not display:
        display = original
    rule = "generic_cleanup" if display != original else "original_readable_name"
    return display, rule


def wrap_two_lines(text: str, width: int = 25) -> str:
    words = text.split()
    if not words:
        return text
    if len(text) <= width:
        return text
    best: tuple[int, str, str] | None = None
    for split in range(1, len(words)):
        left = " ".join(words[:split])
        right = " ".join(words[split:])
        cost = abs(len(left) - len(right)) + max(0, len(left) - width) * 2 + max(0, len(right) - width) * 2
        if best is None or cost < best[0]:
            best = (cost, left, right)
    if best is None:
        return text
    return f"{best[1]}\n{best[2]}"


def shorten_subtask_text(text: str) -> str:
    cleaned = str(text).strip()
    if re.search(r"\bsend\b", cleaned, flags=re.IGNORECASE) and re.search(
        r"\bemail\b", cleaned, flags=re.IGNORECASE
    ) and re.search(r"\brestaurant recommendations\b", cleaned, flags=re.IGNORECASE):
        return "Send selected restaurant\nrecommendations by email"
    replacements = [
        (r"\s+using an external [^,;.]+", ""),
        (r"\s+using an external data API", ""),
        (r"\s+with selected", " selected"),
        (r"Retrieve restaurant reviews or details", "Retrieve restaurant reviews/details"),
        (r"Search for nearby restaurants", "Search nearby restaurants"),
    ]
    for pattern, repl in replacements:
        cleaned = re.sub(pattern, repl, cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    wrapped = textwrap.wrap(cleaned, width=28, break_long_words=False, break_on_hyphens=False)
    if len(wrapped) <= 2:
        return "\n".join(wrapped)
    compact = " ".join(wrapped)
    compact = compact.replace(" restaurant ", " ")
    wrapped = textwrap.wrap(compact, width=28, break_long_words=False, break_on_hyphens=False)
    return "\n".join(wrapped[:2])


def validate_names(rows: list[SelectedRecord]) -> None:
    for row in rows:
        if "..." in row.short_api_name or "\u2026" in row.short_api_name:
            raise ValueError(f"Rendered API name is truncated with ellipses: {row.short_api_name!r}")
        lines = row.short_api_name.splitlines()
        if len(lines) > 2:
            raise ValueError(f"Rendered API name exceeds two lines: {row.short_api_name!r}")
        if any(len(line) > 38 for line in lines):
            raise ValueError(f"Rendered API name is too long to read cleanly: {row.short_api_name!r}")


def configure_matplotlib() -> None:
    cache_root = Path(tempfile.gettempdir()) / "autollmcompose_workflow_case_study"
    mpl_cache_dir = cache_root / "matplotlib"
    xdg_cache_dir = cache_root / "xdg"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_cache_dir.as_posix())
    os.environ.setdefault("XDG_CACHE_HOME", xdg_cache_dir.as_posix())


def preferred_serif_font_name() -> str:
    configure_matplotlib()
    from matplotlib import font_manager

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    return "Times New Roman" if "Times New Roman" in available_fonts else "DejaVu Serif"


def load_pyplot():
    configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    serif_font = preferred_serif_font_name()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [serif_font],
            "font.size": 7.5,
            "axes.linewidth": 0.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.max_open_warning": 0,
        }
    )
    return plt


def fmt_metric(value: Any) -> str:
    number = finite_float(value)
    return "N/A" if number is None else f"{number:.3f}"


def score_text(row: SelectedRecord) -> str:
    value = "N/A" if row.score_value is None else f"{row.score_value:.3f}"
    return f"{row.score_label}={value}"


def row_metrics_text(metric: dict[str, Any] | None) -> str:
    if not metric:
        return ""
    parts = []
    if metric.get("qacs") is not None:
        parts.append(f"QACS={fmt_metric(metric.get('qacs'))}")
    if metric.get("fc") is not None:
        parts.append(f"FC={fmt_metric(metric.get('fc'))}")
    if metric.get("nqs") is not None:
        parts.append(f"NQS={fmt_metric(metric.get('nqs'))}")
    return "\n".join(parts)


def concise_case_subtitle(query_text: str) -> str:
    text = query_text.lower()
    if "restaurant" in text and "email" in text:
        return "Restaurant workflow"
    if not query_text:
        return ""
    cleaned = re.sub(r"^build a service that\s+", "", query_text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"^create\s+", "", cleaned, flags=re.IGNORECASE)
    words = cleaned.split()
    return " ".join(words[:5]).rstrip(",.;")


def mode_functional_counts(rows: list[SelectedRecord]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for mode in MODE_ORDER:
        mode_rows = [row for row in rows if row.mode == mode]
        fm1 = sum(1 for row in mode_rows if row.functional_match == 1)
        fm0 = sum(1 for row in mode_rows if row.functional_match == 0)
        counts[mode] = {"fm1": fm1, "fm0": fm0, "total": len(mode_rows)}
    return counts


def highest_qacs_modes(selected_query: str, metrics: dict[tuple[str, str], dict[str, Any]]) -> list[str]:
    qacs_values = {
        mode: finite_float(metrics.get((selected_query, mode), {}).get("qacs"))
        for mode in MODE_ORDER
    }
    finite_values = [value for value in qacs_values.values() if value is not None]
    if not finite_values:
        return []
    best = max(finite_values)
    return [mode for mode, value in qacs_values.items() if value is not None and math.isclose(value, best, rel_tol=1e-12, abs_tol=1e-12)]


def bbox_contains(container: Any, inner: Any, pad: float = 0.0) -> bool:
    return (
        inner.x0 >= container.x0 + pad
        and inner.x1 <= container.x1 - pad
        and inner.y0 >= container.y0 + pad
        and inner.y1 <= container.y1 - pad
    )


def validate_rendered_layout(fig: Any, renderer: Any, layout_items: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    errors: list[str] = []
    figure_bbox = fig.bbox
    text_artists = layout_items["texts"]

    for label, artist in text_artists:
        text = artist.get_text()
        if "..." in text or "\u2026" in text:
            errors.append(f"Rendered text contains ellipses in {label}: {text!r}")
        if "->" in text:
            errors.append(f"Rendered text contains '->' in {label}: {text!r}")
        if re.search(r"\brecs\b", text, flags=re.IGNORECASE):
            errors.append(f"Rendered text contains informal abbreviation 'recs' in {label}: {text!r}")
        bbox = artist.get_window_extent(renderer)
        if not bbox_contains(figure_bbox, bbox, pad=-1.0):
            errors.append(f"Rendered text is clipped outside the figure canvas in {label}: {text!r}")

    text_bboxes = {label: artist.get_window_extent(renderer) for label, artist in text_artists}
    for item in layout_items["cards"]:
        card_bbox = item["patch"].get_window_extent(renderer)
        for text_artist in item["texts"]:
            text_bbox = text_artist.get_window_extent(renderer)
            if not bbox_contains(card_bbox, text_bbox, pad=3.0):
                errors.append(
                    f"Card text extends outside its card for {item['mode']} S{item['subtask']}: "
                    f"{text_artist.get_text()!r}"
                )

    metrics_bboxes = [artist.get_window_extent(renderer) for artist in layout_items["metrics"]]
    card_bboxes = [item["patch"].get_window_extent(renderer) for item in layout_items["cards"]]
    annotation_bboxes = [artist.get_window_extent(renderer) for artist in layout_items.get("annotations", [])]
    arrow_bboxes = [patch.get_window_extent(renderer) for patch in layout_items.get("arrows", [])]
    badge_items = layout_items.get("badges", [])
    badge_bboxes = [
        (item["kind"], item["patch"].get_window_extent(renderer), item["text"].get_window_extent(renderer))
        for item in badge_items
    ]
    hybrid_band = layout_items.get("hybrid_band")
    hybrid_band_bbox = hybrid_band.get_window_extent(renderer) if hybrid_band is not None else None
    metrics_boundary_x = ax_data_to_display(layout_items["ax"], layout_items["metrics_column_left"])[0]

    for bbox in annotation_bboxes:
        if bbox.x1 > metrics_boundary_x:
            errors.append("An annotation overlaps or enters the metrics column.")

    for kind, badge_bbox, text_bbox in badge_bboxes:
        if badge_bbox.x1 > metrics_boundary_x:
            errors.append(f"The {kind} callout badge overlaps or enters the metrics column.")
        if not bbox_contains(badge_bbox, text_bbox, pad=2.0):
            errors.append(f"The {kind} callout text extends outside its badge.")

    for metric_bbox in metrics_bboxes:
        for card_bbox in card_bboxes:
            if metric_bbox.overlaps(card_bbox):
                errors.append("A metrics block overlaps an API card.")
        for annotation_bbox in annotation_bboxes:
            if metric_bbox.overlaps(annotation_bbox):
                errors.append("A metrics block overlaps a callout annotation.")
        for kind, badge_bbox, _ in badge_bboxes:
            if metric_bbox.overlaps(badge_bbox):
                errors.append(f"A metrics block overlaps the {kind} callout badge.")

    for annotation_bbox in annotation_bboxes:
        for card_bbox in card_bboxes:
            if annotation_bbox.overlaps(card_bbox):
                errors.append("A callout annotation overlaps an API card.")

    for kind, badge_bbox, _ in badge_bboxes:
        for card_bbox in card_bboxes:
            if badge_bbox.overlaps(card_bbox):
                errors.append(f"The {kind} callout badge overlaps an API card.")
        for arrow_bbox in arrow_bboxes:
            if badge_bbox.overlaps(arrow_bbox):
                errors.append(f"The {kind} callout badge overlaps a workflow arrow.")
        if hybrid_band_bbox is not None:
            if kind == "qos_topsis" and badge_bbox.overlaps(hybrid_band_bbox):
                errors.append("The QoS-TOPSIS callout badge overlaps the QoS-Hybrid highlight band.")
            if kind == "qos_hybrid" and not bbox_contains(hybrid_band_bbox, badge_bbox, pad=1.0):
                errors.append("The QoS-Hybrid callout badge is not fully inside the QoS-Hybrid highlight band.")
        if kind == "qos_hybrid":
            for text_label in ("qos_hybrid_label", "qos_hybrid_logic"):
                text_bbox = text_bboxes.get(text_label)
                if text_bbox is not None and badge_bbox.overlaps(text_bbox):
                    errors.append(f"The QoS-Hybrid callout badge overlaps {text_label}.")

    s3_description_bbox = text_bboxes.get("s3_description")
    if s3_description_bbox is not None and s3_description_bbox.x1 > metrics_boundary_x - 2.0:
        errors.append("The S3 subtask label overlaps or enters the workflow metrics column.")

    note_bbox = layout_items["note"].get_window_extent(renderer)
    for legend_artist in layout_items["legend"]:
        if note_bbox.overlaps(legend_artist.get_window_extent(renderer)):
            errors.append("The score semantics note overlaps the legend.")

    if errors:
        raise ValueError("Layout validation failed:\n- " + "\n- ".join(errors))
    return warnings


def ax_data_to_display(ax: Any, point: tuple[float, float]) -> tuple[float, float]:
    x, y = point
    return tuple(ax.transData.transform((x, y)))


def draw_case_study_figure(
    selected_query: str,
    query_text: str,
    query_subtasks: dict[int, str],
    rows: list[SelectedRecord],
    metrics: dict[tuple[str, str], dict[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    plt = load_pyplot()
    from matplotlib.patches import FancyArrowPatch, Rectangle

    fig, ax = plt.subplots(figsize=(7.16, 4.30))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    selected_lookup = {(row.mode, row.subtask_index): row for row in rows}
    fm_counts = mode_functional_counts(rows)
    highest_modes = highest_qacs_modes(selected_query, metrics)
    row_y = {
        "no_qos": 0.735,
        "qos_pure_llm": 0.555,
        "qos_topsis": 0.375,
        "qos_hybrid": 0.185,
    }
    x_centers = {1: 0.335, 2: 0.565, 3: 0.795}
    label_x = 0.022
    metrics_x = 0.925
    metrics_column_left = (0.907, 0.0)
    card_w = 0.178
    card_h = 0.128
    layout_items: dict[str, Any] = {
        "ax": ax,
        "texts": [],
        "cards": [],
        "metrics": [],
        "annotations": [],
        "badges": [],
        "arrows": [],
        "hybrid_band": None,
        "legend": [],
        "note": None,
        "metrics_column_left": metrics_column_left,
    }

    def add_text(label: str, *args: Any, **kwargs: Any) -> Any:
        artist = ax.text(*args, **kwargs)
        layout_items["texts"].append((label, artist))
        return artist

    hybrid_y = row_y["qos_hybrid"]
    hybrid_band_patch = Rectangle(
        (0.008, hybrid_y - 0.087),
        0.892,
        0.163,
        facecolor="#f8f8f8",
        edgecolor="#c4c4c4",
        linewidth=0.45,
        zorder=0,
    )
    ax.add_patch(hybrid_band_patch)
    layout_items["hybrid_band"] = hybrid_band_patch
    ax.plot(
        [metrics_column_left[0], metrics_column_left[0]],
        [0.104, 0.886],
        color="#d8d8d8",
        linewidth=0.45,
        zorder=1,
    )

    def add_badge(
        kind: str,
        label: str,
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        text: str,
        facecolor: str,
        edgecolor: str,
        text_color: str,
        fontsize: float,
        linespacing: float = 1.0,
        linewidth: float = 0.70,
    ) -> None:
        badge_patch = Rectangle(
            (center_x - width / 2, center_y - height / 2),
            width,
            height,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            joinstyle="miter",
            zorder=2.5,
        )
        ax.add_patch(badge_patch)
        badge_text = add_text(
            label,
            center_x,
            center_y,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=text_color,
            linespacing=linespacing,
            zorder=3,
        )
        layout_items["badges"].append({"kind": kind, "patch": badge_patch, "text": badge_text})

    for subtask_index in (1, 2, 3):
        x = x_centers[subtask_index]
        description = shorten_subtask_text(query_subtasks[subtask_index])
        add_text(f"s{subtask_index}_header", x, 0.935, f"S{subtask_index}", ha="center", va="bottom", fontsize=10.2, fontweight="bold")
        add_text(f"s{subtask_index}_description", x, 0.897, description, ha="center", va="top", fontsize=8.0, linespacing=1.05)

    subtitle = concise_case_subtitle(query_text)
    add_text("case_label", label_x, 0.912, f"Representative case: {selected_query}", ha="left", va="top", fontsize=7.4, fontweight="bold")
    if subtitle:
        add_text("case_subtitle", label_x, 0.877, subtitle, ha="left", va="top", fontsize=6.9, color="#333333")
    add_text("metrics_header", metrics_x, 0.897, "Workflow metrics", ha="left", va="top", fontsize=7.1, fontweight="bold")

    for mode in MODE_ORDER:
        y = row_y[mode]
        add_text(f"{mode}_label", label_x, y + 0.033, MODE_DISPLAY[mode], ha="left", va="center", fontsize=8.9, fontweight="bold")
        add_text(f"{mode}_logic", label_x, y - 0.006, MODE_LOGIC[mode], ha="left", va="center", fontsize=7.1, color="#333333")
        if mode == "qos_hybrid":
            add_badge(
                "qos_hybrid",
                f"{mode}_callout",
                label_x + 0.086,
                y - 0.042,
                0.172,
                0.033,
                HYBRID_CALLOUT_TEXT,
                "#fbfdfb",
                "#2e6b3c",
                "#244a2b",
                5.15,
                linewidth=0.60,
            )

        for subtask_index in (1, 2, 3):
            row = selected_lookup[(mode, subtask_index)]
            x = x_centers[subtask_index]
            is_match = row.functional_match == 1
            face = "#edf5ec" if is_match else "#fbefef"
            edge = "#2e6b3c" if is_match else "#8a2f2f"
            linestyle = "solid" if is_match else (0, (3, 2))
            linewidth = 1.05 if is_match else 1.35
            if mode == "qos_hybrid":
                linewidth += 0.35
            card_patch = Rectangle(
                (x - card_w / 2, y - card_h / 2),
                card_w,
                card_h,
                facecolor=face,
                edgecolor=edge,
                linewidth=linewidth,
                linestyle=linestyle,
                joinstyle="miter",
                zorder=3,
            )
            ax.add_patch(card_patch)
            name_artist = add_text(
                f"{mode}_s{subtask_index}_api_name",
                x,
                y + 0.023,
                row.short_api_name,
                ha="center",
                va="center",
                fontsize=6.75,
                linespacing=1.02,
                zorder=4,
            )
            score_artist = add_text(
                f"{mode}_s{subtask_index}_fm_score",
                x,
                y - 0.034,
                f"FM={row.functional_match}    {score_text(row)}",
                ha="center",
                va="center",
                fontsize=6.8,
                zorder=4,
            )
            layout_items["cards"].append(
                {"mode": mode, "subtask": subtask_index, "patch": card_patch, "texts": [name_artist, score_artist]}
            )

        for left, right in ((1, 2), (2, 3)):
            x0 = x_centers[left] + card_w / 2 + 0.008
            x1 = x_centers[right] - card_w / 2 - 0.008
            arrow_patch = FancyArrowPatch(
                (x0, y),
                (x1, y),
                arrowstyle="-|>",
                mutation_scale=6.5,
                linewidth=0.7,
                color="#555555",
                zorder=2,
            )
            ax.add_patch(arrow_patch)
            layout_items["arrows"].append(arrow_patch)

        metric = metrics.get((selected_query, mode))
        if metric:
            qacs_weight = "bold" if mode in highest_modes else "normal"
            fc_value = finite_float(metric.get("fc"))
            fc_weight = "bold" if fc_value is not None and fc_value < 1.0 else "normal"
            fc_color = "#5a2525" if fc_value is not None and fc_value < 1.0 else "#222222"
            metric_lines = [
                ("qacs", f"QACS={fmt_metric(metric.get('qacs'))}", qacs_weight, "#222222"),
                ("fc", f"FC={fmt_metric(metric.get('fc'))}", fc_weight, fc_color),
                ("nqs", f"NQS={fmt_metric(metric.get('nqs'))}", "normal", "#222222"),
            ]
            for offset, (metric_name, text, weight, color) in zip((0.030, 0.000, -0.030), metric_lines):
                artist = add_text(
                    f"{mode}_{metric_name}",
                    metrics_x,
                    y + offset,
                    text,
                    ha="left",
                    va="center",
                    fontsize=7.0,
                    fontweight=weight,
                    color=color,
                )
                layout_items["metrics"].append(artist)

    topsis_fm0 = fm_counts["qos_topsis"]["fm0"]
    add_badge(
        "qos_topsis",
        "qos_topsis_data_callout",
        0.627,
        0.286,
        0.430,
        0.036,
        TOPSIS_CALLOUT_TEMPLATE.format(fm0=topsis_fm0),
        "#fff8f8",
        "#8a2f2f",
        "#5a2525",
        4.55,
        linewidth=0.45,
    )

    legend_y = 0.067
    legend_x = 0.248
    legend_patch_1 = Rectangle((legend_x, legend_y - 0.011), 0.032, 0.022, facecolor="#edf5ec", edgecolor="#2e6b3c", linewidth=1.0)
    ax.add_patch(legend_patch_1)
    legend_text_1 = add_text("legend_fm1", legend_x + 0.04, legend_y, "FM=1 functional match", ha="left", va="center", fontsize=6.6)
    legend_patch_2 = Rectangle(
        (legend_x + 0.235, legend_y - 0.011),
        0.032,
        0.022,
        facecolor="#fbefef",
        edgecolor="#8a2f2f",
        linewidth=1.2,
        linestyle=(0, (3, 2)),
    )
    ax.add_patch(legend_patch_2)
    legend_text_2 = add_text("legend_fm0", legend_x + 0.275, legend_y, "FM=0 mismatch", ha="left", va="center", fontsize=6.6)
    layout_items["legend"].extend([legend_patch_1, legend_patch_2, legend_text_1, legend_text_2])

    note = add_text(
        "score_semantics_note",
        0.5,
        0.026,
        SCORE_NOTE_TEXT.replace(" In QoS-Hybrid", "\nIn QoS-Hybrid"),
        ha="center",
        va="center",
        fontsize=5.25,
        color="#333333",
        linespacing=1.05,
    )
    layout_items["note"] = note

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    layout_warnings = validate_rendered_layout(fig, renderer, layout_items)

    pdf_path = out_dir / f"{OUTPUT_BASENAME}.pdf"
    svg_path = out_dir / f"{OUTPUT_BASENAME}.svg"
    png_path = out_dir / f"{OUTPUT_BASENAME}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return {
        "pdf": str(pdf_path.resolve()),
        "svg": str(svg_path.resolve()),
        "png": str(png_path.resolve()),
        "layout_validation_status": "passed",
        "layout_warnings": layout_warnings,
    }


def write_data_csv(path: Path, selected_query: str, query_text: str, query_subtasks: dict[int, str], rows: list[SelectedRecord], metrics: dict[tuple[str, str], dict[str, Any]]) -> None:
    columns = [
        "query_id",
        "query_text_or_goal",
        "subtask_index",
        "subtask_text",
        "mode_display",
        "raw_mode",
        "api_id_or_endpoint_id",
        "tool_name",
        "endpoint_name",
        "original_api_name",
        "display_api_name",
        "display_name_rule",
        "short_api_name",
        "functional_match",
        "score_label",
        "score_value",
        "response_time",
        "throughput",
        "availability",
        "query_qacs",
        "query_fc",
        "query_nqs",
        "source_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (MODE_ORDER.index(item.mode), item.subtask_index)):
            metric = metrics.get((selected_query, row.mode), {})
            writer.writerow(
                {
                    "query_id": selected_query,
                    "query_text_or_goal": query_text,
                    "subtask_index": row.subtask_index,
                    "subtask_text": query_subtasks[row.subtask_index],
                    "mode_display": MODE_DISPLAY[row.mode],
                    "raw_mode": row.raw_mode,
                    "api_id_or_endpoint_id": row.api_id,
                    "tool_name": row.tool_name or "",
                    "endpoint_name": row.endpoint_name or "",
                    "original_api_name": row.original_api_name,
                    "display_api_name": row.display_api_name,
                    "display_name_rule": row.display_name_rule,
                    "short_api_name": row.short_api_name.replace("\n", " / "),
                    "functional_match": row.functional_match,
                    "score_label": row.score_label,
                    "score_value": "" if row.score_value is None else f"{row.score_value:.6f}",
                    "response_time": "" if row.response_time is None else row.response_time,
                    "throughput": "" if row.throughput is None else row.throughput,
                    "availability": "" if row.availability is None else row.availability,
                    "query_qacs": "" if metric.get("qacs") is None else metric.get("qacs"),
                    "query_fc": "" if metric.get("fc") is None else metric.get("fc"),
                    "query_nqs": "" if metric.get("nqs") is None else metric.get("nqs"),
                    "source_file": row.source_file,
                }
            )


def write_latex(path: Path, query_id: str) -> None:
    latex = f"""\\begin{{figure*}}[t]
  \\centering
  \\includegraphics[width=\\textwidth]{{fig_workflow_case_study.pdf}}
  \\caption{{Representative workflow case study for query~\\texttt{{{query_id}}}. Each row shows the selected API path for the same three subtasks under one ranking mode; each box reports the selected API, functional-match label, and mode-specific score. QoS-TOPSIS assigns high TOPSIS values to APIs that may not satisfy the subtask, reducing functional coverage, whereas QoS-Hybrid first enforces functional suitability and then applies TOPSIS as a QoS refinement step.}}
  \\label{{fig:workflow_case_study}}
\\end{{figure*}}
"""
    path.write_text(latex, encoding="utf-8")


def output_nonempty(paths: Iterable[Path]) -> None:
    for path in paths:
        if not path.exists():
            raise ValueError(f"Expected output was not created: {path}")
        if path.stat().st_size <= 0:
            raise ValueError(f"Expected output is empty: {path}")


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    serif_font = preferred_serif_font_name()
    if serif_font != "Times New Roman":
        warnings.append("Times New Roman is unavailable; using DejaVu Serif.")
    loaded_files = discover_files(run_dir, warnings)
    detected_sources = detect_source_files(loaded_files)
    (
        query_dirs,
        query_texts,
        subtasks,
        selected_records_raw,
        candidate_records,
        functional_index,
        candidate_by_api,
        metrics,
    ) = build_indexes(loaded_files, run_dir)

    modes_found = sorted({row.mode for row in selected_records_raw})
    missing_modes = [mode for mode in MODE_ORDER if mode not in modes_found]
    if missing_modes:
        raise ValueError(f"Fewer than four required modes found in selected API records; missing {missing_modes}. Found {modes_found}.")

    selected_records = enrich_selected_records(selected_records_raw, candidate_by_api, functional_index)
    selected_query, selection_reason, candidate_scores = select_representative_query(selected_records, subtasks, metrics)
    selected_lookup = build_selected_lookup(selected_records)

    query_subtasks = subtasks.get(selected_query, {})
    if len(query_subtasks) != 3:
        raise ValueError(f"Selected query {selected_query} must have exactly three subtasks; found {len(query_subtasks)}.")
    if set(query_subtasks) != {1, 2, 3}:
        raise ValueError(f"Selected query {selected_query} subtasks must be indexed 1, 2, 3; found {sorted(query_subtasks)}.")

    selected_rows: list[SelectedRecord] = []
    for mode in MODE_ORDER:
        for subtask_index in (1, 2, 3):
            row = selected_lookup.get((selected_query, mode, subtask_index))
            if row is None:
                raise ValueError(f"Selected API missing for query {selected_query}, mode {mode}, subtask S{subtask_index}.")
            if row.functional_match is None:
                raise ValueError(
                    f"Functional match label could not be determined for query {selected_query}, "
                    f"mode {mode}, subtask S{subtask_index}, API {row.api_id}."
                )
            selected_rows.append(row)

    score_status = recover_scores(selected_rows, candidate_records, warnings)

    for row in selected_rows:
        original_name = readable_api_name(row.tool_name, row.endpoint_name, row.api_id)
        display_name, display_rule = clean_api_display_name(row.tool_name, row.endpoint_name, original_name)
        row.original_api_name = original_name
        row.display_api_name = display_name
        row.display_name_rule = display_rule
        row.short_api_name = wrap_two_lines(display_name, width=24)
    validate_names(selected_rows)

    for mode in MODE_ORDER:
        if (selected_query, mode) not in metrics:
            warnings.append(f"Row-level QACS/FC/NQS metrics not found for {selected_query}/{mode}; row metrics omitted.")

    query_text = query_texts.get(selected_query) or ""
    selected_query_label = concise_case_subtitle(query_text) or selected_query

    fm_counts = mode_functional_counts(selected_rows)
    highest_modes = highest_qacs_modes(selected_query, metrics)

    figure_paths = draw_case_study_figure(selected_query, query_text, query_subtasks, selected_rows, metrics, out_dir)
    data_csv = out_dir / f"{OUTPUT_BASENAME}_data.csv"
    meta_json = out_dir / f"{OUTPUT_BASENAME}_meta.json"
    latex_path = out_dir / f"{OUTPUT_BASENAME}_latex_snippet.tex"

    write_data_csv(data_csv, selected_query, query_text, query_subtasks, selected_rows, metrics)
    write_latex(latex_path, selected_query)

    used_sources = sorted(
        {
            row.source_file
            for row in selected_rows
        }
        | {row.functional_source_file for row in selected_rows if row.functional_source_file}
        | {metrics[(selected_query, mode)]["source_file"] for mode in MODE_ORDER if (selected_query, mode) in metrics}
        | {relative_to(query_dirs[selected_query] / "0_decomposer.json", run_dir) if selected_query in query_dirs else ""}
    )
    used_sources = [source for source in used_sources if source]
    selected_display_records = [
        {
            "mode": row.mode,
            "mode_display": MODE_DISPLAY[row.mode],
            "subtask_index": row.subtask_index,
            "api_id": row.api_id,
            "original_api_name": row.original_api_name,
            "display_api_name": row.display_api_name,
            "display_name_rule": row.display_name_rule,
        }
        for row in sorted(selected_rows, key=lambda item: (MODE_ORDER.index(item.mode), item.subtask_index))
    ]

    meta = {
        "selected_query_id": selected_query,
        "selected_query_label": selected_query_label,
        "selected_query_text_or_goal": query_text,
        "selected_query_subtasks": {f"S{index}": text for index, text in sorted(query_subtasks.items())},
        "selected_api_display_names": selected_display_records,
        "why_this_query_was_selected": selection_reason,
        "fallback_reason": None if "Fallback" not in selection_reason else selection_reason,
        "candidate_query_selection_scores": sorted(candidate_scores, key=lambda item: query_sort_key(item["query_id"])),
        "mode_functional_match_counts": fm_counts,
        "qos_topsis_has_fm0_selections": fm_counts["qos_topsis"]["fm0"] > 0,
        "qos_hybrid_has_all_fm1_selections": fm_counts["qos_hybrid"]["fm1"] == 3 and fm_counts["qos_hybrid"]["fm0"] == 0,
        "highest_qacs_mode": MODE_DISPLAY[highest_modes[0]] if highest_modes else None,
        "highest_qacs_modes": [MODE_DISPLAY[mode] for mode in highest_modes],
        "final_callout_texts": {
            "qos_topsis": TOPSIS_CALLOUT_TEMPLATE.format(fm0=fm_counts["qos_topsis"]["fm0"]),
            "qos_hybrid": HYBRID_CALLOUT_TEXT,
        },
        "final_score_note_text": SCORE_NOTE_TEXT,
        "detected_source_files": detected_sources,
        "source_files_used": used_sources,
        "score_recovery": score_status,
        "topsis_logged_or_recomputed": {
            "logged_count": len(score_status["topsis"]["logged"]),
            "recomputed_count": len(score_status["topsis"]["recomputed"]),
            "unavailable_count": len(score_status["topsis"]["unavailable"]),
        },
        "normalized_qos_logged_or_recomputed": {
            "logged_count": len(score_status["normalized_qos"]["logged"]),
            "recomputed_count": len(score_status["normalized_qos"]["recomputed"]),
            "unavailable_count": len(score_status["normalized_qos"]["unavailable"]),
        },
        "warnings": warnings,
        "layout_warnings": figure_paths.get("layout_warnings", []),
        "layout_validation_status": figure_paths.get("layout_validation_status", "unknown"),
        "output_file_paths": {
            "pdf": figure_paths["pdf"],
            "svg": figure_paths["svg"],
            "png": figure_paths["png"],
            "data_csv": str(data_csv.resolve()),
            "meta_json": str(meta_json.resolve()),
            "latex_snippet": str(latex_path.resolve()),
            "script": str(Path(__file__).resolve()),
        },
    }
    meta_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    output_nonempty(
        [
            Path(figure_paths["pdf"]),
            Path(figure_paths["svg"]),
            Path(figure_paths["png"]),
            data_csv,
            meta_json,
            latex_path,
        ]
    )

    print("Detected source files:")
    for category, files in detected_sources.items():
        print(f"  {category}: {len(files)}")
        for source in files[:12]:
            print(f"    - {source}")
        if len(files) > 12:
            print(f"    ... {len(files) - 12} more listed in metadata")
    print()
    print(f"Selected query: {selected_query}")
    print(f"Selected query label: {selected_query_label}")
    print(f"Selected query goal: {query_text}")
    print(f"Selection reason: {selection_reason}")
    print("Selected APIs by mode:")
    for mode in MODE_ORDER:
        parts = []
        for subtask_index in (1, 2, 3):
            row = selected_lookup[(selected_query, mode, subtask_index)]
            parts.append(f"S{subtask_index}: {row.api_id} (FM={row.functional_match}, {score_text(row)})")
        print(f"  {MODE_DISPLAY[mode]}: " + " | ".join(parts))
    print("FM=1/FM=0 counts by mode:")
    for mode in MODE_ORDER:
        counts = fm_counts[mode]
        print(f"  {MODE_DISPLAY[mode]}: FM=1 {counts['fm1']}, FM=0 {counts['fm0']}")
    print("QACS/FC/NQS by mode:")
    for mode in MODE_ORDER:
        metric = metrics.get((selected_query, mode), {})
        print(
            f"  {MODE_DISPLAY[mode]}: "
            f"QACS={fmt_metric(metric.get('qacs'))}, "
            f"FC={fmt_metric(metric.get('fc'))}, "
            f"NQS={fmt_metric(metric.get('nqs'))}"
        )
    print("Output paths:")
    for label, path in meta["output_file_paths"].items():
        print(f"  {label}: {path}")
    print(f"Validation status: {meta['layout_validation_status']}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - provide a clear CLI failure.
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
