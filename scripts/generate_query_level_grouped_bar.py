#!/usr/bin/env python3
"""Generate a grouped query-level QoS-adjusted score bar chart.

Example:
    python scripts/generate_query_level_grouped_bar.py \
        --results-dir results/logs/RUNS_MAY_31_NEW_5/fireworks_gpt-oss-120b

Outputs are written to <results-dir>/figures by default:
    query_level_qos_adjusted_score_by_mode_grouped_bar.png
    query_level_qos_adjusted_score_matrix.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

# These optional pandas accelerators can be binary-incompatible with the
# local NumPy install; pandas falls back cleanly when they are unavailable.
sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)

import pandas as pd


MODE_ORDER = ["No-QoS", "QoS-Pure-LLM", "QoS-TOPSIS", "QoS-Hybrid"]
MODE_COLORS = {
    "No-QoS": "#0B7DB4",
    "QoS-Pure-LLM": "#E9A300",
    "QoS-TOPSIS": "#0AA177",
    "QoS-Hybrid": "#C875A7",
}
MODE_VALUE_ALIASES = {
    "noqos": "No-QoS",
    "no_qos": "No-QoS",
    "qos_pure_llm": "QoS-Pure-LLM",
    "pure_llm": "QoS-Pure-LLM",
    "qos_topsis": "QoS-TOPSIS",
    "topsis": "QoS-TOPSIS",
    "qos_hybrid": "QoS-Hybrid",
    "hybrid": "QoS-Hybrid",
}
MODE_KEY_CANDIDATES = {
    "mode",
    "evaluation_mode",
    "qos_mode",
    "composition_mode",
    "method",
    "variant",
    "selection_mode",
    "pipeline_mode",
}
SCORE_FIELDS = [
    "qos_adjusted_composition_score",
    "qos_adjusted_score",
    "final_qos_adjusted_score",
    "final_composition_score",
    "composition_score",
    "final_score",
]
PREFERRED_FILE_PATTERNS = [
    "composition_qos",
    "composition",
    "qos",
    "evaluation",
    "eval",
    "metrics",
    "final",
]
QUERY_DIR_RE = re.compile(r"^(q\d{2})_.*$", re.IGNORECASE)
QUERY_ID_RE = re.compile(r"^q?(\d{1,2})$", re.IGNORECASE)
TIMESTAMP_RE = re.compile(r"q\d{2}_(\d{8}T\d{6})", re.IGNORECASE)
SUPPORTED_SUFFIXES = {".csv", ".json", ".jsonl"}
OUTPUT_STEM = "query_level_qos_adjusted_score_by_mode_grouped_bar"
CSV_NAME = "query_level_qos_adjusted_score_matrix.csv"


@dataclass(frozen=True)
class ScoreRecord:
    mode: str
    score: float
    source_path: Path
    score_field: str
    file_rank: tuple[int, int, str]
    field_priority: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a grouped bar chart of query-level QoS-Adjusted "
            "Composition Score by evaluation mode."
        )
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        type=Path,
        help="Directory containing qXX_* query run folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to <results-dir>/figures.",
    )
    parser.add_argument(
        "--strict-one-run-per-query",
        action="store_true",
        help="Fail if more than one qXX_* folder exists for any query id.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow missing mode scores and write NaN values to the CSV.",
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        help="Optional query ids to plot, for example --queries q01 q02 q03.",
    )
    parser.add_argument("--png-dpi", type=int, default=300, help="PNG export resolution.")
    return parser.parse_args()


def find_query_dirs(results_dir: Path) -> dict[str, list[Path]]:
    if not results_dir.exists() or not results_dir.is_dir():
        raise ValueError(f"Results directory does not exist or is not a directory: {results_dir}")

    query_dirs: dict[str, list[Path]] = {}
    for path in results_dir.iterdir():
        if not path.is_dir():
            continue
        match = QUERY_DIR_RE.match(path.name)
        if not match:
            continue
        query_id = match.group(1).lower()
        query_dirs.setdefault(query_id, []).append(path)

    if not query_dirs:
        raise ValueError(f"No qXX_* query directories found under {results_dir}")
    return query_dirs


def select_latest_query_dirs(
    query_dirs: dict[str, list[Path]],
    strict_one_run_per_query: bool = False,
) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    for query_id, paths in sorted(query_dirs.items(), key=lambda item: query_sort_key(item[0])):
        ordered_paths = sorted(paths, key=lambda path: path.name)
        if len(ordered_paths) == 1:
            selected[query_id] = ordered_paths[0]
            continue

        if strict_one_run_per_query:
            options = ", ".join(str(path) for path in ordered_paths)
            raise ValueError(f"Duplicate folders found for {query_id}: {options}")

        timestamped = [(parse_timestamp_from_name(path.name), path) for path in ordered_paths]
        timestamped = [(stamp, path) for stamp, path in timestamped if stamp is not None]
        if timestamped:
            chosen = max(timestamped, key=lambda item: (item[0], item[1].name))[1]
            basis = "latest timestamp in folder name"
        else:
            chosen = max(ordered_paths, key=lambda path: (path.stat().st_mtime, path.name))
            basis = "latest modification time"

        options = ", ".join(str(path) for path in ordered_paths)
        print(
            f"Warning: duplicate folders for {query_id}; selected {chosen} by {basis}. Candidates: {options}",
            file=sys.stderr,
        )
        selected[query_id] = chosen
    return selected


def normalize_mode(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    key = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    compact = key.replace("_", "")
    return MODE_VALUE_ALIASES.get(key) or MODE_VALUE_ALIASES.get(compact)


def normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def extract_records_from_csv(path: Path) -> list[ScoreRecord]:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        warn(f"Skipping unreadable CSV {path}: {exc}")
        return []

    if frame.empty:
        return []

    normalized_columns = {column: normalize_key(column) for column in frame.columns}
    mode_columns = [column for column, key in normalized_columns.items() if key in MODE_KEY_CANDIDATES]
    score_columns = [
        column
        for column, key in normalized_columns.items()
        if key in SCORE_FIELDS
    ]
    if not mode_columns or not score_columns:
        return []

    records: list[ScoreRecord] = []
    rank = artifact_rank(path)
    for _, row in frame.iterrows():
        mode = first_normalized_mode(row.get(column) for column in mode_columns)
        if mode is None:
            continue
        score_value, score_field, field_priority = first_score_value(
            {normalized_columns[column]: row.get(column) for column in score_columns}
        )
        if score_value is None or score_field is None or field_priority is None:
            continue
        records.append(
            ScoreRecord(
                mode=mode,
                score=score_value,
                source_path=path,
                score_field=score_field,
                file_rank=rank,
                field_priority=field_priority,
            )
        )
    return records


def extract_records_from_json(path: Path) -> list[ScoreRecord]:
    payloads: list[Any] = []
    try:
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        payloads.append(json.loads(stripped))
                    except json.JSONDecodeError as exc:
                        warn(f"Skipping invalid JSONL line {line_number} in {path}: {exc}")
        else:
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        warn(f"Skipping unreadable JSON artifact {path}: {exc}")
        return []

    records: list[ScoreRecord] = []
    rank = artifact_rank(path)
    for payload in payloads:
        records.extend(records_from_json_value(payload, path, rank))
    return records


def collect_scores_for_query(query_dir: Path) -> dict[str, ScoreRecord]:
    candidate_files = sorted(
        (
            path
            for path in query_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        ),
        key=artifact_rank,
    )
    if not candidate_files:
        warn(f"No CSV, JSON, or JSONL artifacts found under {query_dir}")
        return {}

    records: list[ScoreRecord] = []
    for path in candidate_files:
        if path.suffix.lower() == ".csv":
            records.extend(extract_records_from_csv(path))
        else:
            records.extend(extract_records_from_json(path))

    if not records:
        warn(f"No final QoS-adjusted composition score records extracted from {query_dir}")
        return {}

    selected: dict[str, ScoreRecord] = {}
    for record in sorted(records, key=record_sort_key):
        existing = selected.get(record.mode)
        if existing is None:
            selected[record.mode] = record
            continue
        if not math.isclose(existing.score, record.score, rel_tol=1e-9, abs_tol=1e-12):
            warn(
                "Conflicting score for "
                f"{query_dir.name} {record.mode}; keeping {existing.score:g} from "
                f"{existing.source_path} and ignoring {record.score:g} from {record.source_path}."
            )
    return selected


def build_score_matrix(
    query_dirs: dict[str, Path],
    queries: list[str] | None = None,
    allow_missing: bool = False,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    if queries:
        missing_queries = [query_id for query_id in queries if query_id not in query_dirs]
        if missing_queries:
            available = ", ".join(sorted(query_dirs, key=query_sort_key))
            raise ValueError(
                f"Requested queries were not found: {', '.join(missing_queries)}. Available: {available}"
            )
        active_query_ids = queries
    else:
        active_query_ids = sorted(query_dirs, key=query_sort_key)

    rows: list[dict[str, float | str]] = []
    selected_folders: dict[str, Path] = {}
    extracted_record_count = 0
    missing_modes_by_query: dict[str, list[str]] = {}

    for query_id in active_query_ids:
        query_dir = query_dirs[query_id]
        selected_folders[query_id] = query_dir
        scores = collect_scores_for_query(query_dir)
        extracted_record_count += len(scores)
        row: dict[str, float | str] = {"query_id": query_id}
        missing_modes: list[str] = []
        for mode in MODE_ORDER:
            record = scores.get(mode)
            if record is None:
                row[mode] = math.nan
                missing_modes.append(mode)
            else:
                row[mode] = record.score
        if missing_modes:
            missing_modes_by_query[query_id] = missing_modes
        rows.append(row)

    if extracted_record_count == 0:
        raise ValueError("No score rows could be extracted from the selected query directories.")

    if missing_modes_by_query and not allow_missing:
        details = "; ".join(
            f"{query_id}: {', '.join(modes)}"
            for query_id, modes in sorted(missing_modes_by_query.items(), key=lambda item: query_sort_key(item[0]))
        )
        raise ValueError(
            "Missing required mode scores. Re-run with --allow-missing to plot available values. "
            f"Missing: {details}"
        )

    matrix = pd.DataFrame(rows, columns=["query_id", *MODE_ORDER])
    return matrix, selected_folders


def plot_grouped_bar(frame: pd.DataFrame, output_dir: Path, png_dpi: int = 300) -> Path:
    configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    query_ids = frame["query_id"].tolist()
    fig_width = max(13.0, min(16.0, 0.58 * len(query_ids) + 5.4))
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))
    x_positions = list(range(len(query_ids)))
    width = 0.20
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

    for offset, mode in zip(offsets, MODE_ORDER):
        values = pd.to_numeric(frame[mode], errors="coerce").tolist()
        ax.bar(
            [position + offset for position in x_positions],
            values,
            width,
            label=mode,
            color=MODE_COLORS[mode],
            edgecolor="#FFFFFF",
            linewidth=0.5,
        )

    ax.set_xlabel("Query ID")
    ax.set_ylabel("QoS-Adjusted Composition Score")
    ax.set_xticks(x_positions)
    rotation = 35 if len(query_ids) > 15 or any(len(query_id) > 4 for query_id in query_ids) else 0
    ax.set_xticklabels(query_ids, rotation=rotation, ha="right" if rotation else "center")
    ax.set_ylim(0.0, 1.08)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=4, frameon=False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.65)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    output = output_dir / f"{OUTPUT_STEM}.png"
    fig.savefig(output, dpi=png_dpi, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> int:
    try:
        args = parse_args()
        results_dir = args.results_dir.expanduser().resolve()
        output_dir = resolve_output_dir(results_dir, args.output_dir)
        requested_queries = parse_queries(args.queries)

        query_dirs = find_query_dirs(results_dir)
        selected_query_dirs = select_latest_query_dirs(
            query_dirs,
            strict_one_run_per_query=args.strict_one_run_per_query,
        )
        matrix, selected_folders = build_score_matrix(
            selected_query_dirs,
            queries=requested_queries,
            allow_missing=args.allow_missing,
        )

        csv_path = output_dir / CSV_NAME
        matrix.to_csv(csv_path, index=False)
        plot_path = plot_grouped_bar(matrix, output_dir, png_dpi=args.png_dpi)
        output_paths = [plot_path, csv_path]

        print(f"Queries plotted: {len(matrix)}")
        print("Selected query folders:")
        for query_id, path in selected_folders.items():
            print(f"  {query_id}: {path}")
        print("Output files:")
        for path in output_paths:
            print(f"  {path}")
        print("Rerun command:")
        print(f"  {build_rerun_command(args)}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def parse_timestamp_from_name(name: str) -> datetime | None:
    match = TIMESTAMP_RE.search(name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")
    except ValueError:
        return None


def query_sort_key(query_id: str) -> tuple[int, str]:
    match = re.search(r"\d+", query_id)
    if not match:
        return (10_000, query_id)
    return (int(match.group()), query_id)


def parse_queries(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    parsed: list[str] = []
    seen: set[str] = set()
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if not token:
                continue
            match = QUERY_ID_RE.fullmatch(token)
            if not match:
                raise ValueError(f"Invalid query id {token!r}; use q01, 01, or 1.")
            query_id = f"q{int(match.group(1)):02d}"
            if query_id not in seen:
                parsed.append(query_id)
                seen.add(query_id)
    if not parsed:
        raise ValueError("--queries was provided but no query ids were parsed.")
    return parsed


def resolve_output_dir(results_dir: Path, output_dir: Path | None) -> Path:
    path = results_dir / "figures" if output_dir is None else output_dir.expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def configure_matplotlib() -> None:
    cache_root = Path(tempfile.gettempdir()) / "autollmcompose_figures"
    mpl_cache_dir = cache_root / "matplotlib"
    xdg_cache_dir = cache_root / "xdg"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_cache_dir.as_posix())
    os.environ.setdefault("XDG_CACHE_HOME", xdg_cache_dir.as_posix())


def artifact_rank(path: Path) -> tuple[int, int, str]:
    lower_name = path.name.lower()
    lower_path = path.as_posix().lower()
    pattern_scores = [
        index
        for index, pattern in enumerate(PREFERRED_FILE_PATTERNS)
        if pattern in lower_name or pattern in lower_path
    ]
    first_pattern = min(pattern_scores) if pattern_scores else len(PREFERRED_FILE_PATTERNS)
    if "summary" in lower_name:
        first_pattern -= 2
    if "rows" in lower_name:
        first_pattern -= 1
    if "candidate_api_rankings" in lower_name or "planner_selection" in lower_name:
        first_pattern += 4
    suffix_rank = {".csv": 0, ".json": 1, ".jsonl": 2}.get(path.suffix.lower(), 9)
    return (first_pattern, suffix_rank, path.as_posix())


def first_normalized_mode(values: Iterable[Any]) -> str | None:
    for value in values:
        mode = normalize_mode(value)
        if mode is not None:
            return mode
    return None


def first_score_value(values_by_normalized_key: dict[str, Any]) -> tuple[float | None, str | None, int | None]:
    for priority, field in enumerate(SCORE_FIELDS):
        if field not in values_by_normalized_key:
            continue
        score = finite_float_or_none(values_by_normalized_key[field])
        if score is not None:
            return score, field, priority
    return None, None, None


def finite_float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    return score


def records_from_json_value(
    value: Any,
    source_path: Path,
    rank: tuple[int, int, str],
    inherited_mode: str | None = None,
) -> list[ScoreRecord]:
    records: list[ScoreRecord] = []
    if isinstance(value, dict):
        normalized = {normalize_key(key): item for key, item in value.items()}
        direct_mode = first_normalized_mode(
            item for key, item in normalized.items() if key in MODE_KEY_CANDIDATES
        )
        active_mode = direct_mode or inherited_mode
        score_value, score_field, field_priority = first_score_value(normalized)
        if active_mode and score_value is not None and score_field is not None and field_priority is not None:
            records.append(
                ScoreRecord(
                    mode=active_mode,
                    score=score_value,
                    source_path=source_path,
                    score_field=score_field,
                    file_rank=rank,
                    field_priority=field_priority,
                )
            )

        for raw_key, child in value.items():
            child_mode = normalize_mode(raw_key) or active_mode
            records.extend(records_from_json_value(child, source_path, rank, child_mode))
    elif isinstance(value, list):
        for child in value:
            records.extend(records_from_json_value(child, source_path, rank, inherited_mode))
    return records


def record_sort_key(record: ScoreRecord) -> tuple[tuple[int, int, str], int, str]:
    return (record.file_rank, record.field_priority, record.source_path.as_posix())


def warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)


def build_rerun_command(args: argparse.Namespace) -> str:
    parts = [
        "python",
        "scripts/generate_query_level_grouped_bar.py",
        "--results-dir",
        str(args.results_dir),
    ]
    if args.output_dir is not None:
        parts.extend(["--output-dir", str(args.output_dir)])
    if args.strict_one_run_per_query:
        parts.append("--strict-one-run-per-query")
    if args.allow_missing:
        parts.append("--allow-missing")
    if args.queries:
        parts.append("--queries")
        parts.extend(args.queries)
    if args.png_dpi != 300:
        parts.extend(["--png-dpi", str(args.png_dpi)])
    return " ".join(shlex.quote(part) for part in parts)


if __name__ == "__main__":
    raise SystemExit(main())
