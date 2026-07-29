#!/usr/bin/env python3
"""Create IEEE-ready Figure 3: query-level winner/tie heatmap.

Example:
    python scripts/make_fig3_winner_tie_heatmap.py \
        --run-dir results/logs/RUNS_MAY_31_NEW_5/fireworks_gpt-oss-120b \
        --out-dir paper/figures \
        --tie-tol 1e-6
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


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
    "noqos": "no_qos",
    "no_qos": "no_qos",
    "qos_pure_llm": "qos_pure_llm",
    "pure_llm": "qos_pure_llm",
    "qos_topsis": "qos_topsis",
    "topsis": "qos_topsis",
    "qos_hybrid": "qos_hybrid",
    "hybrid": "qos_hybrid",
}
QUERY_IDS = tuple(f"q{idx:02d}" for idx in range(1, 16))
QUERY_ID_RE = re.compile(r"^q?(\d{1,2})$", re.IGNORECASE)
QUERY_DIR_RE = re.compile(r"(q\d{2})(?:_|$)", re.IGNORECASE)
SUPPORTED_SUFFIXES = {".csv", ".json", ".jsonl"}
QUERY_KEYS = {
    "query_id",
    "query",
    "queryid",
    "official_query",
    "official_query_id",
}
MODE_KEYS = {
    "mode",
    "selection_mode",
    "evaluation_mode",
    "qos_mode",
    "composition_mode",
    "method",
    "variant",
}
SCORE_KEYS = (
    "qos_adjusted_composition_score",
    "qacs",
    "qacs_score",
    "final_qacs",
    "qos_adjusted_score",
    "final_qos_adjusted_score",
    "final_composition_score",
    "composition_score",
    "final_score",
)
EXCLUDE_PATH_PATTERNS = (
    "figures",
    "weigh_sensitivity",
    "weight_sensitivity",
    "sensitivity",
    "ranking_similarity",
    "rank_similarity",
    "candidate_api_rankings",
    "dashboard",
)
STATUS_NOT_BEST = 0
STATUS_UNIQUE_BEST = 1
STATUS_TIED_BEST = 2
STATUS_LABELS = {
    STATUS_NOT_BEST: "not_best",
    STATUS_UNIQUE_BEST: "unique_best",
    STATUS_TIED_BEST: "tied_best",
}
EXPECTED_PURE_LLM_TIES = {"q06", "q08", "q11", "q14"}


@dataclass(frozen=True)
class ScoreRow:
    query_id: str
    mode: str
    score: float
    source_path: Path
    row_ref: str
    score_key: str


@dataclass(frozen=True)
class CandidateFile:
    path: Path
    rows: tuple[ScoreRow, ...]
    rank: tuple[int, int, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate IEEE Figure 3 winner/tie heatmap from composition-level QACS results."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("results/logs/RUNS_MAY_31_NEW_5/fireworks_gpt-oss-120b"),
        help="Run directory containing q01-q15 artifacts.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("paper/figures"),
        help="Directory for Figure 3 outputs.",
    )
    parser.add_argument(
        "--tie-tol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for tied-best QACS values.",
    )
    return parser.parse_args()


def normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def normalize_query_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = QUERY_ID_RE.fullmatch(text)
    if not match:
        return None
    number = int(match.group(1))
    if number < 1:
        return None
    return f"q{number:02d}"


def normalize_mode(value: Any) -> str | None:
    if value is None:
        return None
    key = normalize_key(value)
    if not key:
        return None
    compact = key.replace("_", "")
    return MODE_ALIASES.get(key) or MODE_ALIASES.get(compact)


def finite_float(value: Any) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def query_from_path(path: Path) -> str | None:
    for part in path.parts:
        match = QUERY_DIR_RE.search(part)
        if match:
            return match.group(1).lower()
    return None


def mode_from_path(path: Path) -> str | None:
    for part in reversed(path.parts):
        mode = normalize_mode(part)
        if mode:
            return mode
    return None


def first_normalized_query(values: Iterable[Any]) -> str | None:
    for value in values:
        query_id = normalize_query_id(value)
        if query_id:
            return query_id
    return None


def first_normalized_mode(values: Iterable[Any]) -> str | None:
    for value in values:
        mode = normalize_mode(value)
        if mode:
            return mode
    return None


def first_score(values_by_key: dict[str, Any]) -> tuple[float | None, str | None]:
    for key in SCORE_KEYS:
        if key not in values_by_key:
            continue
        score = finite_float(values_by_key[key])
        if score is not None:
            return score, key
    return None, None


def should_skip_file(path: Path) -> bool:
    lower = path.as_posix().lower()
    return any(pattern in lower for pattern in EXCLUDE_PATH_PATTERNS)


def discover_artifact_files(run_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in run_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_SUFFIXES
            and not should_skip_file(path)
        ),
        key=artifact_rank,
    )


def artifact_rank(path: Path) -> tuple[int, int, str]:
    lower = path.as_posix().lower()
    suffix_rank = {".csv": 0, ".json": 1, ".jsonl": 2}.get(path.suffix.lower(), 9)
    rank = 100
    if lower.endswith("summary/all_15_query_composition_results.csv"):
        rank = 0
    elif "all_15" in lower and "composition" in lower:
        rank = 5
    elif "consolidated" in lower and "composition" in lower:
        rank = 10
    elif "/summary/" in lower and "composition" in lower:
        rank = 15
    elif "composition_qos_eval_rows" in lower:
        rank = 25
    elif "composition_qos_eval_summary" in lower:
        rank = 35
    elif "composition" in lower and ("qos" in lower or "eval" in lower):
        rank = 45
    elif "metrics" in lower or "final" in lower:
        rank = 60
    return (rank, suffix_rank, lower)


def extract_rows_from_csv(path: Path) -> list[ScoreRow]:
    rows: list[ScoreRow] = []
    inherited_query = query_from_path(path)
    inherited_mode = mode_from_path(path)
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return []
            normalized_columns = {column: normalize_key(column) for column in reader.fieldnames}
            for row_number, raw_row in enumerate(reader, start=2):
                values_by_key = {
                    normalized_columns[column]: raw_row.get(column)
                    for column in reader.fieldnames
                }
                query_id = first_normalized_query(
                    value for key, value in values_by_key.items() if key in QUERY_KEYS
                ) or inherited_query
                mode = first_normalized_mode(
                    value for key, value in values_by_key.items() if key in MODE_KEYS
                ) or inherited_mode
                score, score_key = first_score(values_by_key)
                if query_id and mode and score is not None and score_key:
                    rows.append(ScoreRow(query_id, mode, score, path, f"line {row_number}", score_key))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        warn(f"Skipping unreadable CSV {path}: {exc}")
    return rows


def extract_rows_from_json(path: Path) -> list[ScoreRow]:
    rows: list[ScoreRow] = []
    inherited_query = query_from_path(path)
    inherited_mode = mode_from_path(path)
    try:
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        payload = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        warn(f"Skipping invalid JSONL line {line_number} in {path}: {exc}")
                        continue
                    rows.extend(
                        rows_from_json_value(
                            payload,
                            path,
                            f"line {line_number}",
                            inherited_query,
                            inherited_mode,
                        )
                    )
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.extend(rows_from_json_value(payload, path, "$", inherited_query, inherited_mode))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        warn(f"Skipping unreadable JSON artifact {path}: {exc}")
    return rows


def rows_from_json_value(
    value: Any,
    source_path: Path,
    location: str,
    inherited_query: str | None,
    inherited_mode: str | None,
) -> list[ScoreRow]:
    rows: list[ScoreRow] = []
    if isinstance(value, dict):
        normalized = {normalize_key(key): child for key, child in value.items()}
        query_id = first_normalized_query(
            child for key, child in normalized.items() if key in QUERY_KEYS
        ) or inherited_query
        mode = first_normalized_mode(
            child for key, child in normalized.items() if key in MODE_KEYS
        ) or inherited_mode
        score, score_key = first_score(normalized)
        if query_id and mode and score is not None and score_key:
            rows.append(ScoreRow(query_id, mode, score, source_path, location, score_key))

        for raw_key, child in value.items():
            key_text = str(raw_key)
            child_query = normalize_query_id(key_text) or query_id
            child_mode = normalize_mode(key_text) or mode
            child_location = f"{location}.{key_text}" if location else key_text
            rows.extend(rows_from_json_value(child, source_path, child_location, child_query, child_mode))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            rows.extend(
                rows_from_json_value(
                    child,
                    source_path,
                    f"{location}[{index}]",
                    inherited_query,
                    inherited_mode,
                )
            )
    return rows


def load_candidates(run_dir: Path) -> list[CandidateFile]:
    if not run_dir.exists() or not run_dir.is_dir():
        raise ValueError(f"Run directory does not exist or is not a directory: {run_dir}")
    candidates: list[CandidateFile] = []
    for path in discover_artifact_files(run_dir):
        rows = extract_rows_from_csv(path) if path.suffix.lower() == ".csv" else extract_rows_from_json(path)
        rows = [
            row
            for row in rows
            if row.query_id in QUERY_IDS and row.mode in MODE_ORDER
        ]
        if rows:
            candidates.append(CandidateFile(path=path, rows=tuple(rows), rank=artifact_rank(path)))
    if not candidates:
        raise ValueError(f"No composition-level QACS rows found under {run_dir}")
    return candidates


def select_score_rows(candidates: list[CandidateFile]) -> tuple[list[ScoreRow], list[Path], str]:
    full_candidates = [
        candidate
        for candidate in candidates
        if is_exact_full_scope(candidate.rows)
    ]
    if full_candidates:
        chosen = sorted(full_candidates, key=lambda candidate: candidate.rank)[0]
        return list(chosen.rows), [chosen.path], "single consolidated/full-scope file"

    per_query: dict[str, CandidateFile] = {}
    ambiguous: dict[str, list[CandidateFile]] = {}
    for query_id in QUERY_IDS:
        query_candidates = [
            candidate
            for candidate in candidates
            if is_exact_single_query_scope(candidate.rows, query_id)
        ]
        if not query_candidates:
            continue
        ordered = sorted(query_candidates, key=lambda candidate: candidate.rank)
        per_query[query_id] = ordered[0]
        if len(ordered) > 1:
            ambiguous[query_id] = ordered

    if len(per_query) == len(QUERY_IDS):
        for query_id, options in ambiguous.items():
            warn(
                f"Multiple per-query QACS files for {query_id}; selected {options[0].path}. "
                f"Other candidates: {', '.join(str(item.path) for item in options[1:])}"
            )
        rows: list[ScoreRow] = []
        sources: list[Path] = []
        for query_id in QUERY_IDS:
            candidate = per_query[query_id]
            rows.extend(candidate.rows)
            sources.append(candidate.path)
        return rows, sources, "one per-query composition-level file per query"

    duplicate_details = duplicate_diagnostics(candidates)
    candidate_details = "\n".join(
        f"  {candidate.path}: {len(candidate.rows)} extracted rows"
        for candidate in sorted(candidates, key=lambda candidate: candidate.rank)[:20]
    )
    missing_queries = [query_id for query_id in QUERY_IDS if query_id not in per_query]
    message = (
        "Could not identify a unique 15-query x 4-mode QACS table. "
        "No single consolidated full-scope file was usable, and the per-query fallback was incomplete."
    )
    if missing_queries:
        message += f"\nMissing per-query sources for: {', '.join(missing_queries)}"
    if duplicate_details:
        message += f"\nDuplicate extracted query-mode rows seen across candidate files:\n{duplicate_details}"
    message += f"\nCandidate files inspected:\n{candidate_details}"
    raise ValueError(message)


def is_exact_full_scope(rows: Iterable[ScoreRow]) -> bool:
    row_list = list(rows)
    pairs = [(row.query_id, row.mode) for row in row_list]
    return len(row_list) == len(QUERY_IDS) * len(MODE_ORDER) and set(pairs) == expected_pairs() and len(pairs) == len(set(pairs))


def is_exact_single_query_scope(rows: Iterable[ScoreRow], query_id: str) -> bool:
    row_list = list(rows)
    pairs = [(row.query_id, row.mode) for row in row_list]
    expected = {(query_id, mode) for mode in MODE_ORDER}
    return len(row_list) == len(MODE_ORDER) and set(pairs) == expected and len(pairs) == len(set(pairs))


def expected_pairs() -> set[tuple[str, str]]:
    return {(query_id, mode) for query_id in QUERY_IDS for mode in MODE_ORDER}


def duplicate_diagnostics(candidates: list[CandidateFile]) -> str:
    locations: dict[tuple[str, str], list[ScoreRow]] = {}
    for candidate in candidates:
        for row in candidate.rows:
            locations.setdefault((row.query_id, row.mode), []).append(row)
    lines = []
    for pair, rows in sorted(locations.items()):
        if len(rows) < 2:
            continue
        source_text = "; ".join(f"{row.source_path} ({row.row_ref}, {row.score:g})" for row in rows[:8])
        if len(rows) > 8:
            source_text += f"; ... {len(rows) - 8} more"
        lines.append(f"  {pair[0]} {pair[1]}: {source_text}")
    return "\n".join(lines)


def validate_and_matrix(rows: list[ScoreRow]) -> tuple[dict[str, dict[str, float]], dict[tuple[str, str], ScoreRow]]:
    by_pair: dict[tuple[str, str], ScoreRow] = {}
    duplicates: dict[tuple[str, str], list[ScoreRow]] = {}
    for row in rows:
        pair = (row.query_id, row.mode)
        if pair in by_pair:
            duplicates.setdefault(pair, [by_pair[pair]]).append(row)
        else:
            by_pair[pair] = row
    if duplicates:
        detail = "\n".join(
            f"  {query_id} {mode}: "
            + "; ".join(f"{row.source_path} ({row.row_ref}, {row.score:g})" for row in dup_rows)
            for (query_id, mode), dup_rows in sorted(duplicates.items())
        )
        raise ValueError(f"Final extracted table has duplicate query-mode rows:\n{detail}")

    missing = sorted(expected_pairs() - set(by_pair), key=lambda pair: (pair[0], MODE_ORDER.index(pair[1])))
    extra = sorted(set(by_pair) - expected_pairs(), key=lambda pair: (pair[0], pair[1]))
    if missing:
        raise ValueError("Missing query-mode QACS rows: " + ", ".join(f"{query_id}/{mode}" for query_id, mode in missing))
    if extra:
        raise ValueError("Unexpected query-mode rows: " + ", ".join(f"{query_id}/{mode}" for query_id, mode in extra))

    matrix = {
        query_id: {mode: by_pair[(query_id, mode)].score for mode in MODE_ORDER}
        for query_id in QUERY_IDS
    }
    return matrix, by_pair


def compute_statuses(
    score_matrix: dict[str, dict[str, float]],
    tie_tol: float,
) -> dict[str, dict[str, int]]:
    statuses: dict[str, dict[str, int]] = {}
    for query_id in QUERY_IDS:
        scores = score_matrix[query_id]
        max_score = max(scores.values())
        tied_modes = [mode for mode in MODE_ORDER if abs(scores[mode] - max_score) <= tie_tol]
        row_status: dict[str, int] = {}
        for mode in MODE_ORDER:
            if mode not in tied_modes:
                row_status[mode] = STATUS_NOT_BEST
            elif len(tied_modes) == 1:
                row_status[mode] = STATUS_UNIQUE_BEST
            else:
                row_status[mode] = STATUS_TIED_BEST
        statuses[query_id] = row_status
    return statuses


def count_statuses(statuses: dict[str, dict[str, int]]) -> tuple[dict[str, int], dict[str, int], list[str]]:
    unique_counts = {mode: 0 for mode in MODE_ORDER}
    tied_counts = {mode: 0 for mode in MODE_ORDER}
    tied_queries: list[str] = []
    for query_id in QUERY_IDS:
        tied_in_query = False
        for mode in MODE_ORDER:
            if statuses[query_id][mode] == STATUS_UNIQUE_BEST:
                unique_counts[mode] += 1
            elif statuses[query_id][mode] == STATUS_TIED_BEST:
                tied_counts[mode] += 1
                tied_in_query = True
        if tied_in_query:
            tied_queries.append(query_id)
    return unique_counts, tied_counts, tied_queries


def validate_expected_outcomes(
    score_matrix: dict[str, dict[str, float]],
    statuses: dict[str, dict[str, int]],
) -> None:
    unique_counts, tied_counts, tied_queries = count_statuses(statuses)
    errors = []
    hybrid_best_or_tied = sum(1 for query_id in QUERY_IDS if statuses[query_id]["qos_hybrid"] > STATUS_NOT_BEST)
    if hybrid_best_or_tied != 15:
        errors.append(f"QoS-Hybrid best/tied-best count is {hybrid_best_or_tied}, expected 15.")
    if unique_counts["qos_hybrid"] != 11:
        errors.append(f"QoS-Hybrid unique-best count is {unique_counts['qos_hybrid']}, expected 11.")
    if tied_counts["qos_hybrid"] != 4:
        errors.append(f"QoS-Hybrid tied-best count is {tied_counts['qos_hybrid']}, expected 4.")
    pure_ties = {query_id for query_id in QUERY_IDS if statuses[query_id]["qos_pure_llm"] == STATUS_TIED_BEST}
    if pure_ties != EXPECTED_PURE_LLM_TIES:
        errors.append(
            "QoS-Pure-LLM tied-best queries are "
            f"{sorted(pure_ties)}, expected {sorted(EXPECTED_PURE_LLM_TIES)}."
        )
    for mode in ("no_qos", "qos_topsis"):
        best_count = unique_counts[mode] + tied_counts[mode]
        if best_count != 0:
            errors.append(f"{MODE_LABELS[mode]} has {best_count} best/tied-best cells, expected 0.")
    if set(tied_queries) != EXPECTED_PURE_LLM_TIES:
        errors.append(f"Tied query IDs are {tied_queries}, expected {sorted(EXPECTED_PURE_LLM_TIES)}.")
    if errors:
        print("Extracted score matrix:", file=sys.stderr)
        print(format_score_matrix(score_matrix), file=sys.stderr)
        raise ValueError("Sanity check failed:\n" + "\n".join(f"- {error}" for error in errors))


def configure_matplotlib() -> None:
    cache_root = Path(tempfile.gettempdir()) / "autollmcompose_figures"
    mpl_cache_dir = cache_root / "matplotlib"
    xdg_cache_dir = cache_root / "xdg"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_cache_dir.as_posix())
    os.environ.setdefault("XDG_CACHE_HOME", xdg_cache_dir.as_posix())


def plot_winner_tie_heatmap(
    statuses: dict[str, dict[str, int]],
    out_dir: Path,
) -> tuple[Path, Path]:
    configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    grid = [[statuses[query_id][mode] for mode in MODE_ORDER] for query_id in QUERY_IDS]
    cmap = ListedColormap(["#F3F4F6", "#0072B2", "#E69F00"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, ax = plt.subplots(figsize=(3.45, 3.05))
    ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")
    ax.set_ylabel("Query ID", fontsize=8)
    ax.set_xticks(range(len(MODE_ORDER)))
    ax.set_xticklabels([MODE_TICK_LABELS[mode] for mode in MODE_ORDER], rotation=0, ha="center")
    ax.set_yticks(range(len(QUERY_IDS)))
    ax.set_yticklabels(QUERY_IDS)
    ax.tick_params(axis="both", which="major", length=0)
    ax.set_xticks([idx - 0.5 for idx in range(1, len(MODE_ORDER))], minor=True)
    ax.set_yticks([idx - 0.5 for idx in range(1, len(QUERY_IDS))], minor=True)
    ax.grid(which="minor", color="#B8BEC6", linewidth=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for y_index, query_id in enumerate(QUERY_IDS):
        for x_index, mode in enumerate(MODE_ORDER):
            value = statuses[query_id][mode]
            if value == STATUS_UNIQUE_BEST:
                ax.text(x_index, y_index, "U", ha="center", va="center", color="white", fontweight="bold", fontsize=8)
            elif value == STATUS_TIED_BEST:
                ax.text(x_index, y_index, "T", ha="center", va="center", color="black", fontweight="bold", fontsize=8)

    legend_handles = [
        Patch(facecolor="#F3F4F6", edgecolor="#B8BEC6", label="not best"),
        Patch(facecolor="#0072B2", edgecolor="#0072B2", label="unique best (U)"),
        Patch(facecolor="#E69F00", edgecolor="#E69F00", label="tied best (T)"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=3,
        frameon=False,
        handlelength=0.9,
        columnspacing=0.7,
        handletextpad=0.35,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.16, right=0.99, top=0.99, bottom=0.31)

    pdf_path = out_dir / "fig3_winner_tie_heatmap.pdf"
    png_path = out_dir / "fig3_winner_tie_heatmap.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def plot_debug_score_heatmap(
    score_matrix: dict[str, dict[str, float]],
    out_dir: Path,
) -> Path:
    configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    grid = [[score_matrix[query_id][mode] for mode in MODE_ORDER] for query_id in QUERY_IDS]
    fig, ax = plt.subplots(figsize=(4.7, 4.2))
    image = ax.imshow(grid, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_title("Debug QACS Score Heatmap", pad=6)
    ax.set_xlabel("Selection mode")
    ax.set_ylabel("Query ID")
    ax.set_xticks(range(len(MODE_ORDER)))
    ax.set_xticklabels([MODE_TICK_LABELS[mode] for mode in MODE_ORDER], rotation=0, ha="center")
    ax.set_yticks(range(len(QUERY_IDS)))
    ax.set_yticklabels(QUERY_IDS)
    ax.set_xticks([idx - 0.5 for idx in range(1, len(MODE_ORDER))], minor=True)
    ax.set_yticks([idx - 0.5 for idx in range(1, len(QUERY_IDS))], minor=True)
    ax.grid(which="minor", color="#C8CDD3", linewidth=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", which="major", length=0)
    for y_index, query_id in enumerate(QUERY_IDS):
        for x_index, mode in enumerate(MODE_ORDER):
            score = score_matrix[query_id][mode]
            text_color = "white" if score > 0.72 else "black"
            ax.text(x_index, y_index, f"{score:.3f}", ha="center", va="center", fontsize=6.1, color=text_color)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    colorbar.set_label("QACS", rotation=270, labelpad=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    output = out_dir / "debug_fig3_qacs_score_heatmap.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def write_scores_csv(
    score_matrix: dict[str, dict[str, float]],
    statuses: dict[str, dict[str, int]],
    out_dir: Path,
) -> Path:
    output = out_dir / "fig3_winner_tie_heatmap_scores.csv"
    header = (
        ["query_id"]
        + [f"{mode}_qacs" for mode in MODE_ORDER]
        + [f"{mode}_status" for mode in MODE_ORDER]
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for query_id in QUERY_IDS:
            writer.writerow(
                [query_id]
                + [format_float(score_matrix[query_id][mode]) for mode in MODE_ORDER]
                + [STATUS_LABELS[statuses[query_id][mode]] for mode in MODE_ORDER]
            )
    return output


def write_summary(
    source_files: list[Path],
    selection_note: str,
    rows_loaded: int,
    score_matrix: dict[str, dict[str, float]],
    statuses: dict[str, dict[str, int]],
    out_dir: Path,
) -> Path:
    unique_counts, tied_counts, tied_queries = count_statuses(statuses)
    output = out_dir / "fig3_winner_tie_heatmap_summary.txt"
    lines = [
        "Figure 3 winner/tie heatmap summary",
        "",
        f"Selection: {selection_note}",
        "Source file(s) used:",
    ]
    lines.extend(f"  {path}" for path in source_files)
    lines.extend(
        [
            "",
            f"Number of rows loaded: {rows_loaded}",
            "",
            "Extracted score matrix:",
            format_score_matrix(score_matrix),
            "",
            "Unique-best counts by mode:",
        ]
    )
    lines.extend(f"  {MODE_LABELS[mode]}: {unique_counts[mode]}" for mode in MODE_ORDER)
    lines.append("")
    lines.append("Tied-best counts by mode:")
    lines.extend(f"  {MODE_LABELS[mode]}: {tied_counts[mode]}" for mode in MODE_ORDER)
    lines.extend(
        [
            "",
            "Tied query IDs:",
            "  " + (", ".join(tied_queries) if tied_queries else "none"),
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def format_float(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def format_score_matrix(score_matrix: dict[str, dict[str, float]]) -> str:
    header = ["query_id", *MODE_ORDER]
    lines = [",".join(header)]
    for query_id in QUERY_IDS:
        lines.append(",".join([query_id, *[format_float(score_matrix[query_id][mode]) for mode in MODE_ORDER]]))
    return "\n".join(lines)


def warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    try:
        run_dir = args.run_dir.expanduser().resolve()
        out_dir = args.out_dir.expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_dir = out_dir.resolve()

        candidates = load_candidates(run_dir)
        selected_rows, source_files, selection_note = select_score_rows(candidates)
        score_matrix, _ = validate_and_matrix(selected_rows)
        statuses = compute_statuses(score_matrix, args.tie_tol)
        validate_expected_outcomes(score_matrix, statuses)

        figure_pdf, figure_png = plot_winner_tie_heatmap(statuses, out_dir)
        scores_csv = write_scores_csv(score_matrix, statuses, out_dir)
        summary_txt = write_summary(
            source_files,
            selection_note,
            len(selected_rows),
            score_matrix,
            statuses,
            out_dir,
        )
        debug_pdf = plot_debug_score_heatmap(score_matrix, out_dir)

        unique_counts, tied_counts, tied_queries = count_statuses(statuses)
        print("Command completed successfully.")
        print(f"Run directory: {run_dir}")
        print(f"Tie tolerance: {args.tie_tol:g}")
        print(f"Rows loaded: {len(selected_rows)}")
        print("Source file(s) used:")
        for path in source_files:
            print(f"  {path}")
        print("Generated outputs:")
        for path in (figure_pdf, figure_png, scores_csv, summary_txt, debug_pdf):
            print(f"  {path}")
        print("Unique-best counts by mode:")
        for mode in MODE_ORDER:
            print(f"  {MODE_LABELS[mode]}: {unique_counts[mode]}")
        print("Tied-best counts by mode:")
        for mode in MODE_ORDER:
            print(f"  {MODE_LABELS[mode]}: {tied_counts[mode]}")
        print("Tied query IDs: " + (", ".join(tied_queries) if tied_queries else "none"))
        print("Sanity check: passed")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
