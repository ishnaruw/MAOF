# Post-Experiment Analysis Scripts

This directory contains deterministic post-processing scripts for completed
AutoLLMCompose experiment runs. Use them after the main experiment pipeline has
finished and the run folder contains one directory per query, named `qXX_*`.

The scripts are intended for research reporting: they read saved experiment
artifacts, reject ambiguous or malformed inputs, and write CSV, JSON, Markdown,
and PNG outputs that can be traced back to source files. They do not invent
missing values, hard-code conclusion numbers, or depend on thesis figure numbers.

Example run folder used below, relative to the `AutoLLMCompose` project root:

```bash
RUN_DIR="results/logs/RUNS_MAY_31_NEW_5/fireworks_gpt-oss-120b"
```

Run commands from the `AutoLLMCompose` project root unless otherwise noted:

```bash
cd AutoLLMCompose
python -m pip install -r requirements.txt
```

The scripts require the project dependencies used by the repository, including
`pandas`, `scipy`, `openpyxl`, and `matplotlib`.

## Recommended Post-Run Order

Run the analysis in this order after the experiment completes:

```bash
python scripts/consolidate_composition_results.py "$RUN_DIR"

python scripts/run_ranking_eval.py "$RUN_DIR"

python scripts/generate_weight_sensitivity_tables.py "$RUN_DIR"

python scripts/generate_research_figures.py "$RUN_DIR" \
  --path-query q02 \
  --panel-queries q01,q02,q03,q04,q05 \
  --panel-queries q06,q07,q08,q09,q10 \
  --panel-queries q11,q12,q13,q14,q15
```

This order matters:

- `consolidate_composition_results.py` creates `<run-folder>/summary`.
- `run_ranking_eval.py` creates `<run-folder>/ranking_eval`.
- `generate_weight_sensitivity_tables.py` consumes both `summary` and
  `ranking_eval`.
- `generate_research_figures.py` consumes per-query evaluation JSON artifacts
  and, for ranking heatmaps, `ranking_eval`.

If a run contains additional exploratory query folders that should not be part
of the official thesis set, filter explicitly:

```bash
python scripts/consolidate_composition_results.py "$RUN_DIR" \
  --include-query-ids q01,q02,q03,q04,q05,q06,q07,q08,q09,q10,q11,q12,q13,q14,q15
```

## 1. Consolidate Composition Results

Command:

```bash
python scripts/consolidate_composition_results.py "$RUN_DIR"
```

Default output directory:

```text
<run-folder>/summary
```

Generated files:

- `all_15_query_composition_results.csv`: one row per query and mode.
- `aggregate_mode_scores.csv`: one row per mode, aggregated from the detailed
  CSV.
- `summary_generation_report.json`: reproducibility metadata, query coverage,
  missing modes, tied-best query metadata, and filter metadata.
- `README.md`: short generated summary for the output directory.

Important options:

```bash
python scripts/consolidate_composition_results.py "$RUN_DIR" \
  --output-dir summary_official \
  --include-query-ids q01,q02,q03 \
  --allow-missing-modes
```

- `--output-dir`: output directory. Relative paths are resolved under the run
  folder.
- `--include-query-ids`: comma-separated query filter. Query IDs may be written
  as `q01`, `01`, or `1`.
- `--allow-missing-modes`: allow query rows that do not contain all canonical
  modes. Use only for diagnostics; final reporting should normally require all
  canonical modes.

Input requirements:

- The run folder must contain query folders named `qXX_*`.
- Duplicate folders for the same query ID are rejected.
- Each included query folder must contain composition evaluation rows at
  `evaluation/query_qXX_composition_qos_eval_rows.json`, or an
  `evaluation_result.json` pointer named `composition_qos_eval_rows_json`.
- When any canonical mode is present for a query, the script expects all four
  canonical modes unless `--allow-missing-modes` is supplied:
  `no_qos`, `qos_hybrid`, `qos_pure_llm`, `qos_topsis`.

Use these outputs for conclusions about:

- Per-query mode scores.
- Mean and standard deviation of QoS-adjusted composition score by mode.
- Mean validity, completeness, functional coverage, normalized QoS, response
  time, throughput, and availability by mode.
- Count of complete compositions, invalid compositions, and best-query wins.

## 2. Evaluate Ranking Agreement

Command:

```bash
python scripts/run_ranking_eval.py "$RUN_DIR"
```

Default output directory:

```text
<run-folder>/ranking_eval
```

Generated files:

- `spearman_matrix.csv`
- `average_overlap_matrix.csv`
- `rbo_matrix.csv`
- `jaccard_matrix.csv`
- `spearman_included_counts.csv`
- `average_overlap_included_counts.csv`
- `rbo_included_counts.csv`
- `jaccard_included_counts.csv`
- `pairwise_scores.csv`
- `included_cases.csv`
- `invalid_cases.csv`
- `loaded_rows.csv`
- `warnings.json`
- `summary.json`

Important options:

```bash
python scripts/run_ranking_eval.py "$RUN_DIR" \
  --output-dir "$RUN_DIR/ranking_eval" \
  --rbo-p 0.9 \
  --modes no_qos qos_pure_llm qos_topsis qos_hybrid
```

- `--output-dir`: output directory. Relative paths are resolved from the command
  invocation directory.
- `--rbo-p`: Rank-Biased Overlap persistence parameter. Default: `0.9`.
- `--modes`: selected modes to include. Default: all four modes in ranking
  metric order: `no_qos`, `qos_pure_llm`, `qos_topsis`, `qos_hybrid`.
- `--inclusion-policy`: internal/debug option. The default and final-reporting
  policy is strict selected-mode evaluation.

Input requirements:

- The run folder must contain query folders that look like `q*`.
- Each query folder must contain a usable Excel ranking or functional-match
  report. The script searches common locations such as:
  `evaluation/query_*_candidate_api_rankings.xlsx`,
  `functional_match_eval/query_*_candidate_api_rankings.xlsx`,
  and other matching `*rank*`, `*report*`, or `.xlsx` files under the query
  folder.
- The selected workbook sheet must contain columns that can be normalized to:
  `query_id`, `mode`, `subtask_id`, `api_id`, `mode_rank`, and
  `functional_match_label`.
- Optional columns such as `planner_selection_k`, `selected_for_planner`,
  `failure_flag`, `failure_stage`, `failure_reason`,
  `exclude_from_ranking_eval`, `is_hallucinated`, `is_duplicated`, and
  `ranking_anomaly` are preserved where available.

Metric definitions implemented by the script:

- `spearman`: standard Spearman correlation over the complete shared candidate
  set. Cases with duplicate rankings or unequal full candidate sets are excluded
  for Spearman.
- `average_overlap`: average prefix overlap over top-K lists.
- `rbo`: finite extrapolated Rank-Biased Overlap over top-K lists.
- `jaccard`: top-K set Jaccard similarity.

Top-K selection:

- For each query/subtask case, K is the number of functional matches in
  `qos_hybrid` when available and positive.
- If `qos_hybrid` is unavailable, the script uses a single positive
  `planner_selection_k` when present.
- Otherwise, it falls back to K=`5` and records a warning.

Use these outputs for conclusions about:

- Whether ranking modes agree on candidate API ordering.
- Which pairwise mode comparisons are supported by valid cases.
- Which query/subtask/mode rows were invalid and therefore excluded.
- How much warning or exclusion evidence should qualify ranking conclusions.

## 3. Generate Weight-Sensitivity Tables

Command:

```bash
python scripts/generate_weight_sensitivity_tables.py "$RUN_DIR"
```

Default output directory:

```text
<run-folder>/weigh_sensitivity
```

Generated files:

- `table_5_2_representative_alpha_beta_sensitivity.csv`
- `table_5_3_candidate_topn_risk_summary.csv`
- `selected_query_metrics.csv`
- `checked_topn_candidates.csv`
- `candidate_normalized_qos_scores.csv`
- `weight_sensitivity_tables.md`
- `methodology.json`

Important option:

```bash
python scripts/generate_weight_sensitivity_tables.py "$RUN_DIR" \
  --poor-qos-threshold 0.25
```

- `--output-dir`: output directory. Default is intentionally named
  `weigh_sensitivity` by the script.
- `--poor-qos-threshold`: candidate-level normalized QoS threshold for poor-QoS
  risk. Default: `0.25`.

Input requirements:

- `<run-folder>/summary/all_15_query_composition_results.csv`, produced by
  `consolidate_composition_results.py`.
- `<run-folder>/ranking_eval/included_cases.csv`, produced by
  `run_ranking_eval.py`.
- Per-query candidate ranking JSON files:
  `qXX_*/evaluation/query_qXX_candidate_api_rankings_rows.json`.

Weight settings evaluated:

- `0/100`: QoS-only extreme.
- `25/75`: QoS-dominant setting.
- `50/50`: balanced setting.
- `70/30`: selected primary setting.
- `75/25`: conservative robustness setting.
- `100/0`: functional-only extreme.

Workflow score formula used for sensitivity analysis:

```text
Composition_Completeness * (alpha * Functional_Coverage + beta * Normalized_QoS_Score)
```

Candidate-level QoS normalization:

- Response time is min-max normalized within each query/subtask candidate pool
  with lower values treated as better.
- Throughput is min-max normalized within each query/subtask candidate pool with
  higher values treated as better.
- Availability is min-max normalized within each query/subtask candidate pool
  with higher values treated as better.
- Candidate normalized QoS is the average of the three normalized components.

Use these outputs for conclusions about:

- Whether the best workflow mode is stable across functional/QoS weight choices.
- Which queries have unique best modes versus tied best modes.
- Whether selected top-N candidate rows include functionally invalid APIs or
  poor-QoS APIs under each weight setting.
- The exact methodology and threshold used for the reported sensitivity tables.

## 4. Generate Research Figures

Command:

```bash
python scripts/generate_research_figures.py "$RUN_DIR" \
  --path-query q02 \
  --panel-queries q01,q02,q03,q04,q05 \
  --panel-queries q06,q07,q08,q09,q10 \
  --panel-queries q11,q12,q13,q14,q15
```

Default output directory:

```text
<run-folder>/figures
```

Generated files:

- `selected_api_path_qXX.png`
- `average_composition_by_mode.png`
- `coverage_qos_tradeoff.png`
- `functional_coverage_normalized_qos.png`
- `qos_component_metrics.png`
- `ranking_similarity_heatmaps.png`
- `query_composition_scores.png`
- `winner_status_heatmap.png`
- `query_score_panels_qXX_qYY.png`

Important options:

- `--path-query`: query ID for the selected API path comparison. If omitted,
  the first discovered query folder is used.
- `--panel-queries`: comma-separated query IDs for one panel figure. This option
  may be repeated. If omitted, the script generates panels in discovered-query
  groups of five.
- `--output-dir`: output directory. Relative paths are resolved under the run
  folder.
- `--png-dpi`: PNG export resolution. Default: `300`.

Input requirements:

- Per-query summaries:
  `qXX_*/evaluation/query_qXX_composition_qos_eval_summary.json`.
- Selected API path figure:
  `qXX_*/0_decomposer.json`, `qXX_*/<mode>/4_planner.json`, and
  `qXX_*/evaluation/query_qXX_candidate_api_rankings_rows.json`.
- Ranking similarity figure:
  `<run-folder>/ranking_eval/spearman_matrix.csv`,
  `average_overlap_matrix.csv`, `rbo_matrix.csv`, and `jaccard_matrix.csv`.

Use these figures for conclusions about:

- Mode-level mean composition performance and uncertainty.
- Functional coverage versus normalized QoS trade-offs.
- QoS component behavior: response time, throughput, and availability.
- Ranking agreement between modes.
- Query-level winners and tied winners.
- Selected API paths and functional-match labels for a chosen query.

## Individual Figure Commands

Selected API path comparison:

```bash
python scripts/plot_api_path_comparison.py "$RUN_DIR" --query q02
```

Aggregate mode summaries:

```bash
python scripts/plot_mode_summary.py "$RUN_DIR"
```

Ranking similarity heatmaps:

```bash
python scripts/plot_ranking_similarity.py "$RUN_DIR"
```

Query-level grouped scores and winner heatmap:

```bash
python scripts/plot_query_score_overview.py "$RUN_DIR"
```

Query score panels:

```bash
python scripts/plot_query_score_panels.py "$RUN_DIR" --queries q01,q02,q03,q04,q05
```

Publication Figure 3 winner/tie heatmap:

```bash
python scripts/make_fig3_winner_tie_heatmap.py \
  --run-dir "$RUN_DIR" \
  --out-dir paper/figures
```

Publication Figure 3 composition diagnostics:

```bash
python scripts/generate_fig3_composition_diagnostics.py \
  --run-dir "$RUN_DIR" \
  --out-dir "$RUN_DIR/figures/paper"
```

Representative workflow case-study figure:

```bash
python scripts/figures/make_workflow_case_study_figure.py \
  --run-dir "$RUN_DIR" \
  --out-dir "$RUN_DIR/figures/paper"
```

Query-level QoS-adjusted score grouped bar chart:

```bash
python scripts/generate_query_level_grouped_bar.py \
  --results-dir "$RUN_DIR"
```

Query IDs may be written as `q01`, `01`, or `1`.

## Determinism and Failure Behavior

The scripts are deterministic for a fixed input run folder:

- Query folders are sorted by query ID.
- Mode rows are ordered by canonical mode order.
- Duplicate query folders are rejected.
- Duplicate mode rows in a query summary are rejected.
- Numeric figure inputs must be finite.
- Missing required files raise source-file-specific errors.
- Missing ranking or composition evidence is reported through errors, warnings,
  `invalid_cases.csv`, or metadata JSON rather than being imputed.
- Matplotlib uses a non-interactive `Agg` backend and writes cache data under a
  temporary directory for reproducible command-line execution.

Before using numbers in the thesis, inspect:

- `<run-folder>/summary/summary_generation_report.json`
- `<run-folder>/ranking_eval/summary.json`
- `<run-folder>/ranking_eval/warnings.json`
- `<run-folder>/ranking_eval/invalid_cases.csv`
- `<run-folder>/weigh_sensitivity/methodology.json`

These files document which query folders, modes, ranking cases, thresholds, and
warnings were included in the generated conclusions.

## Script Inventory

- `consolidate_composition_results.py`: builds official composition summary
  CSVs from per-query evaluation rows.
- `run_ranking_eval.py`: command-line entry point for ranking agreement metrics.
- `ranking_metrics.py`: reusable ranking metric implementation and report
  normalization logic.
- `generate_weight_sensitivity_tables.py`: builds weight-sensitivity and
  candidate-risk tables from summary and ranking outputs.
- `generate_research_figures.py`: orchestrates all supported figure-generation
  scripts.
- `figure_data.py`: shared data-loading, validation, sorting, aggregation, and
  plotting helpers.
- `plot_api_path_comparison.py`: selected API path comparison for one query.
- `plot_mode_summary.py`: aggregate score, trade-off, and QoS component figures.
- `plot_ranking_similarity.py`: ranking metric heatmaps from `ranking_eval`.
- `plot_query_score_overview.py`: query score bars and winner/tied-winner
  heatmap.
- `plot_query_score_panels.py`: small-multiple query score panels.
- `make_fig3_winner_tie_heatmap.py`: publication Figure 3 winner/tie heatmap,
  score matrix, summary, and debug heatmap.
- `generate_fig3_composition_diagnostics.py`: publication Figure 3 composition
  diagnostic panels, audit CSVs, and manifest.
- `generate_query_level_grouped_bar.py`: query-by-mode grouped bar chart and
  score matrix CSV.
- `figures/make_workflow_case_study_figure.py`: representative workflow
  case-study figure with CSV, metadata, and LaTeX sidecar.
