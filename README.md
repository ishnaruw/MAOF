# AutoLLMCompose

AutoLLMCompose is a research codebase for multi-agent API discovery, ranking,
composition, and evaluation. Given a user goal, the pipeline decomposes the goal
into ordered API-retrieval subtasks, retrieves candidate APIs from a committed
FAISS-backed API catalog index, ranks candidates under QoS and non-QoS modes,
generates composition plans, and writes deterministic evaluation artifacts for
thesis analysis.

## Current Capabilities

- Query decomposition into 2-5 ordered API-retrieval subtasks.
- Shared semantic retrieval from the local API catalog index.
- LLM binary functional refinement over retrieved candidates.
- Candidate ranking across four modes: `no_qos`, `qos_pure_llm`,
  `qos_topsis`, and `qos_hybrid`.
- Deterministic TOPSIS scoring from QoS metrics.
- Functional-first hybrid ranking with zero-functional-match retrieval retry
  for the hybrid view.
- LLM-based planning over selected APIs.
- Deterministic candidate-ranking, duplicate, hallucination, anomaly, and
  composition-QoS evaluation outputs.
- Streamlit dashboard for running experiments, browsing completed runs,
  inspecting live logs, evaluating ranking similarity, and visualizing planned
  API compositions.
- Post-run scripts for composition summaries, ranking agreement metrics,
  weight-sensitivity tables, and research figures.

## Repository Layout

```text
AutoLLMCompose/
|-- data/
|   |-- processed/api_catalog_sample_balanced/
|   |   |-- api_repo.enriched.jsonl        # Runtime functional catalog
|   |   |-- api_qos.jsonl                  # QoS overlay keyed by api_id
|   |   |-- api_repo.tooldesc.jsonl        # Base functional catalog
|   |   |-- enrichment_manifest.json       # Catalog provenance
|   |   `-- README.md
|   |-- index/faiss_no_qos/                # Committed FAISS index
|   |-- queries/all_user_query.jsonl       # Main query set
|   |-- data_gen/                          # Archival notebooks
|   |-- raw/wsdream/                       # Archival raw matrices
|   `-- results/api_inventory/             # Archival inventory reports
|-- prompts/                               # LLM prompt templates
|-- scripts/                               # Post-experiment analysis scripts
|-- src/
|   |-- agents/                            # Decomposer, refiner, ranker, planner, evaluator
|   |-- config/pipeline_config.py          # Central pipeline defaults
|   |-- core/                              # Schemas, parsing, retry, logging helpers
|   |-- driver/run_autogen_pipeline.py     # Main experiment runner
|   |-- eval/                              # Evaluation and audit helpers
|   |-- llm/                               # Provider backends and AutoGen gateway
|   |-- rag/                               # FAISS retrieval wrapper
|   |-- tools/                             # Catalog loading utilities
|   `-- ui/                                # Streamlit dashboard and live-demo helpers
|-- tests/                                 # Unit tests
|-- requirements.txt
`-- README.md
```

## Requirements

- Python 3.12 is the current local development target.
- Python 3.10+ should work for most of the code, but 3.12 best matches the
  checked-in environment.
- An LLM provider key, or a running LM Studio server for local model use.

## Installation

Run commands from the repository root:

```bash
cd AutoLLMCompose
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` includes `faiss-cpu`. If FAISS wheels are unavailable on
your platform, install FAISS through conda instead:

```bash
conda install -c conda-forge faiss-cpu
```

## Environment Variables

Create a local `.env` file in the repository root. The code loads this file
automatically. Do not commit it.

Set `LLM_PROVIDER` to your default provider, or pass `--provider` when running
the pipeline. If neither is set, the backend defaults to `azure`.

```bash
# Provider selection
LLM_PROVIDER=fireworks

# Azure OpenAI
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-05-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o-dspy

# Mistral
MISTRAL_API_KEY=
MISTRAL_MODEL=mistral-large-latest

# Groq
GROQ_API_KEY=
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_MULTI_MODEL_MODE=false
GROQ_MULTI_MODELS=llama-3.3-70b-versatile,openai/gpt-oss-120b,openai/gpt-oss-20b,llama-3.1-8b-instant,qwen/qwen3-32b
GROQ_MULTI_SAME_MODEL_RETRIES=2
GROQ_MULTI_COMPLETION_TOKEN_RESERVE=2500

# Fireworks AI
FIREWORKS_API_KEY=
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
FIREWORKS_MODEL=accounts/fireworks/models/gpt-oss-120b
FIREWORKS_MODELS=accounts/fireworks/models/gpt-oss-120b,accounts/fireworks/models/deepseek-v4-pro
FIREWORKS_TIMEOUT_SECONDS=180

# Together AI
TOGETHER_API_KEY=
TOGETHER_BASE_URL=https://api.together.xyz/v1
TOGETHER_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo

# Gemini
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash

# Azure AI Foundry
AZURE_FOUNDRY_API_KEY=
AZURE_FOUNDRY_ENDPOINT=https://your-resource.services.ai.azure.com/models
AZURE_FOUNDRY_API_VERSION=2024-05-01-preview
AZURE_FOUNDRY_MODEL=DeepSeek-R1-0528
AZURE_FOUNDRY_TIMEOUT_SECONDS=300

# LM Studio, OpenAI-compatible local server
LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1
LMSTUDIO_MODEL=meta-llama-3.1-8b-instruct
LMSTUDIO_API_KEY=lmstudio

# LM Studio native chat endpoint
LMSTUDIO_QWEN_CHAT_URL=http://localhost:1234/api/v1/chat
LMSTUDIO_QWEN_MODEL=qwen2.5-3b-instruct.gguf
```

Accepted provider names and aliases:

```text
azure
mistral
groq
fireworks, fireworks_ai
together, together_ai
google, gemini
azure_foundry, foundry, azure-deepseek, deepseek
lmstudio, local
lmstudio_qwen, lmstudio_native, local_qwen
```

For Groq, pass `--model multi` or set `GROQ_MULTI_MODEL_MODE=true` to enable
multi-model failover across `GROQ_MULTI_MODELS`.

## Runtime Data

Normal runs use committed data under `data/`:

- `data/queries/all_user_query.jsonl`: default query set.
- `data/processed/api_catalog_sample_balanced/api_repo.enriched.jsonl`:
  primary functional catalog.
- `data/processed/api_catalog_sample_balanced/api_qos.jsonl`: QoS overlay.
- `data/index/faiss_no_qos/`: committed FAISS index and metadata.

The archival directories `data/data_gen/`, `data/raw/wsdream/`, and
`data/results/api_inventory/` are kept for research transparency. They are not
part of the normal runtime path.

This checkout uses the committed FAISS index directly. If you intentionally
change the catalog or embedding setup, rebuild the corresponding index with the
project's index-generation workflow before committing the catalog and index
together.

## Running The Streamlit Dashboard

The main app entry point is `src/ui/ranking_eval_app.py`. It includes the live
demo, ranking evaluation, composition visualization, experiment launcher, and
completed-run browser in one Streamlit sidebar.

Start from the repository root after installation:

```bash
source .venv/bin/activate
python -m streamlit run src/ui/ranking_eval_app.py
```

Streamlit prints the local URL, usually `http://localhost:8501`. If that port is
busy, choose another one:

```bash
python -m streamlit run src/ui/ranking_eval_app.py --server.port 8502
```

For a headless or remote terminal session:

```bash
python -m streamlit run src/ui/ranking_eval_app.py \
  --server.headless true \
  --server.port 8501
```

The sidebar pages are:

| Page | Use it for | Expected data |
| --- | --- | --- |
| `Live Demo Deep Dive` | Defense/demo walkthrough for one completed parent run, including query context, retrieval, ranking, planner outputs, and raw artifacts. | A model run folder such as `results/logs/<run_tag>/<provider_model>/` containing `qXX_*` query folders. |
| `Ranking Evaluation` | Ranking-similarity and composition-workflow evaluation across completed query runs. | A parent runs directory containing one or more `qXX_*` folders with evaluation reports. |
| `Composition Visualizations` | Workflow graph, risk analysis, recommendation, mode comparison, and sequence diagrams for planned compositions. | Composition QoS evaluation rows and planner outputs under completed query run folders. |
| `Run Experiments` | Launch new query runs from the browser and monitor stage progress, logs, and generated reports. | Provider credentials in `.env`, or a running LM Studio server, plus the committed catalog and FAISS index. |
| `Completed Runs` | Browse existing or still-running query folders and inspect logs/artifacts. | Any directory under `results/logs/` that contains `qXX_*` query folders. |

You can browse completed artifacts without an LLM key. Starting a new run from
the `Run Experiments` page uses the same driver as the CLI and writes outputs
under:

```text
results/logs/<run_tag>/<provider_model>/<query_id_timestamp>/
```

Launcher stdout/stderr logs are stored under:

```text
results/logs/streamlit_launches/
```

If you refresh or regenerate reports while the app is open, use the page's
`Reload reports`, `Reload visualization data`, `Reload completed runs`, or
`Reload live demo artifacts` button. Streamlit caches loaded artifacts for
speed, so those buttons clear the relevant cache.

The live-demo page can also run standalone when you only need that view:

```bash
python -m streamlit run src/ui/live_demo_deep_dive.py
```

## Running The Pipeline

Run one query non-interactively:

```bash
python -m src.driver.run_autogen_pipeline \
  --query-ids q01 \
  --provider fireworks \
  --model accounts/fireworks/models/gpt-oss-120b \
  --run-tag DEV_RUN
```

Run multiple queries:

```bash
python -m src.driver.run_autogen_pipeline \
  --query-ids q01,q02,q03 \
  --provider groq \
  --model multi \
  --run-tag GROQ_DEV
```

You can also pass repeated query IDs:

```bash
python -m src.driver.run_autogen_pipeline \
  --query-id q01 \
  --query-id q05 \
  --provider mistral
```

Use a custom query file with the same `id`, `title`, and `goal` fields:

```bash
python -m src.driver.run_autogen_pipeline \
  --queries-path data/queries/all_user_query.jsonl \
  --query-ids q01 \
  --provider fireworks
```

If you run the driver without arguments, it opens an interactive query and
provider selector:

```bash
python -m src.driver.run_autogen_pipeline
```

## Pipeline Stages

For each selected query, the driver performs:

1. Decomposition, written to `0_decomposer.json`.
2. Shared retrieval, written to `1_retriever_s<subtask>.json`.
3. Functional refinement, written under `evaluation/`.
4. Ranking for `no_qos`, `qos_pure_llm`, `qos_topsis`, and `qos_hybrid`.
5. Candidate selection and planning for each mode when planning is enabled.
6. Candidate ranking, audit, and anomaly evaluation output generation.
7. Composition-QoS evaluation when planning and composition evaluation are
   enabled.

| Mode | Ranking Signal | Description |
| --- | --- | --- |
| `no_qos` | Functional evidence only | LLM ranks retrieved APIs without QoS fields. |
| `qos_pure_llm` | Functional labels + LLM QoS score | Scores QoS with an LLM, enriches candidates with functional labels, and enforces functional-first ranking. |
| `qos_topsis` | Deterministic TOPSIS | Ranks by computed QoS closeness score. |
| `qos_hybrid` | Functional match + TOPSIS | Places functional matches first and orders them with TOPSIS metadata; can retry retrieval for subtasks with zero functional matches. |

## Outputs

Runs are written under:

```text
results/logs/<run_tag>/<provider_model>/<query_id_timestamp>/
```

Important files in each query run include:

```text
meta.json                         # Run status, timings, provider/model metadata
run_config.json                   # Pipeline config snapshot
run.log                           # Stage-level run log
model_usage.json                  # Model usage and failover metadata
0_decomposer.json                 # Decomposed subtasks
1_retriever_s<id>.json            # Shared retrieved candidates
evaluation_result.json            # Pointers to evaluation outputs
<mode>/2_ranked_s<id>.json        # Ranked APIs per mode/subtask
<mode>/3_selected_s<id>.json      # Planner input selection per mode/subtask
<mode>/4_planner.json             # Planner output per mode
evaluation/                       # Excel, JSON, audit, anomaly, and QoS reports
```

The evaluation folder can contain:

- `query_<id>_retrieval_functional_match_rows.json`
- `query_<id>_retrieval_functional_match_summary.json`
- `query_<id>_candidate_api_rankings.xlsx`
- `query_<id>_candidate_api_rankings_rows.json`
- `query_<id>_candidate_api_rankings_summary.json`
- `query_<id>_duplicate_audit.json`
- `query_<id>_hallucination_audit.json`
- `query_<id>_ranking_anomaly_audit.json`
- `query_<id>_mode_anomalies.xlsx`
- `query_<id>_planner_selection_k_summary.json`
- `query_<id>_composition_qos_eval_rows.json`
- `query_<id>_composition_qos_eval_summary.json`
- `query_<id>_composition_qos_eval.xlsx`
- composition validity issue JSON/log outputs when applicable

## Post-Experiment Analysis

The deterministic scripts in `scripts/` operate on completed parent run folders
that contain one directory per query, named `qXX_*`.

Example run folder:

```bash
RUN_DIR="results/logs/RUNS_MAY_31_NEW_5/fireworks_gpt-oss-120b"
```

Recommended order:

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

Key outputs:

- `<run-folder>/summary/`: consolidated composition scores and aggregate mode
  summaries.
- `<run-folder>/ranking_eval/`: Spearman, average-overlap, RBO, and Jaccard
  ranking agreement matrices plus included/invalid cases.
- `<run-folder>/weigh_sensitivity/`: representative alpha/beta sensitivity
  tables and candidate Top-N risk summaries.
- `<run-folder>/figures/`: publication-ready PNG/PDF figures.

See `scripts/README.md` for detailed script inputs, outputs, and methodology
guards.

## Testing

Run the full unit test suite:

```bash
python -m unittest discover -s tests
```

Run focused tests while changing a subsystem:

```bash
python -m unittest tests.test_json_parsing tests.test_output_schemas
python -m unittest tests.test_ranker_parser
python -m unittest tests.test_composition_qos_eval
python -m unittest tests.test_fireworks_model_selection tests.test_groq_failover_backend
python -m unittest tests.test_live_demo_loader tests.test_composition_visualization_recommendation
```

For a quick syntax check:

```bash
python -m compileall src tests scripts
```

## Configuration

Pipeline defaults are defined in `src/config/pipeline_config.py`. Common values:

- `run_tag`: default output folder under `results/logs/`.
- `shared_index_dir`: FAISS index path.
- `catalog_path`: base functional catalog path.
- `catalog_enriched_path`: runtime functional catalog path.
- `api_qos_path`: QoS overlay path.
- `rag_top_k`: candidates retrieved per subtask.
- `zero_functional_retrieval_retry_enabled`: enables hybrid-only retrieval retry
  when a subtask has zero functional matches.
- `ranker_max_candidates` and `ranker_pool_n`: ranker candidate limits.
- `functional_match_chunk_size`: functional-refinement batch size.
- `functional_refinement_enabled`: enables LLM binary functional labeling.
- `selector_top_n`: fallback number of APIs selected for planner input.
- `planner_enabled`: enables planner generation.
- `planner_candidate_mode`: planner selection mode.
- `hybrid_workflow_selector`: hybrid composition selection strategy.
- `composition_qos_eval_enabled`: enables composition-level QoS evaluation.
- `llm_validation_max_retries`: bounded retries for structurally invalid LLM
  outputs.
- `qos_llm_batch_size`, `qos_llm_validate_formula`, and
  `qos_llm_formula_audit`: QoS LLM scoring controls.

Prefer changing defaults in code only for persistent project-wide behavior. For
one-off runs, use CLI flags such as `--provider`, `--model`, `--query-ids`,
`--queries-path`, and `--run-tag`.

## Troubleshooting

`RuntimeError: faiss is required`

Install `faiss-cpu` with pip or conda, and make sure the committed index exists
under `data/index/faiss_no_qos/`.

`FileNotFoundError` for `data/index/faiss_no_qos/faiss.index`

Restore or regenerate the FAISS index before running the pipeline. The normal
checkout expects these files:

```text
data/index/faiss_no_qos/faiss.index
data/index/faiss_no_qos/meta.jsonl
data/index/faiss_no_qos/config.json
```

Provider key errors such as `FIREWORKS_API_KEY missing`

Add the required key to `.env`, pass a different provider with `--provider`, or
use a running LM Studio server.

Groq prompt-size or rate-limit failures

Use Groq failover mode:

```bash
python -m src.driver.run_autogen_pipeline \
  --query-ids q01 \
  --provider groq \
  --model multi
```

LM Studio timeouts or connection errors

Start the LM Studio local server and verify the configured URL:

```bash
LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1
LMSTUDIO_QWEN_CHAT_URL=http://localhost:1234/api/v1/chat
```

No services loaded from catalog

Check that `data/processed/api_catalog_sample_balanced/api_repo.enriched.jsonl`
exists and that the repository data files were pulled correctly.

Streamlit opens but a dashboard page shows no runs

Point the sidebar directory selector at the parent folder that contains query
run folders named `qXX_*`. For most pages this is a model-level folder such as:

```text
results/logs/<run_tag>/<provider_model>/
```

If you just launched a run from the app, use `Completed Runs` or `Run
Experiments` and check `results/logs/streamlit_launches/` for the launcher log.

Dashboard still shows old values after recalculating reports

Use the relevant reload button in the sidebar, or restart Streamlit:

```bash
python -m streamlit run src/ui/ranking_eval_app.py
```

NumPy optional dependency warnings from Anaconda

The app hardens the scientific runtime before importing dashboard plotting
libraries, which avoids common `numexpr`/`bottleneck` import failures in mixed
NumPy environments. If the same warnings appear in direct test runs, use the
project virtual environment created from `requirements.txt`, or reinstall the
affected optional packages inside the active environment.

## Notes For Contributors

- Keep generated outputs under `results/`; this directory is ignored by git.
- Keep local secrets in `.env`; it is ignored by git.
- Rebuild retrieval artifacts after intentionally changing the functional
  catalog.
- Use JSON sidecars as the detailed source of truth for evaluation. Excel
  workbooks are user-facing reports.
- Do not overwrite historical run artifacts unless the task explicitly asks for
  regeneration.

## Citation

If you use this framework or datasets, please cite:

```bibtex
@research{Subramanian2025AutoLLMCompose,
  title={AutoLLMCompose: Multi-Agent LLM Framework for Service Discovery and Composition},
  author={Ishwarya Narayana Subramanian and Eyhab Al-Masri},
  year={2025},
  institution={University of Washington Tacoma}
}
```

## License

MIT License (c) 2025 Ishwarya Narayana Subramanian.
See [LICENSE](LICENSE) for details.

## Acknowledgments

- **Prof. Eyhab Al-Masri**, University of Washington Tacoma - ealmasri@uw.edu
- Supported by the University of Washington Master's in Computer Science &
  Systems program

**Researcher:** Ishwarya Narayana Subramanian, University of Washington Tacoma  
Contact: ishnaruw@uw.edu
