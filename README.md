# MAOF: Multi-Agent Observability Framework  
*Automating API Orchestration using Large Language Models*

--- 

## Overview

**MAOF (Multi-Agent Observability Framework)** is a modular system that explores how Large Language Models (LLMs) can discover, rank, and compose APIs based on **observability metrics** such as response time, throughput, and availability.
It combines Retrieval-Augmented Generation (RAG), TOPSIS-based QoS ranking, and multi-agent orchestration through AutoGen to enable transparent and performance-aware API automation.

---

## Architecture

MAOF organizes its components as modular agents in a transparent, observable pipeline.

```

User Query
│
▼
[Retriever Agent] ───► Selects relevant APIs from the catalog
│
▼
[Ranker Agent] ──────► Applies TOPSIS ranking using QoS metrics (rt_ms, tp_rps, availability)
│
▼
[Planner Agent] ─────► Generates a coherent composition plan
│
▼
[Coordinator Agent] ─► Fuses results across multiple LLMs

```

---

## Repository Structure

```

MAOF/
├── data/                   # API datasets and generated artifacts
│   ├── raw/                # Original ToolBench / WSDream datasets
│   ├── processed/          # Cleaned catalogs and capability tags
│   ├── data_gen/           # Jupyter notebooks for data extraction
│   └── results/            # Generated API inventories
│
├── prompts/                # LLM instruction templates
│   ├── retriever.md
│   ├── ranker_topsis.md
│   └── planner.md
│
├── src/
│   ├── tools/fetch_services.py     # JSONL loader and batch fetcher
│   ├── agents/
│   │   ├── retriever.py            # LLM-based candidate selection
│   │   ├── ranker_topsis.py        # TOPSIS QoS ranking
│   │   └── planner.py              # Plan composition generator
│   ├── core/topsis_verify.py       # Numeric TOPSIS verification
│   └── driver/run_autogen_pipeline.py  # Main pipeline script
│
├── results/
│   ├── logs/               # Latest agent outputs (retriever, ranker, planner)
│   └── comparisons/        # Evaluation summaries and plots
│
├── runs/                   # Dated experimental runs
├── requirements.txt
└── README.md

````

---

## Installation & Setup

### 1. Create environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
````

### 2. Configure environment variables

MAOF currently supports **Azure OpenAI** deployments (default: `gpt-4o-dspy`).

Create a `.env` file in the root with:

```bash
AZURE_OPENAI_API_KEY=your_azure_api_key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-05-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o-dspy
```

Alternatively, export them directly:

```bash
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=...
```

---

## Running the Pipeline

Run the full LLM-based retrieval–ranking–planning pipeline:

```bash
python -m src.driver.run_autogen_pipeline
```

### Outputs

The pipeline saves intermediate and final outputs under:

```
results/logs/
├── retriever_autogen.json   # Selected candidate APIs
├── ranker_autogen.json      # Ranked APIs (TOPSIS)
├── planner_autogen.json     # Generated orchestration plan
└── topsis_verify.json       # Numeric verification of LLM scores
```

---

## Experiment Modes

MAOF supports four experimental configurations for evaluating retrieval and observability effects.

| Mode | Retrieval | QoS | Status | Description |
|------|------------|-----|---------|-------------|
| **1. noRAG_noQoS** | LLM retriever only | ✗ | Planned | Baseline without QoS ranking |
| **2. noRAG_QoS** | LLM retriever | ✓ |  Implemented | Current pipeline with TOPSIS QoS ranking |
| **3. RAG_noQoS** | FAISS prefilter + LLM retriever | ✗ | Planned | Adds embedding prefiltering |
| **4. RAG_QoS** | FAISS prefilter + LLM retriever | ✓ | Planned | Full hybrid RAG + QoS pipeline |

The current implementation runs **Mode 2 (noRAG_QoS)** using three LLM agents (Retriever, Ranker, Planner) in the AutoGen framework.


---

## Agents Overview

| Agent                            | Function                             | Key File                      | Notes                                        |
| -------------------------------- | ------------------------------------ | ----------------------------- | -------------------------------------------- |
| **Retriever Agent**              | Selects relevant APIs for user goal  | `src/agents/retriever.py`     | Uses LLM to filter JSON catalog              |
| **Ranker Agent**                 | Performs QoS-based scoring           | `src/agents/ranker_topsis.py` | Follows TOPSIS ranking logic                 |
| **Planner Agent**                | Composes selected APIs into workflow | `src/agents/planner.py`       | Produces JSON plan output                    |
| **Coordinator Agent (in progress)** | Aggregates ranked outputs from multiple LLMs | (to be added) | Planned for cross-model fusion and consensus scoring |


---

## Result Interpretation

* **retriever_autogen.json** → Candidate APIs (`api_id`, `reason`)
* **ranker_autogen.json** → TOPSIS results (`C`, `D_plus`, `D_minus`)
* **planner_autogen.json** → Final orchestration plan (`step`, `api_id`, `why`)
* **topsis_verify.json** → Numerical verification of LLM ranking

Higher `C` means closer to the ideal QoS point (fast, reliable, available).

---

## Evaluation (in progress)

MAOF supports multi-LLM evaluation across:

* **LLMs:** GPT-4o, Mistral, OpenAI GPT-4o-mini, and local TinyLlama
* **Metrics:**

  * Candidate overlap @k
  * Kendall τ agreement (LLM vs numeric TOPSIS)
  * Plan completeness and logical order
  * RAG vs no-RAG improvement
  * QoS impact on ranking consistency

Coordinator fusion and multi-LLM comparison are under active development.

---

## Data Sources

* **ToolBench** (API capability metadata)
* **WSDream** (QoS measurements: latency, throughput, availability)
* **Custom curated catalogs** for cross-domain service discovery experiments

---

## Future Extensions

* Add **semantic RAG module** using FAISS/Chroma for pre-retrieval filtering
* Extend coordinator agent for **cross-LLM fusion and justification**
* Automate batch runs across 10+ user queries × 3 models × 4 modes
* Integrate **evaluation dashboards** (e.g., Streamlit or Jupyter notebooks)

---

<!-- ## 🧑‍💻 Citation

If you use this framework or datasets, please cite:

```
@research{Subramanian2025MAOF,
  title={MAOF: Multi-Agent Observability Framework for Service Discovery and Composition},
  author={Ishwarya Narayana Subramanian and Eyhab Al-Masri},
  year={2025},
  institution={University of Washington Tacoma}
}
```

---

## 📜 License

MIT License © 2025 Ishwarya Narayana Subramanian
See [LICENSE](LICENSE) for details.

--- -->

## Acknowledgments

* **Prof. Eyhab Al-Masri**, University of Washington Tacoma — ealmasri@uw.edu  
* Supported by the University of Washington Master’s in Computer Science & Systems program

---

**Researcher:** Ishwarya Narayana Subramanian (University of Washington Tacoma)  
Contact: ishnaruw@uw.edu

