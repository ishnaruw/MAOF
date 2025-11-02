# src/driver/run_autogen_pipeline.py

import os
import json
from pathlib import Path
from openai import AzureOpenAI

from src.tools.fetch_services import fetch_services
from src.agents.retriever import collect_candidates
from src.agents.ranker_topsis import make_qos_table, ranker_call
from src.agents.planner import planner_call
from src.core.topsis_verify import topsis_verify

# -------- Azure config --------
def _require_env(k: str) -> str:
    v = os.getenv(k)
    if not v:
        raise RuntimeError(f"{k} is not set. export it or put it in a .env")
    return v

AZURE_OPENAI_API_KEY = _require_env("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = _require_env("AZURE_OPENAI_ENDPOINT")  # e.g. https://autoai-openai.openai.azure.com/
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-dspy")  # your deployment name

_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

def _chat(system_message: str, user_prompt: str, force_json: bool = True) -> str:
    """
    Call Azure OpenAI. If force_json=True, request a strict JSON object response.
    """
    kwargs = dict(
        model=AZURE_OPENAI_DEPLOYMENT,  # deployment name
        temperature=0,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ],
    )
    if force_json:
        kwargs["response_format"] = {"type": "json_object"}
    r = _client.chat.completions.create(**kwargs)
    return r.choices[0].message.content or ""

# -------- role prompts --------
RETRIEVER_SYS = (
    "You are a retrieval agent that selects relevant APIs from a JSON catalog based on the user goal. "
    "Return strict JSON as instructed by the prompt; never invent services."
)
RANKER_SYS = (
    "You are a QoS evaluator. Apply TOPSIS to rt_ms (cost), tp_rps (benefit), and availability (benefit). "
    "Follow the prompt strictly and return valid JSON."
)
PLANNER_SYS = (
    "You are an orchestration planner that composes a logical API workflow using only the ranked APIs. "
    "Follow the prompt strictly and return valid JSON."
)

# -------- pipeline --------
def run_autogen_once(user_goal: str, category: str, with_qos: bool):
    # 1) RETRIEVER
    picks = collect_candidates(
        llm_call=lambda p: _chat(RETRIEVER_SYS, p, force_json=True),
        user_goal=user_goal,
        fetch_fn=fetch_services,
        category=category,
        with_qos=with_qos,
        max_batches=5,
    )
    pick_ids = [p["api_id"] for p in picks]

    # 2) GATHER CATALOG ITEMS FOR RANKER
    cat_items = []
    offset = 0
    while True:
        batch = fetch_services(category, offset, 200, with_qos)
        if not batch:
            break
        cat_items.extend([b for b in batch if b["api_id"] in pick_ids])
        offset += 200

    # Build QoS table (or placeholders when with_qos=False)
    qos_rows = (
        make_qos_table(cat_items)
        if with_qos
        else [{"api_id": s["api_id"], "rt_ms": None, "tp_rps": None, "availability": None} for s in cat_items]
    )

    # 3) RANKER (LLM TOPSIS) + OPTIONAL VERIFIER
    ranked = ranker_call(
        llm_call=lambda p: _chat(RANKER_SYS, p, force_json=True),
        qos_rows=qos_rows,
    )
    verified = topsis_verify(qos_rows)

    # 4) PLANNER — define ranked_top before calling planner
    if ranked:
        ranked_top = ranked[:6]
    else:
        # Fallback: if ranker returned nothing, take first few catalog items with C=0.0
        ranked_top = [{"api_id": s["api_id"], "C": 0.0} for s in cat_items[:6]]

    plan = planner_call(
        llm_call=lambda p: _chat(PLANNER_SYS, p, force_json=True),
        user_goal=user_goal,
        ranked_top=ranked_top,
    )

    # 5) LOGS
    out = Path("results/logs")
    out.mkdir(parents=True, exist_ok=True)
    (out / "retriever_autogen.json").write_text(json.dumps(picks, indent=2))
    (out / "ranker_autogen.json").write_text(json.dumps(ranked, indent=2))
    (out / "planner_autogen.json").write_text(json.dumps(plan, indent=2))
    (out / "topsis_verify.json").write_text(json.dumps(verified, indent=2))

    print("Saved to results/logs/")
    return ranked, plan

if __name__ == "__main__":
    run_autogen_once(
        "Get a 5 day forecast by city and yesterday weather",
        category="Weather",
        with_qos=True,
    )
