# src/driver/run_autogen_pipeline.py

import os
import json
import time
from pathlib import Path

from src.llm.backends import make_backend
from src.tools.fetch_services import fetch_services
from src.agents.retriever import collect_candidates
from src.agents.ranker_topsis import make_qos_table, ranker_call
from src.agents.planner import planner_call
from src.core.topsis_verify import topsis_verify


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


def _run_dir(model_tag: str) -> Path:
    run_id = time.strftime("%Y%m%dT%H%M%S")
    d = Path(f"results/logs/{model_tag}/{run_id}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_autogen_once(user_goal: str, category: str, with_qos: bool):
    backend = make_backend()  # Azure or Mistral, chosen by LLM_PROVIDER
    model_tag = backend.name()

    # 1) RETRIEVER
    picks = collect_candidates(
        llm_call=lambda p: backend.chat_json(RETRIEVER_SYS, p, temperature=0, force_json=True),
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
        llm_call=lambda p: backend.chat_json(RANKER_SYS, p, temperature=0, force_json=True),
        qos_rows=qos_rows,
    )
    verified = topsis_verify(qos_rows)

    # 4) PLANNER
    ranked_top = ranked[:6] if ranked else [{"api_id": s["api_id"], "C": 0.0} for s in cat_items[:6]]
    plan = planner_call(
        llm_call=lambda p: backend.chat_json(PLANNER_SYS, p, temperature=0, force_json=True),
        user_goal=user_goal,
        ranked_top=ranked_top,
    )

    # 5) LOGS
    out = _run_dir(model_tag)
    (out / "retriever_autogen.json").write_text(json.dumps(picks, indent=2))
    (out / "ranker_autogen.json").write_text(json.dumps(ranked, indent=2))
    (out / "planner_autogen.json").write_text(json.dumps(plan, indent=2))
    (out / "topsis_verify.json").write_text(json.dumps(verified, indent=2))
    (out / "meta.json").write_text(json.dumps({
        "model_tag": model_tag,
        "provider": model_tag.split(":")[0],
        "model": model_tag.split(":")[1] if ":" in model_tag else model_tag,
        "category": category,
        "with_qos": with_qos,
        "user_goal": user_goal,
    }, indent=2))

    print(f"Saved to {out}")
    return ranked, plan


if __name__ == "__main__":
    # Example single run
    run_autogen_once(
        user_goal="Get a 5 day forecast by city and yesterday weather",
        category="Weather",
        with_qos=True,
    )
