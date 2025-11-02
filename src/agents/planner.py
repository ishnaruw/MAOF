# src/agents/planner.py
import json
import re

def _coerce_json(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "{}"
    try:
        json.loads(s)
        return s
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    return m.group(0) if m else "{}"

def planner_call(llm_call, user_goal: str, ranked_top):
    """
    Compose a simple orchestration plan from ranked candidates.
    Expects ranked_top like: [{"api_id":"...", "C":0.88}, ...]
    """
    compact = [{"api_id": r["api_id"], "C": round(r.get("C", 0) or 0, 4)} for r in ranked_top]
    prompt = open("prompts/planner.md", "r", encoding="utf-8").read() \
        .replace("{user_goal}", user_goal) \
        .replace("{ranked_compact}", json.dumps(compact))
    resp = llm_call(prompt)
    resp = _coerce_json(resp)
    return json.loads(resp)
