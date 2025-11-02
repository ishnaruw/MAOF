# src/agents/ranker_topsis.py
import json
import re

def _coerce_json(s: str) -> str:
    """Return a valid JSON string or {} if we can't parse."""
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

def make_qos_table(catalog_items):
    """
    Build rows: [{api_id, rt_ms, tp_rps, availability}, ...]
    Pulls from item['qos'] if present; handles missing fields.
    """
    rows = []
    for s in catalog_items:
        qos = s.get("qos", {}) or {}
        rows.append({
            "api_id": s["api_id"],
            "rt_ms": qos.get("rt_ms"),
            "tp_rps": qos.get("tp_rps"),
            "availability": qos.get("availability"),
        })
    return rows

def ranker_call(llm_call, qos_rows):
    """
    Ask the LLM to run TOPSIS using the prompt template and qos_rows table.
    Returns a list sorted by 'C' desc:
      [{"api_id":"...", "C":0.78, "D_plus":..., "D_minus":...}, ...]
    """
    template = open("prompts/ranker_topsis.md", "r", encoding="utf-8").read()
    prompt = template + "\n\nTable:\n" + json.dumps(qos_rows, ensure_ascii=False)
    resp = llm_call(prompt)
    resp = _coerce_json(resp)
    data = json.loads(resp)
    ranked = data.get("ranked", [])
    ranked = [r for r in ranked if "api_id" in r and "C" in r]
    ranked.sort(key=lambda x: x["C"], reverse=True)
    return ranked
