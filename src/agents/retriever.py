# src/agents/retriever.py
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
    # Fallback: try to extract the largest {...} block
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    return m.group(0) if m else "{}"

def retriever_call(llm_call, prompt: str):
    """
    Call the LLM with the retriever prompt.
    Expect strict JSON: {"keep":[{"api_id":"...", "reason":"..."}]}
    """
    resp = llm_call(prompt)
    resp = _coerce_json(resp)
    data = json.loads(resp)
    keep = data.get("keep", [])
    # normalize a little
    out = []
    for k in keep:
        api_id = k.get("api_id")
        if api_id:
            out.append({"api_id": api_id, "reason": k.get("reason", "")})
    return out

def collect_candidates(llm_call, user_goal: str, fetch_fn, category: str, with_qos: bool, max_batches=5):
    """
    Iterate through catalog batches, ask the LLM to keep relevant APIs,
    and return up to 8–12 unique candidates.
    """
    keep = {}
    offset = 0
    for _ in range(max_batches):
        batch = fetch_fn(category=category, offset=offset, limit=50, with_qos=with_qos)
        if not batch:
            break
        # Build the prompt from template
        prompt_tmpl = open("prompts/retriever.md", "r", encoding="utf-8").read()
        prompt = (
            prompt_tmpl
            .replace("{user_goal}", user_goal)
            .replace("{batch_json}", json.dumps(batch, ensure_ascii=False))
        )
        picks = retriever_call(llm_call, prompt)
        for p in picks:
            if p.get("api_id"):
                keep[p["api_id"]] = p.get("reason", "")
        if len(keep) >= 12:
            break
        offset += 50

    # Limit to 12, prefer insertion order
    items = list(keep.items())[:12]
    return [{"api_id": k, "reason": v} for k, v in items]
