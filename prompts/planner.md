Produce a numbered orchestration plan using only the ranked APIs provided.

User goal:
{user_goal}

Ranked candidates:
{ranked_compact}

Rules:
1) Prefer higher C when two APIs are equivalent.
2) Output numbered steps. Each step must include api_id and a one sentence why.
3) Do not execute anything. Do not invent parameters.

Return JSON:
{
 "plan":[
   {"step":1, "api_id":"...", "action":"...", "why":"..."}
 ],
 "selected_api_ids":["...", "..."]
}