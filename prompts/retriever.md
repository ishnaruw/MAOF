You select relevant APIs from the batch I give you. Use only fields present. Never invent services.
Return strict JSON with this schema:
{"keep":[{"api_id":"...", "reason":"one short sentence"}]}

Goal:
{user_goal}

Batch:
{batch_json}

Rules:
1) Keep 8 to 12 candidates that match the goal.
2) Prefer APIs whose description explicitly mentions the needed capability.
3) If nothing matches, return {"keep":[]}.