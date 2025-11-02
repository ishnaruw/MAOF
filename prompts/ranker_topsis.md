Rank the given APIs using TOPSIS and return strict JSON.

Inputs:
- Table rows: api_id, rt_ms, tp_rps, availability
- Criteria: rt_ms (cost), tp_rps (benefit), availability (benefit)
- Weights: w_rt=0.5, w_tp=0.3, w_avail=0.2
- Missing or -1 means unavailable for that column.

Steps:
1) Build the decision matrix X.
2) Normalize each column with v_ij = x_ij / sqrt(sum x_ij^2) using available values.
3) Apply weights per column.
4) Ideal best for benefits is max, for cost is min. Ideal worst conversely.
5) Distances: D_plus to ideal best, D_minus to ideal worst.
6) Preference: C_i = D_minus / (D_plus + D_minus).

Return JSON:
{"ranked":[{"api_id":"...", "C":0.78, "D_plus":..., "D_minus":...}, ...]}

Do not invent numbers. Use only the rows provided.