import math

def _norm(vals):
    xs = [v for v in vals if v is not None]
    return math.sqrt(sum(x*x for x in xs)) or 1.0

def topsis_verify(rows, weights=(0.5,0.3,0.2)):
    cleaned = []
    for r in rows:
        def safe(v): return None if v is None or v == -1 else float(v)
        cleaned.append({
            "api_id": r["api_id"],
            "rt": safe(r.get("rt_ms")),
            "tp": safe(r.get("tp_rps")),
            "av": safe(r.get("availability")),
        })

    rt_n = _norm([r["rt"] for r in cleaned if r["rt"] is not None])
    tp_n = _norm([r["tp"] for r in cleaned if r["tp"] is not None])
    av_n = _norm([r["av"] for r in cleaned if r["av"] is not None])
    wrt, wtp, wav = weights

    vals = []
    for r in cleaned:
        rt = (r["rt"]/rt_n)*wrt if r["rt"] is not None else None
        tp = (r["tp"]/tp_n)*wtp if r["tp"] is not None else None
        av = (r["av"]/av_n)*wav if r["av"] is not None else None
        vals.append({"api_id": r["api_id"], "rt": rt, "tp": tp, "av": av})

    def best_worst(col, benefit):
        xs = [v[col] for v in vals if v[col] is not None]
        if not xs: return None, None
        return (max(xs), min(xs)) if benefit else (min(xs), max(xs))

    rt_b, rt_w = best_worst("rt", benefit=False)
    tp_b, tp_w = best_worst("tp", benefit=True)
    av_b, av_w = best_worst("av", benefit=True)

    ranked = []
    for v in vals:
        dp = dm = 0.0
        for x, b, w in [(v["rt"], rt_b, rt_w), (v["tp"], tp_b, tp_w), (v["av"], av_b, av_w)]:
            if x is None or b is None or w is None:
                continue
            dp += (x - b)**2
            dm += (x - w)**2
        dp, dm = math.sqrt(dp), math.sqrt(dm)
        C = dm/(dp+dm) if (dp+dm) > 0 else 0.0
        ranked.append({"api_id": v["api_id"], "C": C, "D_plus": dp, "D_minus": dm})
    ranked.sort(key=lambda x: x["C"], reverse=True)
    return ranked
