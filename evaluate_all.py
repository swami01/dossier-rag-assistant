"""Score every retrieval mode on data/questions.json."""
import json

from pipeline import admin, office_user, retrieve

qs = json.load(open("data/questions.json"))
MODES = ["vector", "filters", "hybrid", "rerank"]
types = sorted({q["type"] for q in qs})

print(f"{'mode':<9}" + "".join(f"{t:<14}" for t in types))
for mode in MODES:
    res = {}
    for q in qs:
        user = admin if q["office"] is None else office_user(q["office"])
        _, metas = retrieve(q["question"], user, mode, n=5)
        got = [m["dossier_no"] for m in metas]
        exp = set(q["expected_ids"])
        if q["type"] == "unanswerable":
            s = 1.0 if not got else 0.0
        elif len(exp) > 5:
            s = sum(g in exp for g in got) / max(len(got), 1)     # precision@5
        else:
            s = 1.0 if exp & set(got) else 0.0                    # hit@5
        res.setdefault(q["type"], []).append(s)
    print(f"{mode:<9}" + "".join(f"{sum(res[t])/len(res[t]):<14.2f}" for t in types))
