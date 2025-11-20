import json, os
REPORT = os.getenv("REPORT_FILE","report.json")
OUT = os.getenv("SUMMARY_OUT","summary.json")
THRESH = float(os.getenv("SIMILARITY_THRESHOLD","0.70"))
MIN_CHARS = int(os.getenv("MIN_CHUNK_CHARS","40"))
MAX_SIM = int(os.getenv("MAX_SIM_OUT","10"))

r = json.load(open(REPORT,"r", encoding="utf-8"))
summary = {
    "repo": r.get("repo"),
    "health_score": r.get("health_score"),
    "top_complexity": [],
    "top_similarity": []
}
funcs = r.get("complexity",{}).get("functions",[])
funcs_sorted = sorted(funcs, key=lambda x: x.get("complexity",0), reverse=True)[:5]
summary["top_complexity"] = [{"file":f["filename"], "name":f["name"], "cc":f["complexity"], "nloc":f["nloc"]} for f in funcs_sorted]

sims = []
for s in r.get("similarity",[]):
    if len(s.get("chunk_preview","")) < MIN_CHARS:
        continue
    top = s.get("top_matches",[])
    if not top:
        continue
    best = float(top[0].get("score",0.0))
    if best < THRESH:
        continue
    sims.append({
        "score": best,
        "target_file": s.get("file"),
        "target_preview": s.get("chunk_preview")[:300],
        "origin_repo": top[0]["meta"].get("repo"),
        "origin_file": top[0]["meta"].get("file"),
        "origin_snippet": top[0]["meta"].get("snippet")[:300]
    })
sims = sorted(sims, key=lambda x: x["score"], reverse=True)[:MAX_SIM]
summary["top_similarity"] = sims

open(OUT,"w", encoding="utf-8").write(json.dumps(summary, indent=2))
print("Wrote", OUT)
