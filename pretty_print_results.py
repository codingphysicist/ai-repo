import json
import os
import textwrap

REPORT = os.getenv("REPORT_FILE", "report.json")
THRESH = float(os.getenv("SIMILARITY_THRESHOLD", "0.70"))
MIN_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "40"))
TOP_SIM = int(os.getenv("TOP_SIMILARITY_SHOW", "5"))
TOP_COMPLEX = int(os.getenv("TOP_COMPLEX_SHOW", "5"))

def short(s, n=320):
    s = s.replace("\n", "\\n")
    return (s[:n] + "...") if len(s) > n else s

def print_health(r):
    print("="*60)
    print(f"Repo: {r.get('repo')}")
    print(f"Health score: {r.get('health_score')}")
    for k,v in r.get("breakdown", {}).items():
        print(f"  {k:20s}: {v}")
    print("="*60)

def print_complexity(r):
    comp = r.get("complexity", {})
    funcs = comp.get("functions", [])
    if not funcs:
        print("No complexity data.")
        return
    # sort by complexity desc
    funcs_sorted = sorted(funcs, key=lambda x: x.get("complexity",0), reverse=True)
    print(f"Top {TOP_COMPLEX} complexity hotspots:")
    for i,f in enumerate(funcs_sorted[:TOP_COMPLEX],1):
        print(f"{i}. {f['filename']}  |  fn: {f['name']}  |  cc={f['complexity']}  nloc={f['nloc']}")
    print("-"*60)

def print_similarity(r):
    sims = r.get("similarity", [])
    if not sims:
        print("No similarity results.")
        return
    # flatten and filter by threshold & min chars
    filtered = []
    for s in sims:
        preview = s.get("chunk_preview","")
        if len(preview) < MIN_CHARS:
            continue
        top = s.get("top_matches", [])
        if not top:
            continue
        best = top[0]
        score = float(best.get("score",0.0))
        if score < THRESH:
            continue
        filtered.append((score, s))
    if not filtered:
        print("No high-confidence similarity matches (threshold {:.2f})".format(THRESH))
        return
    filtered.sort(key=lambda x: x[0], reverse=True)
    print(f"Top {TOP_SIM} similarity matches (threshold={THRESH}):")
    for i,(score, s) in enumerate(filtered[:TOP_SIM],1):
        meta0 = s["top_matches"][0]["meta"]
        repo = meta0.get("repo","")
        file = meta0.get("file","")
        print(f"{i}. score={score:.3f}  origin_repo={repo}")
        print("   target file:", s.get("file"))
        print("   matched file:", file)
        print("   preview:", short(s.get("chunk_preview",""), 300))
        print("   origin snippet preview:", short(meta0.get("snippet",""), 300))
        print("-"*40)

def main():
    if not os.path.exists(REPORT):
        print("Report not found:", REPORT)
        return
    r = json.load(open(REPORT,"r", encoding="utf-8"))
    print_health(r)
    print_complexity(r)
    print_similarity(r)
    print("="*60)
    print("Counts: similarity total =", len(r.get("similarity",[])), "functions =", r.get("complexity",{}).get("function_count",0))

if __name__ == "__main__":
    main()
