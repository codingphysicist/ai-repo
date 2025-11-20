import os
import subprocess
import tempfile
import shutil
import requests
import json
import time
import re
from datetime import datetime, timedelta
from pathlib import Path
import lizard
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in environment")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
USE_LOCAL = os.getenv("USE_LOCAL_EMBEDDINGS", "0") in ("1", "true", "True")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", None)
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

# Attempt to support both new and old OpenAI SDKs
try:
    from openai import OpenAI as OpenAIClient
    _OPENAI_CLIENT = OpenAIClient(api_key=OPENAI_API_KEY)
    _OPENAI_NEW = True
except Exception:
    import openai as _old_openai
    _old_openai.api_key = OPENAI_API_KEY
    _OPENAI_CLIENT = _old_openai
    _OPENAI_NEW = False

# Optional local embeddings via sentence-transformers
_local_embed = None
if USE_LOCAL:
    try:
        from sentence_transformers import SentenceTransformer
        _local_embed = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        _local_embed = None

# --- Git utilities ---------------------------------------------------------
def clone_repo(repo_url):
    tmpdir = tempfile.mkdtemp(prefix="repo_")
    cmd = ["git", "clone", "--depth", "1", repo_url, tmpdir]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmpdir

def cleanup_repo(tmpdir):
    try:
        shutil.rmtree(tmpdir)
    except Exception:
        pass

def parse_owner_repo(repo_url):
    m = re.search(r'github\.com[:/]+([^/]+)/([^/]+?)(?:\.git|/|$)', repo_url)
    if not m:
        raise ValueError("Can't parse owner/repo from url: " + repo_url)
    return m.group(1), m.group(2)

def iso_30_days_ago():
    return (datetime.utcnow() - timedelta(days=30)).isoformat() + "Z"

# --- GitHub API helpers ---------------------------------------------------
def get_repo_api_info(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    d = r.json()
    return {
        "full_name": d.get("full_name"),
        "stars": d.get("stargazers_count", 0),
        "forks": d.get("forks_count", 0),
        "open_issues": d.get("open_issues_count", 0),
        "created_at": d.get("created_at"),
        "updated_at": d.get("updated_at"),
        "default_branch": d.get("default_branch", "main"),
    }

def get_recent_commits_count(owner, repo, since_iso):
    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    params = {"since": since_iso, "per_page": 100}
    r = requests.get(url, headers=HEADERS, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    return len(data)

def count_contributors(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/contributors"
    params = {"per_page": 100}
    r = requests.get(url, headers=HEADERS, params=params, timeout=20)
    if r.status_code == 204:
        return 0
    r.raise_for_status()
    data = r.json()
    return len(data) if isinstance(data, list) else 0

def estimate_pr_merge_ratio(owner, repo):
    merged_q = f"repo:{owner}/{repo} is:pr is:merged"
    total_q = f"repo:{owner}/{repo} is:pr"
    base = "https://api.github.com/search/issues"
    r1 = requests.get(base, headers=HEADERS, params={"q": merged_q}, timeout=20)
    r2 = requests.get(base, headers=HEADERS, params={"q": total_q}, timeout=20)
    if r1.status_code != 200 or r2.status_code != 200:
        return 1.0
    merged = r1.json().get("total_count", 0)
    total = r2.json().get("total_count", 0)
    if total == 0:
        return 1.0
    return merged / total

def get_closed_issues_count(owner, repo):
    q = f"repo:{owner}/{repo} is:issue is:closed"
    url = "https://api.github.com/search/issues"
    r = requests.get(url, headers=HEADERS, params={"q": q}, timeout=20)
    if r.status_code != 200:
        return 0
    return r.json().get("total_count", 0)

# --- Complexity analysis --------------------------------------------------
def analyze_complexity(repo_dir):
    """
    Complexity analysis compatible with lizard versions that do NOT have analyze_path.
    Scans files using analyze_file and aggregates results.
    """
    MAX_FILES = 500
    functions = []
    try:
        files = find_code_files(repo_dir)
    except Exception:
        files = []
    files = files[:MAX_FILES]

    for fpath in files:
        try:
            res = lizard.analyze_file(fpath)
            for fn in getattr(res, "function_list", []):
                functions.append({
                    "filename": getattr(fn, "filename", fpath),
                    "name": getattr(fn, "name", getattr(fn, "long_name", "<unknown>")),
                    "complexity": getattr(fn, "cyclomatic_complexity", getattr(fn, "complexity", 0)),
                    "nloc": getattr(fn, "length", getattr(fn, "nloc", 0))
                })
        except Exception:
            continue

    complexities = [f["complexity"] for f in functions] or [0]
    avg_complexity = sum(complexities) / len(complexities) if complexities else 0
    max_complexity = max(complexities) if complexities else 0

    return {
        "functions": functions,
        "avg_complexity": avg_complexity,
        "max_complexity": max_complexity,
        "function_count": len(functions)
    }
# --- File listing & chunking ----------------------------------------------
CODE_EXTS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".go": "go",
    ".rs": "rust",
    ".php": "php",
    ".cs": "csharp"
}

def find_code_files(repo_dir):
    out = []
    for p in Path(repo_dir).rglob("*"):
        if p.is_file() and p.suffix.lower() in CODE_EXTS:
            out.append(str(p))
    return out

def ext_to_lang(path):
    return CODE_EXTS.get(Path(path).suffix.lower(), "text")

def read_file(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        with open(path, "r", errors="ignore") as f:
            return f.read()

def chunk_file_by_language(path, language):
    code = read_file(path)
    if not code or not code.strip():
        return []
    if language == "python":
        parts = re.split(r'\n(?=def |class )', code)
        return [p for p in parts if p.strip()]
    if language in ("javascript", "typescript"):
        parts = re.split(r'\n(?=function |const |let |class )', code)
        return [p for p in parts if p.strip()]
    # fallback: whole file
    return [code]

# --- Embeddings & index ---------------------------------------------------
def _normalize_vec(vec):
    v = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v

def get_embedding(text, model=None):
    model = model or EMBED_MODEL
    text = (text or "").strip()
    if not text:
        return None
    if len(text) > 3000:
        text = text[:3000]
    # Local embeddings path (optional)
    if USE_LOCAL and _local_embed is not None:
        vec = _local_embed.encode(text)
        return _normalize_vec(vec)
    # OpenAI embeddings path
    if _OPENAI_NEW:
        resp = _OPENAI_CLIENT.embeddings.create(model=model, input=text)
        vec = resp.data[0].embedding
        return _normalize_vec(vec)
    else:
        resp = _OPENAI_CLIENT.Embedding.create(input=text, model=model)
        vec = resp["data"][0]["embedding"]
        return _normalize_vec(vec)

def batch_get_embeddings(texts, model=None, batch_size=16):
    model = model or EMBED_MODEL
    out = []
    if not texts:
        return out
    # Local batch
    if USE_LOCAL and _local_embed is not None:
        embs = _local_embed.encode(texts, batch_size=batch_size)
        for vec in embs:
            out.append(_normalize_vec(vec))
        return out
    # OpenAI batch
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        if _OPENAI_NEW:
            resp = _OPENAI_CLIENT.embeddings.create(model=model, input=batch)
            for item in resp.data:
                out.append(_normalize_vec(item.embedding))
        else:
            resp = _OPENAI_CLIENT.Embedding.create(input=batch, model=model)
            for item in resp["data"]:
                out.append(_normalize_vec(item["embedding"]))
        time.sleep(0.1)
    return out

# --- Faiss optional -------------------------------------------------------
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False

def build_faiss_index(vectors):
    arr = np.vstack(vectors).astype('float32')
    d = arr.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(arr)
    return index

# --- Scoring --------------------------------------------------------------
def normalize(x, minv, maxv):
    try:
        x = float(x)
    except Exception:
        return 0.0
    if x <= minv:
        return 0.0
    if x >= maxv:
        return 1.0
    return (x - minv) / (maxv - minv)

def compute_health_score(signals):
    commits_score = normalize(signals.get("commits30",0), 0, 100) * 100
    issues_open = signals.get("issues_open", 0)
    issues_closed = signals.get("issues_closed", 0)
    issue_ratio = issues_closed / max(1, issues_open + issues_closed)
    issue_score = issue_ratio * 100
    popularity_score = normalize(signals.get("stars",0), 0, 2000) * 100
    avg_cc = signals.get("avg_cc", 1)
    code_quality_score = max(0.0, 100.0 - normalize(avg_cc, 1, 20)*100.0)
    contributor_score = normalize(signals.get("contributors_count",0), 0, 20) * 100
    maintenance_score = signals.get("pr_merged_ratio", 1.0) * 100

    health = (0.25*commits_score + 0.2*maintenance_score + 0.15*popularity_score + 0.25*code_quality_score + 0.15*contributor_score)
    breakdown = {
        "commits_score": round(commits_score,2),
        "issue_score": round(issue_score,2),
        "popularity_score": round(popularity_score,2),
        "code_quality_score": round(code_quality_score,2),
        "contributor_score": round(contributor_score,2),
        "maintenance_score": round(maintenance_score,2)
    }
    return round(health,2), breakdown

# --- Utils for saving/loading index --------------------------------------
def save_index_faiss(index, metas, path_prefix):
    faiss.write_index(index, f"{path_prefix}.index")
    with open(f"{path_prefix}.meta.json", "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)

def load_index_faiss(path_prefix):
    idx_path = f"{path_prefix}.index"
    meta_path = f"{path_prefix}.meta.json"
    if not os.path.exists(idx_path) or not os.path.exists(meta_path):
        return None, None
    index = faiss.read_index(idx_path)
    metas = json.load(open(meta_path, "r", encoding="utf-8"))
    return index, metas

def save_index_npy(vecs, metas, path_prefix):
    np.save(f"{path_prefix}.npy", np.vstack(vecs))
    with open(f"{path_prefix}.meta.json", "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)

def load_index_npy(path_prefix):
    arr_path = f"{path_prefix}.npy"
    meta_path = f"{path_prefix}.meta.json"
    if not os.path.exists(arr_path) or not os.path.exists(meta_path):
        return None, None
    arr = np.load(arr_path)
    metas = json.load(open(meta_path, "r", encoding="utf-8"))
    return arr, metas

# --- Nearest neighbor search (faiss or brute) -----------------------------
def nn_search(vec, index_dict, top_k=5):
    if vec is None:
        return []
    if _HAS_FAISS and isinstance(index_dict[0], faiss.Index):
        index, metas = index_dict
        D, I = index.search(vec.reshape(1, -1).astype('float32'), top_k)
        res = []
        for score, idx in zip(D[0], I[0]):
            res.append({"score": float(score), "meta": metas[idx]})
        return res
    else:
        arr, metas = index_dict
        sims = arr.dot(vec)
        idxs = np.argsort(-sims)[:top_k]
        res = []
        for idx in idxs:
            res.append({"score": float(sims[idx]), "meta": metas[idx]})
        return res
