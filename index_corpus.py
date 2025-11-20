import os
import json
from utils import clone_repo, cleanup_repo, find_code_files, ext_to_lang, chunk_file_by_language, batch_get_embeddings, save_index_faiss, save_index_npy, _HAS_FAISS
import numpy as np
from tqdm import tqdm
import time

CORPUS_LIST = "corpus_list.txt"
OUT_PREFIX = "corpus_index"
MAX_CHUNKS_PER_REPO = int(os.getenv("MAX_CHUNKS_PER_REPO", "30"))  # default much lower
MAX_FILES_PER_REPO = int(os.getenv("MAX_FILES_PER_REPO", "20"))
SOURCE_FOLDER_WHITELIST = ["src", "lib", "app", "packages", "backend", "frontend"]
SKIP_FOLDERS = ["node_modules", "venv", ".venv", "build", "dist", "docs", "examples", "test", "tests"]

# cost estimate params (text-embedding-3-small pricing)
EST_AVG_TOKENS_PER_CHUNK = int(os.getenv("EST_AVG_TOKENS_PER_CHUNK", "350"))
PRICE_PER_1M_TOKENS = float(os.getenv("PRICE_PER_1M_TOKENS", "0.02"))

def file_in_whitelist(fp):
    lp = fp.replace("\\", "/").lower()
    for bad in SKIP_FOLDERS:
        if f"/{bad}/" in lp or lp.endswith("/" + bad):
            return False
    for folder in SOURCE_FOLDER_WHITELIST:
        if f"/{folder}/" in lp or lp.startswith(folder + "/"):
            return True
    # accept small top-level modules, but skip very long paths
    return False

def read_corpus_list(path):
    if not os.path.exists(path):
        print(f"[ERROR] corpus list not found at '{path}'")
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
    return lines

def estimate_cost(total_chunks):
    total_tokens = total_chunks * EST_AVG_TOKENS_PER_CHUNK
    cost = (total_tokens / 1_000_000) * PRICE_PER_1M_TOKENS
    return total_tokens, round(cost, 6)

def build_corpus_index(corpus_list_file=CORPUS_LIST, out_prefix=OUT_PREFIX):
    repos = read_corpus_list(corpus_list_file)
    if not repos:
        print("[ERROR] No repos found in corpus_list.txt. Add repo URLs (one per line).")
        return
    total_repos = len(repos)
    print(f"[INFO] Starting indexing of {total_repos} repos (max {MAX_CHUNKS_PER_REPO} chunks/repo, max {MAX_FILES_PER_REPO} files/repo)")
    # first pass: pre-scan chunks to estimate cost (cheap, no network except clone)
    estimated_total_chunks = 0
    repo_chunks_preview = {}
    for i, repo in enumerate(repos, 1):
        print(f"[SCAN {i}/{total_repos}] scanning {repo}")
        try:
            tmp = clone_repo(repo)
        except Exception as e:
            print(f"[WARN] Failed to clone {repo}: {e}")
            continue
        try:
            files = find_code_files(tmp)
            if not files:
                print(f"[INFO] No code files found in {repo}")
                continue
            chunks = []
            files_seen = 0
            for fp in files:
                if files_seen >= MAX_FILES_PER_REPO:
                    break
                if not file_in_whitelist(fp):
                    continue
                lang = ext_to_lang(fp)
                try:
                    parts = chunk_file_by_language(fp, lang)
                except Exception:
                    continue
                for p in parts:
                    snippet = p.strip()[:3000]
                    if len(snippet) < 10:
                        continue
                    chunks.append(snippet)
                    if len(chunks) >= MAX_CHUNKS_PER_REPO:
                        break
                if chunks:
                    files_seen += 1
                if len(chunks) >= MAX_CHUNKS_PER_REPO:
                    break
            repo_chunks_preview[repo] = len(chunks)
            estimated_total_chunks += len(chunks)
            print(f"[SCAN] {repo} -> {len(chunks)} chunks (capped)")
        finally:
            cleanup_repo(tmp)
        time.sleep(0.2)
    total_tokens, est_cost = estimate_cost(estimated_total_chunks)
    print(f"[ESTIMATE] total chunks: {estimated_total_chunks}, est tokens: {total_tokens}, est cost: ${est_cost}")
    print("[INFO] If estimate looks okay, indexing will proceed. Press Ctrl+C now to abort.")
    time.sleep(2)

    # second pass: actual embedding & build index
    metas = []
    vectors = []
    try:
        for i, repo in enumerate(repos, 1):
            print(f"[{i}/{total_repos}] Indexing {repo}")
            try:
                tmp = clone_repo(repo)
            except Exception as e:
                print(f"[WARN] Failed to clone {repo}: {e}")
                continue
            try:
                files = find_code_files(tmp)
                if not files:
                    print(f"[INFO] No code files found in {repo}")
                chunks = []
                chunk_meta_batch = []
                files_seen = 0
                for fp in files:
                    try:
                        if files_seen >= MAX_FILES_PER_REPO:
                            break
                        if not file_in_whitelist(fp):
                            continue
                        lang = ext_to_lang(fp)
                        parts = chunk_file_by_language(fp, lang)
                        for p in parts:
                            snippet = p.strip()[:3000]
                            if len(snippet) < 10:
                                continue
                            chunks.append(snippet)
                            chunk_meta_batch.append({"repo": repo, "file": fp, "snippet": snippet[:800]})
                            if len(chunks) >= MAX_CHUNKS_PER_REPO:
                                break
                        if parts and any(len(x.strip())>10 for x in parts):
                            files_seen += 1
                        if len(chunks) >= MAX_CHUNKS_PER_REPO:
                            break
                    except Exception as e:
                        print(f"[WARN] error chunking {fp}: {e}")
                if chunks:
                    print(f"[INFO] Creating embeddings for {len(chunks)} chunks from {repo}")
                    try:
                        vecs = batch_get_embeddings(chunks)
                    except Exception as e:
                        print(f"[ERROR] Embedding failed for {repo}: {e}")
                        vecs = []
                    for v, m in zip(vecs, chunk_meta_batch[:len(vecs)]):
                        vectors.append(v)
                        metas.append(m)
                else:
                    print(f"[INFO] no valid chunks to embed for {repo}")
            finally:
                cleanup_repo(tmp)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("[ABORT] Indexing interrupted by user. Saving partial index if any vectors exist.")
    if not vectors:
        print("[ERROR] No vectors generated; check OpenAI key, corpus and network.")
        return
    print(f"[INFO] Generated {len(metas)} chunk embeddings from {len(repos)} repos (saved)")
    if _HAS_FAISS:
        try:
            import faiss
            d = vectors[0].shape[0]
            index = faiss.IndexFlatIP(d)
            index.add(np.vstack(vectors).astype('float32'))
            save_index_faiss(index, metas, out_prefix)
            print(f"[OK] Faiss index written to {out_prefix}.index and {out_prefix}.meta.json")
        except Exception as e:
            print(f"[ERROR] Faiss index build failed: {e}. Falling back to numpy save.")
            save_index_npy(vectors, metas, out_prefix)
            print(f"[OK] Numpy index written to {out_prefix}.npy and {out_prefix}.meta.json")
    else:
        save_index_npy(vectors, metas, out_prefix)
        print(f"[OK] Numpy index written to {out_prefix}.npy and {out_prefix}.meta.json")

if __name__ == "__main__":
    try:
        build_corpus_index()
    except Exception as e:
        print("[FATAL] indexer crashed:", e)
        raise
