#!/usr/bin/env python3
"""
analyze_repo.py

Usage:
  python analyze_repo.py https://github.com/owner/repo --out report.json

This script writes a UTF-8 JSON file (portable) and prints progress to stderr
via logging so stdout remains clean.
"""
import sys
import json
import os
import argparse
import logging
from pathlib import Path

from utils import (
    parse_owner_repo, clone_repo, cleanup_repo, iso_30_days_ago,
    get_repo_api_info, get_recent_commits_count, count_contributors,
    estimate_pr_merge_ratio, get_closed_issues_count, analyze_complexity,
    find_code_files, ext_to_lang, chunk_file_by_language, batch_get_embeddings,
    get_embedding, _HAS_FAISS, load_index_faiss, load_index_npy, nn_search,
    compute_health_score
)

from tqdm import tqdm

# CLI / defaults
DEFAULT_INDEX_PREFIX = os.getenv("INDEX_PREFIX", "corpus_index")
DEFAULT_MAX_FILES = int(os.getenv("MAX_FILES_ANALYZE", "200"))
DEFAULT_MAX_CHUNKS_PER_FILE = int(os.getenv("MAX_CHUNKS_PER_FILE", "20"))
DEFAULT_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "8"))
DEFAULT_TOP_K = int(os.getenv("ANALYZE_TOP_K", "3"))

# Configure logging to stderr (portable)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)
log = logging.getLogger("analyze_repo")

def load_index(prefix=DEFAULT_INDEX_PREFIX):
    """Return (index_obj, metas) or None if no index."""
    if _HAS_FAISS:
        idx, metas = load_index_faiss(prefix)
        if idx is not None and metas is not None:
            return (idx, metas)
        arr, metas = load_index_npy(prefix)
        if arr is not None and metas is not None:
            return (arr, metas)
        return None
    else:
        arr, metas = load_index_npy(prefix)
        if arr is not None and metas is not None:
            return (arr, metas)
        return None

def relativize_path(fp, base_dir):
    """Return path relative to base_dir where possible, else the filename."""
    try:
        return str(Path(fp).relative_to(Path(base_dir)))
    except Exception:
        return os.path.basename(fp)

def analyze(repo_url, out_path="report.json", index_prefix=DEFAULT_INDEX_PREFIX,
            max_files=DEFAULT_MAX_FILES, max_chunks_per_file=DEFAULT_MAX_CHUNKS_PER_FILE,
            batch_size=DEFAULT_BATCH_SIZE, top_k=DEFAULT_TOP_K):
    owner, repo = parse_owner_repo(repo_url)
    log.info("Cloning %s/%s ...", owner, repo)
    tmp = clone_repo(repo_url)
    report = {}
    try:
        # GitHub signals
        log.info("Fetching repository metadata from GitHub API...")
        api_info = get_repo_api_info(owner, repo)
        commits30 = get_recent_commits_count(owner, repo, iso_30_days_ago())
        contributors = count_contributors(owner, repo)
        pr_merge_ratio = estimate_pr_merge_ratio(owner, repo)
        issues_closed = get_closed_issues_count(owner, repo)

        # Complexity & files
        log.info("Analyzing code complexity (may take a few seconds)...")
        complexity = analyze_complexity(tmp)
        files = find_code_files(tmp)[:max_files]

        # Load index if available
        index = load_index(index_prefix)
        if index is None:
            log.warning("No corpus index found (similarity search disabled).")
            index_available = False
        else:
            log.info("Loaded corpus index for similarity search.")
            index_available = True

        similarity_results = []

        # Process files (chunk -> batch embed -> query)
        log.info("Processing %d files (limit=%d)...", len(files), max_files)
        for fp in tqdm(files, desc="files", file=sys.stderr):
            lang = ext_to_lang(fp)
            chunks = chunk_file_by_language(fp, lang)
            # only keep non-empty chunks and cap
            cleaned = [c for c in chunks if c and c.strip()][:max_chunks_per_file]
            if not cleaned:
                continue

            if index_available:
                # get all embeddings for this file in batches
                try:
                    embs = batch_get_embeddings(cleaned, batch_size=batch_size)
                except Exception as e:
                    # fallback to single-call embedding if batch fails
                    log.warning("batch_get_embeddings failed for %s: %s", fp, e)
                    embs = []
                    for c in cleaned:
                        try:
                            v = get_embedding(c)
                            embs.append(v)
                        except Exception:
                            embs.append(None)

                for chunk, emb in zip(cleaned, embs):
                    if emb is None:
                        continue
                    try:
                        nn = nn_search(emb, index, top_k=top_k)
                    except Exception as e:
                        log.debug("nn_search failed: %s", e)
                        continue
                    if not nn:
                        continue
                    best = nn[0]
                    similarity_results.append({
                        "file": relativize_path(fp, tmp),
                        "chunk_preview": chunk.strip()[:800],
                        "top_matches": nn
                    })

        # Compute health score
        signals = {
            "commits30": commits30,
            "stars": api_info.get("stars", 0),
            "issues_open": api_info.get("open_issues", 0),
            "issues_closed": issues_closed,
            "avg_cc": complexity.get("avg_complexity", 0),
            "contributors_count": contributors,
            "pr_merged_ratio": pr_merge_ratio
        }
        health_score, breakdown = compute_health_score(signals)

        # Build final report object
        report = {
            "repo": api_info.get("full_name"),
            "health_score": health_score,
            "breakdown": breakdown,
            "complexity": complexity,
            "similarity": similarity_results  # post-filtering / thresholding can be applied by consumer
        }

        # Write JSON to out_path in UTF-8 (portable)
        out_dir = Path(out_path).parent
        if out_dir and not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        log.info("Wrote report to %s (UTF-8).", out_path)
        return report
    finally:
        cleanup_repo(tmp)

def main():
    parser = argparse.ArgumentParser(description="Analyze a GitHub repo and produce a JSON report (UTF-8).")
    parser.add_argument("repo", help="GitHub repo URL, e.g. https://github.com/owner/repo")
    parser.add_argument("--out", "-o", default="report.json", help="Output JSON filename (default: report.json)")
    parser.add_argument("--index-prefix", default=DEFAULT_INDEX_PREFIX, help="Index prefix (default from env)")
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES, help="Max files to analyze")
    parser.add_argument("--max-chunks", type=int, default=DEFAULT_MAX_CHUNKS_PER_FILE, help="Max chunks per file")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Embedding batch size")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top-k neighbors for similarity")
    args = parser.parse_args()

    try:
        analyze(
            args.repo,
            out_path=args.out,
            index_prefix=args.index_prefix,
            max_files=args.max_files,
            max_chunks_per_file=args.max_chunks,
            batch_size=args.batch_size,
            top_k=args.top_k
        )
    except Exception as e:
        log.exception("Fatal error during analysis: %s", e)
        sys.exit(2)

if __name__ == "__main__":
    main()
