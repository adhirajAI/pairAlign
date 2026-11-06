#!/usr/bin/env python3
import argparse
import os
import torch
from typing import Any, Dict, List


def sizeof_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def analyze_cache(path: str) -> Dict[str, Any]:
    d = torch.load(path, map_location="cpu")

    segments = d["segments"]          # list of dicts
    tokens = d["tokens"]              # list[list[int]]

    num_segments = len(segments)
    num_token_lists = len(tokens)
    token_lengths = [len(t) for t in tokens]

    total_tokens = sum(token_lengths)
    avg_tokens = total_tokens / max(num_token_lists, 1)
    min_tokens = min(token_lengths) if token_lengths else 0
    max_tokens = max(token_lengths) if token_lengths else 0

    # crude metadata estimates
    total_phone_chars = sum(len(s.get("phone_str", "")) for s in segments)
    total_num_phones = sum(len(s.get("phones", [])) for s in segments)
    total_num_source_files = sum(len(s.get("source_files", [])) for s in segments)

    return {
        "path": path,
        "file_size_mb": sizeof_mb(path),
        "num_segments": num_segments,
        "num_token_lists": num_token_lists,
        "total_tokens": total_tokens,
        "avg_tokens_per_segment": avg_tokens,
        "min_tokens_per_segment": min_tokens,
        "max_tokens_per_segment": max_tokens,
        "total_phone_chars": total_phone_chars,
        "total_num_phones": total_num_phones,
        "total_num_source_files_refs": total_num_source_files,
    }


def compare_caches(path_a: str, path_b: str) -> None:
    a = analyze_cache(path_a)
    b = analyze_cache(path_b)

    print("\n=== CACHE A ===")
    for k, v in a.items():
        print(f"{k}: {v}")

    print("\n=== CACHE B ===")
    for k, v in b.items():
        print(f"{k}: {v}")

    print("\n=== DIFFERENCE (B - A) ===")
    for key in [
        "file_size_mb",
        "num_segments",
        "num_token_lists",
        "total_tokens",
        "avg_tokens_per_segment",
        "min_tokens_per_segment",
        "max_tokens_per_segment",
        "total_phone_chars",
        "total_num_phones",
        "total_num_source_files_refs",
    ]:
        print(f"{key}: {b[key] - a[key]}")

    size_drop_mb = a["file_size_mb"] - b["file_size_mb"]
    size_drop_pct = 100.0 * size_drop_mb / max(a["file_size_mb"], 1e-12)

    print("\n=== STORAGE DROP ===")
    print(f"Absolute drop: {size_drop_mb:.4f} MB")
    print(f"Percent drop : {size_drop_pct:.2f}%")

    if a["total_tokens"] > 0 and b["total_tokens"] > 0:
        token_drop = a["total_tokens"] - b["total_tokens"]
        token_drop_pct = 100.0 * token_drop / a["total_tokens"]
        print(f"Token count drop: {token_drop} ({token_drop_pct:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two retrieval archive caches.")
    parser.add_argument("cache_a")
    parser.add_argument("cache_b")
    args = parser.parse_args()
    compare_caches(args.cache_a, args.cache_b)