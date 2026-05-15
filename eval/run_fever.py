"""Evaluate the factforge agent against FEVER's dev set.

Loads the FEVER shared_task_dev.jsonl, samples N claims balanced across
the three classes (SUPPORTS / REFUTES / NOT ENOUGH INFO), calls the
live /api/v1/claims endpoint for each, and computes accuracy,
per-class P/R/F1, confusion matrix, and latency stats.

Run:
    cd /Users/asutoshpaluri/Documents/factforge
    uv run --project backend python eval/run_fever.py --n 30

Output:
    - Live progress per claim
    - Final metrics printed to stdout
    - Detailed results saved to eval/fever_results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import httpx

# --- Config ---
DEFAULT_FEVER_PATH = Path.home() / "Documents/FEVER Dataset/shared_task_dev.jsonl"
DEFAULT_API_URL = "http://127.0.0.1:8000/api/v1/claims"
DEFAULT_SAMPLE_SIZE = 30
DEFAULT_SEED = 42
CLAIM_TIMEOUT_S = 180.0  # per claim — agent runs include model loading on cold start

# FEVER -> factforge 3-class label mapping
LABEL_MAP = {
    "SUPPORTS": "Real",
    "REFUTES": "Disinformation",
    "NOT ENOUGH INFO": "Misinformation",
}
CLASSES = ["Real", "Misinformation", "Disinformation"]


def load_fever(path: Path) -> list[dict]:
    """Load FEVER dev jsonl into a list of dicts."""
    if not path.exists():
        sys.exit(f"FEVER file not found: {path}")
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def balanced_sample(items: list[dict], n: int, seed: int) -> list[dict]:
    """Sample ~n/3 claims from each FEVER class for balanced metrics."""
    by_label: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        lbl = it.get("label")
        if lbl in LABEL_MAP:
            by_label[lbl].append(it)

    rng = random.Random(seed)
    per_class = n // 3
    extra = n - per_class * 3  # spread the remainder across classes

    sample: list[dict] = []
    for i, lbl in enumerate(LABEL_MAP.keys()):
        take = per_class + (1 if i < extra else 0)
        available = by_label.get(lbl, [])
        if len(available) < take:
            print(f"[warn] only {len(available)} claims for {lbl}, using all")
            take = len(available)
        sample.extend(rng.sample(available, take))

    rng.shuffle(sample)  # interleave classes
    return sample


async def evaluate_claim(
    client: httpx.AsyncClient, api_url: str, claim: str
) -> dict:
    """Call the API for one claim. Returns the parsed response or error dict."""
    t0 = time.time()
    try:
        r = await client.post(api_url, json={"claim": claim}, timeout=CLAIM_TIMEOUT_S)
        r.raise_for_status()
        return {"ok": True, "data": r.json(), "elapsed_s": time.time() - t0}
    except Exception as e:
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "elapsed_s": time.time() - t0,
        }


def fmt_pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


def compute_metrics(results: list[dict]) -> dict:
    """Compute accuracy, per-class P/R/F1, confusion matrix, latency."""
    successful = [r for r in results if r["predicted"] is not None]
    correct = sum(1 for r in successful if r["correct"])
    total = len(successful)
    accuracy = correct / total if total else 0.0

    # Per-class P / R / F1
    per_class: dict[str, dict[str, float]] = {}
    for cls in CLASSES:
        tp = sum(1 for r in successful if r["true_label"] == cls and r["predicted"] == cls)
        fp = sum(1 for r in successful if r["true_label"] != cls and r["predicted"] == cls)
        fn = sum(1 for r in successful if r["true_label"] == cls and r["predicted"] != cls)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for r in successful if r["true_label"] == cls),
        }
    macro_f1 = (
        sum(per_class[c]["f1"] for c in CLASSES) / len(CLASSES) if per_class else 0.0
    )

    # Confusion matrix (rows = true, cols = predicted)
    confusion: dict[str, dict[str, int]] = {
        t: {p: 0 for p in CLASSES} for t in CLASSES
    }
    for r in successful:
        confusion[r["true_label"]][r["predicted"]] += 1

    # Latency
    latencies_ms = sorted([r["duration_ms"] for r in successful if r["duration_ms"]])
    if latencies_ms:
        n = len(latencies_ms)
        latency = {
            "mean_s": sum(latencies_ms) / n / 1000,
            "p50_s": latencies_ms[n // 2] / 1000,
            "p95_s": latencies_ms[min(int(n * 0.95), n - 1)] / 1000,
            "min_s": latencies_ms[0] / 1000,
            "max_s": latencies_ms[-1] / 1000,
        }
    else:
        latency = {}

    return {
        "n_total": len(results),
        "n_successful": total,
        "n_errors": len(results) - total,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion": confusion,
        "latency": latency,
    }


def print_metrics(m: dict) -> None:
    print()
    print("=" * 60)
    print("FEVER evaluation results")
    print("=" * 60)
    print(
        f"Samples: {m['n_successful']}/{m['n_total']} successful "
        f"({m['n_errors']} errors)"
    )
    print(f"Accuracy:  {fmt_pct(m['accuracy'])}")
    print(f"Macro F1:  {fmt_pct(m['macro_f1'])}")
    print()

    # Per-class table
    print(f"{'Class':<18} {'Precision':>11} {'Recall':>9} {'F1':>9} {'N':>5}")
    print("-" * 56)
    for cls in CLASSES:
        pc = m["per_class"].get(cls, {})
        print(
            f"{cls:<18} {fmt_pct(pc.get('precision', 0)):>11} "
            f"{fmt_pct(pc.get('recall', 0)):>9} "
            f"{fmt_pct(pc.get('f1', 0)):>9} "
            f"{pc.get('support', 0):>5}"
        )
    print()

    # Confusion matrix
    print("Confusion matrix (rows=true, cols=predicted):")
    header = " " * 18 + "".join(f"{c[:6]:>9}" for c in CLASSES)
    print(header)
    for t in CLASSES:
        row = f"{t:<18}" + "".join(
            f"{m['confusion'][t][p]:>9}" for p in CLASSES
        )
        print(row)
    print()

    # Latency
    if m["latency"]:
        lat = m["latency"]
        print(
            f"Latency: mean={lat['mean_s']:.1f}s  "
            f"p50={lat['p50_s']:.1f}s  p95={lat['p95_s']:.1f}s  "
            f"(min {lat['min_s']:.1f}s, max {lat['max_s']:.1f}s)"
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate factforge against FEVER dev set.")
    parser.add_argument("--n", type=int, default=DEFAULT_SAMPLE_SIZE,
                        help="number of claims to sample (balanced across classes)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="random seed for sampling")
    parser.add_argument("--api-url", default=DEFAULT_API_URL,
                        help="POST endpoint for /claims")
    parser.add_argument("--fever-path", type=Path, default=DEFAULT_FEVER_PATH,
                        help="path to FEVER shared_task_dev.jsonl")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "fever_results.json",
                        help="where to save detailed results JSON")
    args = parser.parse_args()

    print(f"Loading FEVER from: {args.fever_path}")
    all_items = load_fever(args.fever_path)
    print(f"  -> {len(all_items)} dev claims total")
    label_counts = Counter(it.get("label") for it in all_items)
    print(f"  -> labels: {dict(label_counts)}")
    print()

    sample = balanced_sample(all_items, args.n, args.seed)
    sample_counts = Counter(it["label"] for it in sample)
    print(f"Sampling {len(sample)} balanced claims: {dict(sample_counts)}")
    print(f"API: {args.api_url}")
    print(f"Seed: {args.seed}")
    print()

    results: list[dict] = []
    t_start = time.time()

    async with httpx.AsyncClient() as client:
        for i, item in enumerate(sample, 1):
            claim = item["claim"]
            fever_label = item["label"]
            true_label = LABEL_MAP[fever_label]

            resp = await evaluate_claim(client, args.api_url, claim)

            if resp["ok"]:
                data = resp["data"]
                predicted = data["verdict"]
                confidence = data["confidence"]
                duration_ms = data["duration_ms"]
                error = None
            else:
                predicted = None
                confidence = 0.0
                duration_ms = resp["elapsed_s"] * 1000
                error = resp["error"]

            correct = predicted == true_label
            results.append({
                "id": item.get("id"),
                "claim": claim,
                "fever_label": fever_label,
                "true_label": true_label,
                "predicted": predicted,
                "confidence": confidence,
                "duration_ms": duration_ms,
                "correct": correct,
                "error": error,
            })

            mark = "OK" if correct else ("ER" if error else "X ")
            true_short = true_label[:6]
            pred_short = (predicted or "ERROR")[:6]
            elapsed = duration_ms / 1000
            running_acc = (
                sum(1 for r in results if r["correct"]) / len(results)
            )
            print(
                f"[{i:>2}/{len(sample)}] {mark} "
                f"{elapsed:>5.1f}s  "
                f"true={true_short:<6} pred={pred_short:<6} "
                f"acc={fmt_pct(running_acc)}  "
                f"{claim[:60]}{'...' if len(claim) > 60 else ''}"
            )

    total_s = time.time() - t_start
    print(f"\nDone in {total_s / 60:.1f} min.")

    # Metrics
    metrics = compute_metrics(results)
    print_metrics(metrics)

    # Save detailed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(
            {
                "config": {
                    "n": args.n,
                    "seed": args.seed,
                    "api_url": args.api_url,
                    "fever_path": str(args.fever_path),
                    "total_wall_clock_s": total_s,
                },
                "metrics": metrics,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nDetailed results -> {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
