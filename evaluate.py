#!/usr/bin/env python3
"""
Evaluation harness for token classifier.
DO NOT MODIFY — this is the fixed evaluation script.

Loads the labeled dataset, runs the classifier from strategy.py,
and computes performance metrics.

Usage:
    python3 evaluate.py
"""

import json
import sys
import time
import math
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATASET_FILE = DATA_DIR / "dataset.jsonl"

# How much simulated SOL to "buy" per trade
BUY_AMOUNT_SOL = 0.1


def load_dataset():
    """Load the labeled dataset."""
    if not DATASET_FILE.exists():
        print("ERROR: No dataset found. Run collect.py first.", file=sys.stderr)
        sys.exit(1)

    entries = []
    with open(DATASET_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                # Must have features and a valid outcome
                if entry.get("features") and entry.get("outcome"):
                    entries.append(entry)
            except json.JSONDecodeError:
                continue

    if not entries:
        print("ERROR: Dataset is empty or has no labeled entries.", file=sys.stderr)
        sys.exit(1)

    return entries


def split_dataset(entries, test_ratio=0.3, seed=42):
    """
    Split dataset into train and test sets.
    Uses a deterministic hash-based split so results are reproducible
    regardless of dataset order.
    """
    train, test = [], []
    for entry in entries:
        addr = entry["features"].get("address", "")
        # Deterministic split based on address hash
        h = hash(addr + str(seed)) % 100
        if h < test_ratio * 100:
            test.append(entry)
        else:
            train.append(entry)
    return train, test


def evaluate(strategy_module):
    """Run evaluation and print results."""
    entries = load_dataset()
    train_set, test_set = split_dataset(entries)

    if not test_set:
        print("ERROR: Test set is empty.", file=sys.stderr)
        sys.exit(1)

    # Let strategy see training data if it wants to calibrate
    if hasattr(strategy_module, "calibrate"):
        strategy_module.calibrate(train_set)

    # Run classifier on test set
    results = []
    for entry in test_set:
        features = entry["features"]
        outcome = entry["outcome"]

        try:
            score = strategy_module.score_token(features)
        except Exception as e:
            print(f"  [warn] score_token error on {features.get('symbol', '?')}: {e}",
                  file=sys.stderr)
            score = 0.0

        buy_signal = score > 0.5
        actual_return = outcome.get("final_return_pct", 0)
        max_return = outcome.get("max_return_pct", 0)
        label = outcome.get("label", "unknown")

        results.append({
            "address": features.get("address", ""),
            "symbol": features.get("symbol", ""),
            "score": score,
            "buy_signal": buy_signal,
            "actual_return_pct": actual_return,
            "max_return_pct": max_return,
            "label": label,
        })

    # Compute metrics
    total_tokens = len(results)
    buy_signals = [r for r in results if r["buy_signal"]]
    skip_signals = [r for r in results if not r["buy_signal"]]
    num_buys = len(buy_signals)
    num_skips = len(skip_signals)

    # PnL metrics (on tokens we'd buy)
    if num_buys > 0:
        buy_returns = [r["actual_return_pct"] for r in buy_signals]
        avg_return = sum(buy_returns) / len(buy_returns)
        wins = sum(1 for r in buy_returns if r > 0)
        win_rate = wins / len(buy_returns) * 100
        total_pnl = sum(buy_returns)

        # Simulated portfolio: buy BUY_AMOUNT_SOL of each
        simulated_pnl_sol = sum(
            BUY_AMOUNT_SOL * (r / 100) for r in buy_returns
        )

        # Best and worst trades
        best_return = max(buy_returns)
        worst_return = min(buy_returns)
        median_return = sorted(buy_returns)[len(buy_returns) // 2]

        # Sharpe-like ratio (return / volatility)
        if len(buy_returns) > 1:
            mean_r = avg_return
            variance = sum((r - mean_r) ** 2 for r in buy_returns) / (len(buy_returns) - 1)
            std_r = math.sqrt(variance) if variance > 0 else 0.001
            sharpe = mean_r / std_r
        else:
            sharpe = 0
    else:
        avg_return = 0
        win_rate = 0
        total_pnl = 0
        simulated_pnl_sol = 0
        best_return = 0
        worst_return = 0
        median_return = 0
        sharpe = 0

    # Classification accuracy (moon/up = good, rug/down/pump_dump = bad)
    good_labels = {"moon", "up"}
    bad_labels = {"rug", "down", "pump_dump"}

    true_positives = sum(1 for r in results if r["buy_signal"] and r["label"] in good_labels)
    false_positives = sum(1 for r in results if r["buy_signal"] and r["label"] in bad_labels)
    false_negatives = sum(1 for r in results if not r["buy_signal"] and r["label"] in good_labels)
    true_negatives = sum(1 for r in results if not r["buy_signal"] and r["label"] in bad_labels)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Market baseline (what if we bought everything?)
    all_returns = [r["actual_return_pct"] for r in results]
    market_avg = sum(all_returns) / len(all_returns) if all_returns else 0

    # Alpha = our avg return - market avg
    alpha = avg_return - market_avg

    # Print results in autoresearch format
    print("---")
    print(f"avg_return:       {avg_return:.2f}")
    print(f"alpha:            {alpha:.2f}")
    print(f"win_rate:         {win_rate:.1f}")
    print(f"sharpe:           {sharpe:.3f}")
    print(f"num_buys:         {num_buys}")
    print(f"num_skips:        {num_skips}")
    print(f"total_tokens:     {total_tokens}")
    print(f"precision:        {precision:.3f}")
    print(f"recall:           {recall:.3f}")
    print(f"f1:               {f1:.3f}")
    print(f"simulated_pnl:    {simulated_pnl_sol:.4f}")
    print(f"best_trade:       {best_return:.2f}")
    print(f"worst_trade:      {worst_return:.2f}")
    print(f"median_return:    {median_return:.2f}")
    print(f"market_avg:       {market_avg:.2f}")

    # Per-label breakdown
    print("\n--- Label Breakdown ---")
    for label in ["moon", "up", "crab", "down", "pump_dump", "rug"]:
        in_label = [r for r in results if r["label"] == label]
        bought = [r for r in in_label if r["buy_signal"]]
        if in_label:
            print(f"  {label:>10s}: {len(in_label):3d} total, {len(bought):3d} bought ({len(bought)/len(in_label)*100:5.1f}%)")

    # Top picks (highest scored tokens that we'd buy)
    if buy_signals:
        print("\n--- Top Picks (by score) ---")
        top = sorted(buy_signals, key=lambda r: r["score"], reverse=True)[:5]
        for r in top:
            print(f"  {r['symbol']:>10s}  score={r['score']:.3f}  return={r['actual_return_pct']:+.1f}%  label={r['label']}")

    # Worst picks
    if buy_signals:
        print("\n--- Worst Picks (biggest losses) ---")
        worst = sorted(buy_signals, key=lambda r: r["actual_return_pct"])[:5]
        for r in worst:
            print(f"  {r['symbol']:>10s}  score={r['score']:.3f}  return={r['actual_return_pct']:+.1f}%  label={r['label']}")


if __name__ == "__main__":
    # Import strategy module
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        import strategy
    except ImportError as e:
        print(f"ERROR: Could not import strategy.py: {e}", file=sys.stderr)
        sys.exit(1)

    evaluate(strategy)
