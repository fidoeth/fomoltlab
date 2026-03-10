#!/usr/bin/env python3
"""
Evaluation harness for token classifier.

Loads the labeled dataset, runs the classifier from strategy.py,
and computes performance metrics. Supports optional sell_signal()
for full trade lifecycle simulation.

Usage:
    python3 evaluate.py
"""

import json
import sys
import time
import math
import hashlib
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
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
        # Deterministic split based on address hash (hashlib for cross-run stability)
        h = int(hashlib.md5((addr + str(seed)).encode()).hexdigest(), 16) % 100
        if h < test_ratio * 100:
            test.append(entry)
        else:
            train.append(entry)
    return train, test


def load_candles(address):
    """Load OHLCV candles from raw data file."""
    raw_file = RAW_DIR / f"{address}.json"
    if not raw_file.exists():
        return None
    try:
        with open(raw_file) as f:
            raw = json.load(f)
        candles = raw.get("ohlcv", [])
        return [c for c in candles if c.get("c", 0) > 0]
    except (json.JSONDecodeError, Exception):
        return None


def simulate_trade(candles, features, sell_fn):
    """
    Simulate a trade through OHLCV candles with a sell strategy.

    Steps through candles one at a time after entry, calling sell_fn
    at each step to decide whether to exit.

    Returns dict with exit details, or None if simulation fails.
    """
    if not candles or len(candles) < 2:
        return None

    entry_price = candles[0]["c"]
    if entry_price <= 0:
        return None

    peak_price = entry_price
    peak_candle = 0

    for i, candle in enumerate(candles[1:], 1):
        current_price = candle["c"]
        if current_price <= 0:
            continue

        if current_price > peak_price:
            peak_price = current_price
            peak_candle = i

        position = {
            "entry_price": entry_price,
            "current_price": current_price,
            "high": candle.get("h", current_price),
            "low": candle.get("l", current_price),
            "unrealized_return_pct": (current_price / entry_price - 1) * 100,
            "peak_return_pct": (peak_price / entry_price - 1) * 100,
            "drawdown_from_peak_pct": (peak_price - current_price) / peak_price * 100 if peak_price > 0 else 0,
            "candles_held": i,
            "candles_since_peak": i - peak_candle,
            "volume": candle.get("v", 0),
            "volume_usd": candle.get("v_usd", 0),
            "total_candles": len(candles),
        }

        try:
            if sell_fn(features, position):
                return {
                    "exit_price": current_price,
                    "exit_return_pct": position["unrealized_return_pct"],
                    "candles_held": i,
                    "peak_return_pct": position["peak_return_pct"],
                    "exit_reason": "sell_signal",
                }
        except Exception as e:
            print(f"  [warn] sell_signal error: {e}", file=sys.stderr)

    # Held to end
    final_price = candles[-1]["c"]
    return {
        "exit_price": final_price,
        "exit_return_pct": (final_price / entry_price - 1) * 100 if entry_price > 0 else 0,
        "candles_held": len(candles) - 1,
        "peak_return_pct": (peak_price / entry_price - 1) * 100 if entry_price > 0 else 0,
        "exit_reason": "hold_to_end",
    }


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

    # Check if strategy has sell_signal
    has_sell = hasattr(strategy_module, "sell_signal")
    if has_sell:
        print("[info] sell_signal detected — simulating full trade lifecycle", file=sys.stderr)

    # Run classifier on test set
    results = []
    candle_load_failures = 0
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
        hold_return = outcome.get("final_return_pct", 0)
        max_return = outcome.get("max_return_pct", 0)
        label = outcome.get("label", "unknown")

        # Default: use hold-to-end return
        actual_return = hold_return
        trade_result = None

        # If strategy has sell logic, simulate the trade candle-by-candle
        if buy_signal and has_sell:
            address = features.get("address", "")
            candles = load_candles(address)
            if candles:
                trade_result = simulate_trade(candles, features, strategy_module.sell_signal)
                if trade_result:
                    actual_return = trade_result["exit_return_pct"]
            else:
                candle_load_failures += 1

        results.append({
            "address": features.get("address", ""),
            "symbol": features.get("symbol", ""),
            "score": score,
            "buy_signal": buy_signal,
            "actual_return_pct": actual_return,
            "hold_return_pct": hold_return,
            "max_return_pct": max_return,
            "label": label,
            "trade_result": trade_result,
        })

    if has_sell and candle_load_failures > 0:
        print(f"  [warn] could not load candles for {candle_load_failures} bought tokens (using hold return)",
              file=sys.stderr)

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

    # Sell strategy metrics
    if has_sell and num_buys > 0:
        trades_with_sim = [r for r in buy_signals if r.get("trade_result")]
        if trades_with_sim:
            avg_hold = sum(r["trade_result"]["candles_held"] for r in trades_with_sim) / len(trades_with_sim)
            sell_triggered = sum(1 for r in trades_with_sim if r["trade_result"]["exit_reason"] == "sell_signal")
            sell_pct = sell_triggered / len(trades_with_sim) * 100

            # Capture ratio: how much of peak return was captured
            capture_ratios = []
            for r in trades_with_sim:
                peak = r["trade_result"]["peak_return_pct"]
                actual = r["trade_result"]["exit_return_pct"]
                if peak > 10:  # only meaningful for tokens that had real gains
                    capture_ratios.append(actual / peak)
            avg_capture = sum(capture_ratios) / len(capture_ratios) if capture_ratios else 0

            # Compare sell returns vs hold returns
            hold_returns = [r["hold_return_pct"] for r in trades_with_sim]
            sell_returns = [r["actual_return_pct"] for r in trades_with_sim]
            hold_avg = sum(hold_returns) / len(hold_returns)
            sell_avg = sum(sell_returns) / len(sell_returns)
            sell_edge = sell_avg - hold_avg

            print(f"\n--- Sell Strategy Metrics ---")
            print(f"avg_hold_candles: {avg_hold:.1f}")
            print(f"sell_triggered:   {sell_pct:.1f}%")
            print(f"capture_ratio:    {avg_capture:.3f}")
            print(f"sell_vs_hold:     {sell_edge:+.2f}%")
            print(f"hold_avg_return:  {hold_avg:.2f}")
            print(f"sell_avg_return:  {sell_avg:.2f}")
            print(f"trades_simulated: {len(trades_with_sim)} / {num_buys}")

            # Per-label sell effectiveness
            print(f"\n--- Sell Impact by Label ---")
            for label in ["moon", "up", "crab", "down", "pump_dump", "rug"]:
                label_trades = [r for r in trades_with_sim if r["label"] == label]
                if label_trades:
                    lh = sum(r["hold_return_pct"] for r in label_trades) / len(label_trades)
                    ls = sum(r["actual_return_pct"] for r in label_trades) / len(label_trades)
                    sold = sum(1 for r in label_trades if r["trade_result"]["exit_reason"] == "sell_signal")
                    print(f"  {label:>10s}: {len(label_trades):2d} trades, hold={lh:+7.1f}%, sell={ls:+7.1f}%, edge={ls-lh:+7.1f}%, sold={sold}/{len(label_trades)}")

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
            extra = ""
            if r.get("trade_result") and r["trade_result"]["exit_reason"] == "sell_signal":
                extra = f"  sold@{r['trade_result']['candles_held']}h"
            print(f"  {r['symbol']:>10s}  score={r['score']:.3f}  return={r['actual_return_pct']:+.1f}%  label={r['label']}{extra}")

    # Worst picks
    if buy_signals:
        print("\n--- Worst Picks (biggest losses) ---")
        worst = sorted(buy_signals, key=lambda r: r["actual_return_pct"])[:5]
        for r in worst:
            extra = ""
            if r.get("trade_result") and r["trade_result"]["exit_reason"] == "sell_signal":
                extra = f"  sold@{r['trade_result']['candles_held']}h"
            print(f"  {r['symbol']:>10s}  score={r['score']:.3f}  return={r['actual_return_pct']:+.1f}%  label={r['label']}{extra}")


if __name__ == "__main__":
    # Import strategy module
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        import strategy
    except ImportError as e:
        print(f"ERROR: Could not import strategy.py: {e}", file=sys.stderr)
        sys.exit(1)

    evaluate(strategy)
