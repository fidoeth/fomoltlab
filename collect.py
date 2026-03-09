#!/usr/bin/env python3
"""
Token data collection from fomolt.
DO NOT MODIFY — this is the fixed data collection script.

Collects Solana token data and builds a labeled dataset for classifier training.
Each token gets: search metadata, detailed info, security audit, and OHLCV price history.
Labels are derived from OHLCV price trajectories.

Usage:
    python3 collect.py              # Collect default dataset (~200 tokens)
    python3 collect.py --count 500  # Collect larger dataset
    python3 collect.py --refresh    # Re-collect features for existing tokens
"""

import json
import subprocess
import time
import os
import sys
import argparse
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
DATASET_FILE = DATA_DIR / "dataset.jsonl"
CHAIN = "solana"

# Rate limiting
REQUEST_DELAY = 0.4  # seconds between fomolt calls


def fomolt(*args):
    """Run a fomolt command and return parsed JSON data, or None on failure."""
    cmd = ["fomolt"] + list(args)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout.strip() or result.stderr.strip()
        if not output:
            return None
        # fomolt outputs one JSON line, sometimes followed by a hint line
        first_line = output.split("\n")[0]
        data = json.loads(first_line)
        if data.get("ok"):
            return data.get("data")
        return None
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        print(f"  [warn] fomolt error: {e}", file=sys.stderr)
        return None


def search_tokens(min_age, max_age, limit=50):
    """Search for tokens in a given age range (minutes)."""
    args = [
        "token", "search",
        "-c", CHAIN,
        "--mode", "new",
        "--min-age", str(min_age),
        "--max-age", str(max_age),
        "--min-liquidity", "1000",
        "--min-holders", "10",
        "-n", str(min(limit, 100)),
    ]
    data = fomolt(*args)
    time.sleep(REQUEST_DELAY)
    if data and "tokens" in data:
        return data["tokens"]
    return []


def get_token_info(address):
    """Get detailed token info."""
    data = fomolt("token", "info", "-c", CHAIN, "-t", address)
    time.sleep(REQUEST_DELAY)
    return data


def get_token_security(address):
    """Get token security audit."""
    data = fomolt("token", "security", "-c", CHAIN, "-t", address)
    time.sleep(REQUEST_DELAY)
    return data


def get_ohlcv(address, candle_type="1H"):
    """Get OHLCV candle data."""
    data = fomolt("ohlcv", "--token", address, "--type", candle_type)
    time.sleep(REQUEST_DELAY)
    if data and "items" in data:
        return data["items"]
    return []


def label_from_ohlcv(candles):
    """
    Derive outcome label from OHLCV data.

    Uses the first non-zero candle as the entry price, then looks at
    what happened over the remaining candles.

    Returns dict with:
        entry_price, peak_price, final_price, max_return, final_return,
        max_drawdown_from_peak, label, hours_observed
    """
    # Filter to candles with actual price data
    valid = [c for c in candles if c.get("c", 0) > 0]
    if len(valid) < 3:
        return None

    # Use the first valid candle as "entry"
    entry_price = valid[0]["c"]

    # Track price evolution
    prices = [c["c"] for c in valid]
    peak_price = max(prices)
    final_price = prices[-1]

    max_return = (peak_price / entry_price - 1) * 100  # percent
    final_return = (final_price / entry_price - 1) * 100

    # Max drawdown from peak
    peak_so_far = prices[0]
    max_drawdown = 0
    for p in prices:
        if p > peak_so_far:
            peak_so_far = p
        dd = (peak_so_far - p) / peak_so_far * 100
        if dd > max_drawdown:
            max_drawdown = dd

    hours_observed = len(valid)

    # Label logic
    if max_return >= 100:  # 2x+ at some point
        if final_return >= 20:
            label = "moon"  # pumped and held
        else:
            label = "pump_dump"  # pumped then crashed
    elif final_return <= -70:
        label = "rug"
    elif final_return >= 20:
        label = "up"
    elif final_return <= -30:
        label = "down"
    else:
        label = "crab"

    return {
        "entry_price": entry_price,
        "peak_price": peak_price,
        "final_price": final_price,
        "max_return_pct": round(max_return, 2),
        "final_return_pct": round(final_return, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "hours_observed": hours_observed,
        "label": label,
    }


def extract_features(search_result, info, security, candles):
    """
    Extract a flat feature dict from raw token data.
    These are the features available to the classifier.
    """
    features = {}

    # From search result
    features["address"] = search_result.get("mintAddress", "")
    features["name"] = search_result.get("name", "")
    features["symbol"] = search_result.get("symbol", "")
    features["creation_time"] = search_result.get("creationTime", 0)

    # From token info
    if info:
        features["price"] = float(info.get("price", 0) or 0)
        features["market_cap"] = float(info.get("marketCap", 0) or 0)
        features["liquidity"] = float(info.get("liquidity", 0) or 0)
        features["holder_count"] = int(info.get("holder", 0) or 0)
        features["volume_24h"] = float(info.get("volume24h", 0) or 0)
        features["trade_count_24h"] = int(info.get("trade24h", 0) or 0)
        features["buy_count_24h"] = int(info.get("buy24h", 0) or 0)
        features["sell_count_24h"] = int(info.get("sell24h", 0) or 0)
        features["unique_wallets_24h"] = int(info.get("uniqueWallet24h", 0) or 0)
        features["buy_volume_usd"] = float(info.get("vBuy24hUSD", 0) or 0)
        features["sell_volume_usd"] = float(info.get("vSell24hUSD", 0) or 0)

        # Derived ratios
        if features["sell_count_24h"] > 0:
            features["buy_sell_ratio"] = features["buy_count_24h"] / features["sell_count_24h"]
        else:
            features["buy_sell_ratio"] = float(features["buy_count_24h"]) if features["buy_count_24h"] > 0 else 0

        if features["market_cap"] > 0:
            features["liquidity_to_mcap"] = features["liquidity"] / features["market_cap"]
        else:
            features["liquidity_to_mcap"] = 0

        if features["holder_count"] > 0:
            features["volume_per_holder"] = features["volume_24h"] / features["holder_count"]
        else:
            features["volume_per_holder"] = 0

        if features["trade_count_24h"] > 0:
            features["wallets_per_trade"] = features["unique_wallets_24h"] / features["trade_count_24h"]
        else:
            features["wallets_per_trade"] = 0

        # Social signals
        ext = info.get("extensions", {}) or {}
        features["has_twitter"] = 1 if ext.get("twitter") else 0
        features["has_website"] = 1 if ext.get("website") else 0
        features["has_telegram"] = 1 if ext.get("telegram") else 0

    # From security
    if security:
        checks = security.get("checks", [])
        features["overall_risk"] = security.get("overallRisk", "unknown")
        features["security_check_count"] = len(checks)
        features["security_warnings"] = sum(1 for c in checks if c.get("result") == "warn")
        features["security_failures"] = sum(1 for c in checks if c.get("result") == "fail")

        # Extract specific security features
        for check in checks:
            name = check.get("name", "").lower().replace(" ", "_")
            features[f"sec_{name}"] = check.get("result", "unknown")
    else:
        features["overall_risk"] = "unknown"
        features["security_warnings"] = 0
        features["security_failures"] = 0

    # From OHLCV — early candle features (first 6 candles = first 6 hours)
    valid_candles = [c for c in candles if c.get("c", 0) > 0]
    if len(valid_candles) >= 2:
        early = valid_candles[:6]
        early_prices = [c["c"] for c in early]
        early_volumes = [c.get("v", 0) for c in early]

        features["early_price_change_pct"] = (early_prices[-1] / early_prices[0] - 1) * 100
        features["early_max_price"] = max(early_prices)
        features["early_min_price"] = min(early_prices)
        features["early_volatility"] = (max(early_prices) - min(early_prices)) / early_prices[0] * 100
        features["early_volume_total"] = sum(early_volumes)
        features["early_volume_trend"] = (
            early_volumes[-1] / early_volumes[0] if early_volumes[0] > 0 else 0
        )
        features["candle_count"] = len(valid_candles)
    else:
        features["early_price_change_pct"] = 0
        features["early_volatility"] = 0
        features["early_volume_total"] = 0
        features["candle_count"] = len(valid_candles)

    return features


def build_dataset(target_count=200):
    """Build the labeled token dataset."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Age ranges to sample from (minutes)
    age_ranges = [
        (360, 1440),     # 6h - 1d old
        (1440, 4320),    # 1-3 days old
        (4320, 10080),   # 3-7 days old
        (10080, 20160),  # 7-14 days old
    ]

    per_range = target_count // len(age_ranges) + 1

    # Collect token addresses
    all_tokens = []
    seen_addresses = set()

    # Load existing dataset to avoid re-collecting
    if DATASET_FILE.exists():
        with open(DATASET_FILE) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    seen_addresses.add(entry.get("features", {}).get("address", ""))
                except json.JSONDecodeError:
                    pass
        print(f"Existing dataset has {len(seen_addresses)} tokens")

    for min_age, max_age in age_ranges:
        print(f"\nSearching tokens aged {min_age // 60}h - {max_age // 60}h...")
        tokens = search_tokens(min_age, max_age, limit=per_range)
        new_tokens = [
            t for t in tokens
            if t.get("mintAddress", "") not in seen_addresses
        ]
        print(f"  Found {len(tokens)} tokens, {len(new_tokens)} new")
        all_tokens.extend(new_tokens)

    if not all_tokens:
        print("No new tokens to collect.")
        return

    print(f"\nCollecting features for {len(all_tokens)} tokens...")

    # Append to dataset
    with open(DATASET_FILE, "a") as f:
        for i, token in enumerate(all_tokens):
            address = token.get("mintAddress", "")
            if not address:
                continue

            print(f"  [{i+1}/{len(all_tokens)}] {token.get('symbol', '?'):>10s} ({address[:8]}...)")

            # Collect raw data
            info = get_token_info(address)
            security = get_token_security(address)
            candles = get_ohlcv(address, "1H")

            # Save raw data
            raw = {"search": token, "info": info, "security": security, "ohlcv": candles}
            with open(RAW_DIR / f"{address}.json", "w") as rf:
                json.dump(raw, rf)

            # Extract features
            features = extract_features(token, info, security, candles)

            # Compute label from OHLCV
            outcome = label_from_ohlcv(candles)

            entry = {
                "features": features,
                "outcome": outcome,
                "collected_at": int(time.time()),
            }
            f.write(json.dumps(entry) + "\n")

    # Print summary
    label_counts = {}
    total = 0
    with open(DATASET_FILE) as f:
        for line in f:
            try:
                entry = json.loads(line)
                label = (entry.get("outcome") or {}).get("label", "unknown")
                label_counts[label] = label_counts.get(label, 0) + 1
                total += 1
            except json.JSONDecodeError:
                pass

    print(f"\n--- Dataset Summary ---")
    print(f"Total tokens: {total}")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} ({count/total*100:.1f}%)")
    print(f"Saved to: {DATASET_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect token data from fomolt")
    parser.add_argument("--count", type=int, default=200, help="Target number of tokens")
    parser.add_argument("--refresh", action="store_true", help="Re-collect features for existing tokens")
    args = parser.parse_args()

    build_dataset(target_count=args.count)
