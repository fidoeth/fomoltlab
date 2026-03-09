"""
Token scoring strategy.
THIS IS THE FILE YOU MODIFY.

The score_token function receives a feature dict and returns a score 0.0-1.0.
Tokens scoring > 0.5 get a "buy" signal.

Available features (all numeric unless noted):
    address, name, symbol                    — identifiers (string)
    creation_time                            — unix timestamp
    price, market_cap, liquidity             — current values
    holder_count                             — number of holders
    volume_24h, trade_count_24h              — 24h activity
    buy_count_24h, sell_count_24h            — buy/sell counts
    unique_wallets_24h                       — unique traders
    buy_volume_usd, sell_volume_usd          — buy/sell volume in USD
    buy_sell_ratio                           — buy_count / sell_count
    liquidity_to_mcap                        — liquidity / market_cap
    volume_per_holder                        — volume_24h / holder_count
    wallets_per_trade                        — unique_wallets / trade_count
    has_twitter, has_website, has_telegram   — 1 or 0
    overall_risk                             — "low", "medium", "high", "unknown" (string)
    security_warnings, security_failures     — counts
    early_price_change_pct                   — % change in first ~6 hours
    early_volatility                         — price range / entry in first ~6 hours (%)
    early_volume_total                       — total volume in first ~6 hours
    candle_count                             — number of valid OHLCV candles
"""


_stats = {}


def calibrate(train_set):
    """Learn feature averages per label from training data."""
    global _stats
    by_label = {}
    for entry in train_set:
        label = (entry.get("outcome") or {}).get("label", "unknown")
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(entry["features"])

    # Compute median values for key features per label
    def median(vals):
        s = sorted(vals)
        n = len(s)
        return s[n // 2] if n else 0

    for label, feats_list in by_label.items():
        _stats[label] = {}
        for key in ["liquidity", "holder_count", "buy_sell_ratio", "volume_24h",
                     "buy_volume_usd", "sell_volume_usd", "early_volatility",
                     "volume_per_holder", "early_price_change_pct"]:
            vals = [f.get(key, 0) for f in feats_list if f.get(key, 0) != 0]
            _stats[label][key] = median(vals) if vals else 0


def score_token(features: dict) -> float:
    """
    Score a token from 0.0 (skip) to 1.0 (strong buy).

    v4: data-driven thresholds from calibration + baseline filters.
    """
    score = 0.5

    # --- Hard rejects ---
    liquidity = features.get("liquidity", 0)
    if liquidity < 5000:
        return 0.1
    holders = features.get("holder_count", 0)
    if holders < 20:
        return 0.1
    if features.get("security_failures", 0) > 0:
        return 0.1
    if features.get("overall_risk", "unknown") == "high":
        return 0.1

    # --- Data-driven signals (if calibrated) ---
    moon_stats = _stats.get("moon", {})
    rug_stats = _stats.get("rug", {})

    if moon_stats and rug_stats:
        # Tokens that look more like moons than rugs on key features
        moon_liq = moon_stats.get("liquidity", 20000)
        rug_liq = rug_stats.get("liquidity", 10000)
        if liquidity >= moon_liq:
            score += 0.05
        elif liquidity <= rug_liq:
            score -= 0.05

        moon_holders = moon_stats.get("holder_count", 200)
        rug_holders = rug_stats.get("holder_count", 50)
        if holders >= moon_holders:
            score += 0.05
        elif holders <= rug_holders:
            score -= 0.05

        moon_vph = moon_stats.get("volume_per_holder", 500)
        rug_vph = rug_stats.get("volume_per_holder", 200)
        vph = features.get("volume_per_holder", 0)
        if vph >= moon_vph:
            score += 0.05
        elif vph <= rug_vph:
            score -= 0.03

    # --- Buy/sell ratio ---
    bs_ratio = features.get("buy_sell_ratio", 1.0)
    if bs_ratio > 1.2:
        score += 0.1
    elif bs_ratio < 0.8:
        score -= 0.1

    # --- Security ---
    if features.get("overall_risk", "unknown") == "low":
        score += 0.05

    # --- Volume ---
    volume = features.get("volume_24h", 0)
    if volume > 100000:
        score += 0.05

    # --- Liquidity depth ---
    if liquidity > 50000:
        score += 0.05

    # --- Social ---
    if features.get("has_twitter", 0):
        score += 0.05
    if features.get("has_website", 0):
        score += 0.03

    # --- Early price action ---
    early_change = features.get("early_price_change_pct", 0)
    if early_change > 50:
        score += 0.05
    elif early_change < -50:
        score -= 0.1

    return max(0.0, min(1.0, score))
