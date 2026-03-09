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


def score_token(features: dict) -> float:
    """
    Score a token from 0.0 (skip) to 1.0 (strong buy).

    Baseline strategy: simple heuristic filters.
    """
    score = 0.5  # neutral starting point

    # --- Liquidity filter ---
    liquidity = features.get("liquidity", 0)
    if liquidity < 5000:
        return 0.1  # too thin
    if liquidity > 50000:
        score += 0.05

    # --- Holder count ---
    holders = features.get("holder_count", 0)
    if holders < 20:
        return 0.1  # too few holders, likely insider
    if holders > 200:
        score += 0.05

    # --- Buy/sell ratio ---
    bs_ratio = features.get("buy_sell_ratio", 1.0)
    if bs_ratio > 1.2:
        score += 0.1  # more buying than selling
    elif bs_ratio < 0.8:
        score -= 0.1  # more selling

    # --- Security ---
    risk = features.get("overall_risk", "unknown")
    if risk == "high":
        return 0.1
    if risk == "low":
        score += 0.05

    failures = features.get("security_failures", 0)
    if failures > 0:
        return 0.1

    # --- Volume activity ---
    volume = features.get("volume_24h", 0)
    if volume > 100000:
        score += 0.05

    # --- Social presence ---
    if features.get("has_twitter", 0):
        score += 0.05
    if features.get("has_website", 0):
        score += 0.03

    # --- Early price action ---
    early_change = features.get("early_price_change_pct", 0)
    if early_change > 50:
        score += 0.05  # momentum
    elif early_change < -50:
        score -= 0.1  # dumping early

    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))
