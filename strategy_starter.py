"""
Starter strategy — a simple template to build from.

Copy this to strategy.py and start experimenting:
    cp strategy_starter.py strategy.py

The score_token function receives a feature dict and returns a score 0.0-1.0.
Tokens scoring > 0.5 get a "buy" signal.

The sell_signal function receives features + position state and returns
True (sell) or False (hold) for full trade lifecycle simulation.

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

Position dict (for sell_signal):
    entry_price                              — price at buy
    current_price                            — current candle close
    high, low                                — current candle high/low
    unrealized_return_pct                    — current % return
    peak_return_pct                          — max % return seen so far
    drawdown_from_peak_pct                   — % drop from peak
    candles_held                             — hours since entry (1H candles)
    candles_since_peak                       — hours since highest price was seen
    volume, volume_usd                       — current candle volume
    total_candles                            — total candles in observation
"""


def calibrate(train_set):
    """
    Optional: learn from training data before evaluation.

    train_set is a list of dicts, each with:
        - "features": dict of token features
        - "outcome": dict with "label", "max_return_pct", "final_return_pct", etc.

    Use this to compute thresholds, averages, or any statistics
    that inform your scoring logic.
    """
    pass


def score_token(features: dict) -> float:
    """
    Score a token from 0.0 (skip) to 1.0 (strong buy).
    Tokens scoring > 0.5 get a buy signal.

    Start simple, then add complexity. Ideas:
    - Filter out obvious rugs (low liquidity, high risk)
    - Reward positive signals (buy pressure, social presence)
    - Use calibrate() to learn data-driven thresholds
    """
    score = 0.5

    # --- Hard rejects ---
    if features.get("liquidity", 0) < 5000:
        return 0.1
    if features.get("holder_count", 0) < 20:
        return 0.1
    if features.get("security_failures", 0) > 0:
        return 0.1

    # --- Buy pressure ---
    bs_ratio = features.get("buy_sell_ratio", 1.0)
    if bs_ratio > 1.2:
        score += 0.1
    elif bs_ratio < 0.8:
        score -= 0.1

    # --- Social presence ---
    if features.get("has_twitter", 0):
        score += 0.05
    if features.get("has_website", 0):
        score += 0.03

    # --- Security ---
    if features.get("overall_risk", "unknown") == "low":
        score += 0.05
    elif features.get("overall_risk", "unknown") == "high":
        score -= 0.15

    return max(0.0, min(1.0, score))


def sell_signal(features: dict, position: dict) -> bool:
    """
    Decide whether to sell a position.
    Returns True to sell, False to hold.

    Start with no sells (return False) and add rules as you learn
    from the candle data. Ideas:
    - Trailing stop: sell when drawdown exceeds threshold
    - Take profit: sell when return exceeds target
    - Time stop: sell after holding too long with no gains
    """
    return False
