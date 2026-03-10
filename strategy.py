"""
Token scoring strategy.
THIS IS THE FILE YOU MODIFY.

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


_stats = {}
_sell_stats = {}


def calibrate(train_set):
    """Learn feature statistics per label from training data."""
    global _stats, _sell_stats
    by_label = {}
    for entry in train_set:
        label = (entry.get("outcome") or {}).get("label", "unknown")
        if label not in by_label:
            by_label[label] = {"features": [], "outcomes": []}
        by_label[label]["features"].append(entry["features"])
        by_label[label]["outcomes"].append(entry.get("outcome", {}))

    def median(vals):
        s = sorted(vals)
        n = len(s)
        return s[n // 2] if n else 0

    # Buy-side calibration: feature medians per label
    for label, data in by_label.items():
        feats_list = data["features"]
        _stats[label] = {}
        for key in ["liquidity", "holder_count", "buy_sell_ratio", "volume_24h",
                     "buy_volume_usd", "sell_volume_usd", "early_volatility",
                     "volume_per_holder", "early_price_change_pct"]:
            vals = [f.get(key, 0) for f in feats_list if f.get(key, 0) != 0]
            _stats[label][key] = median(vals) if vals else 0

    # Sell-side calibration: outcome stats per label
    for label, data in by_label.items():
        outcomes = data["outcomes"]
        max_returns = [o.get("max_return_pct", 0) for o in outcomes if o.get("max_return_pct", 0) > 0]
        final_returns = [o.get("final_return_pct", 0) for o in outcomes]
        drawdowns = [o.get("max_drawdown_pct", 0) for o in outcomes if o.get("max_drawdown_pct", 0) > 0]

        _sell_stats[label] = {
            "median_max_return": median(max_returns) if max_returns else 0,
            "median_final_return": median(final_returns) if final_returns else 0,
            "median_drawdown": median(drawdowns) if drawdowns else 0,
            "count": len(outcomes),
        }


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
    down_stats = _stats.get("down", {})

    if moon_stats and rug_stats:
        # Compare token to moon vs rug/down profiles
        moon_score = 0
        bad_score = 0

        pump_stats = _stats.get("pump_dump", {})

        for key, weight in [("liquidity", 1), ("holder_count", 1),
                            ("volume_per_holder", 1), ("buy_sell_ratio", 1.5),
                            ("early_price_change_pct", 2)]:
            val = features.get(key, 0)
            moon_val = moon_stats.get(key, 0)
            rug_val = rug_stats.get(key, 0)
            down_val = down_stats.get(key, 0) if down_stats else rug_val
            pump_val = pump_stats.get(key, 0) if pump_stats else rug_val

            if moon_val > 0 and val >= moon_val:
                moon_score += weight
            if rug_val > 0 and val <= rug_val:
                bad_score += weight
            if down_val > 0 and val <= down_val:
                bad_score += weight * 0.5
            if pump_val > 0 and val <= pump_val:
                bad_score += weight * 0.3

        # Net calibration signal
        net = moon_score - bad_score
        score += net * 0.03  # scale to reasonable range

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

    # --- Trade count activity ---
    trades = features.get("trade_count_24h", 0)
    if trades > 10000:
        score += 0.06  # very high trading activity
    elif trades > 5000:
        score += 0.04

    # --- Liquidity depth ---
    if liquidity > 50000:
        score += 0.05

    # --- Social ---
    if features.get("has_twitter", 0):
        score += 0.05
    if features.get("has_website", 0):
        score += 0.03

    # --- Liquidity-to-mcap ratio ---
    # Counter-intuitive: rugs have HIGH liq/mcap (crashed price), moons have LOW
    liq_mcap = features.get("liquidity_to_mcap", 0)
    if liq_mcap > 0:
        if liq_mcap > 0.35:
            score -= 0.08  # looks like crashed/rugged token
        elif liq_mcap < 0.15:
            score += 0.03  # healthy valuation relative to liquidity

    # --- Buy/sell volume USD imbalance (pump_dump detection) ---
    buy_vol = features.get("buy_volume_usd", 0)
    sell_vol = features.get("sell_volume_usd", 0)
    if sell_vol > buy_vol > 0:
        score -= 0.05  # net selling pressure in dollar terms

    # --- Security warnings ---
    warnings = features.get("security_warnings", 0)
    if warnings >= 2:
        score -= 0.05

    # --- Early price action (strongest signal) ---
    early_change = features.get("early_price_change_pct", 0)
    if early_change > 50:
        score += 0.10  # very strong momentum
    elif early_change > 20:
        score += 0.06
    elif early_change < -50:
        return 0.15  # hard reject — deep early dump
    elif early_change < -20:
        score -= 0.10

    # --- Early volatility (pump_dump has med=148, moon=75) ---
    early_vol = features.get("early_volatility", 0)
    if early_vol > 200:
        score -= 0.08  # extreme swings = pump_dump signal
    elif early_vol > 100:
        score -= 0.03

    return max(0.0, min(1.0, score))


def sell_signal(features: dict, position: dict) -> bool:
    """
    Decide whether to sell a position.

    Grid-search-optimized sell logic. Key findings from candle analysis:
    - Moons and pump_dumps are feature-identical (same risk, social, ratios)
    - Both can pump 100%+, but pump_dumps crash while moons recover
    - Critical insight: moons with peak > 150% NEVER drop below +10% return
    - Pump_dumps cross below +10% within hours of peak, then keep crashing

    Rule: sell if peak > 150%, 12h+ past peak, and return dropped below +10%.
    Catches 6/8 pump_dumps (+20.6% edge) with zero false moon sells.

    Returns True to sell, False to hold.
    """
    ret = position["unrealized_return_pct"]
    peak_ret = position["peak_return_pct"]
    candles_since_peak = position.get("candles_since_peak", 0)

    # Pump_dump catcher: token pumped 150%+ then sustained decline
    # peak > 150% means it's either a moon or pump_dump (not crab/up/down)
    # Key finding: moons with peak > 150% NEVER drop below +10% return
    # Pump_dumps cross below +10% within 12h of peak, then keep crashing
    # Selling at +10% (still positive) vs holding to avg -49% = huge edge
    if peak_ret > 150 and candles_since_peak > 12 and ret < 10:
        return True

    return False
