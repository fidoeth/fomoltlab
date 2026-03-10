# tokenlab

Autonomous experiment loop for Solana token classification strategies.
Adapted from the autoresearch framework.

## Overview

The goal: build a classifier that predicts which low-liquidity Solana tokens will go up vs rug. The classifier is evaluated on historical data — tokens we already know the outcome for — and scored on how profitable its "buy" picks would have been.

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar8`). The branch `tokenlab/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b tokenlab/<tag>` from current master.
3. **Read the in-scope files**:
   - `program.md` — this file.
   - `collect.py` — data collection from fomolt. Do not modify.
   - `evaluate.py` — evaluation harness. Do not modify.
   - `strategy.py` — the file you modify. Token scoring logic.
4. **Verify data exists**: Check that `data/dataset.jsonl` exists and has entries. If not, run `python3 collect.py` to build the dataset (takes ~10 minutes).
5. **Initialize results.tsv**: Create with just the header row.
6. **Confirm and go.**

## Files

| File | Role |
|---|---|
| `collect.py` | FIXED — collects token data from fomolt, builds labeled dataset |
| `evaluate.py` | Evaluation harness — loads dataset, runs strategy, simulates trades |
| `strategy.py` | MODIFIABLE — buy scoring + sell signal logic |
| `data/dataset.jsonl` | Cached dataset (gitignored) |
| `data/raw/*.json` | Raw per-token data with OHLCV candles (gitignored) |
| `results.tsv` | Experiment log |

## Experimentation

Each experiment is instant — just running the classifier on cached data. No training time, no GPU needed. Launch it as: `python3 evaluate.py`

**What you CAN do:**
- Modify `strategy.py` — the main file you edit. Contains:
  - `score_token(features)` — buy classifier (score > 0.5 = buy signal)
  - `sell_signal(features, position)` — sell logic (True = exit trade)
  - `calibrate(train_set)` — optional, receives training split for statistics
- Everything is fair game: thresholds, feature combinations, weighting schemes, ensemble logic, derived features.

**What you CANNOT do:**
- Modify `collect.py`.
- Look at test set labels when building the strategy (no cheating — the split is deterministic).

## Sell strategy

The evaluator supports full trade lifecycle simulation. If `strategy.sell_signal()` exists, after a buy signal the evaluator steps through OHLCV candles one-by-one, calling `sell_signal(features, position)` at each candle.

The `position` dict provides:
- `entry_price`, `current_price`, `high`, `low` — prices
- `unrealized_return_pct` — current % return
- `peak_return_pct` — max % return seen so far
- `drawdown_from_peak_pct` — % drop from peak
- `candles_held` — hours since entry (1H candles)
- `candles_since_peak` — hours since the highest price was seen
- `volume`, `volume_usd` — current candle volume
- `total_candles` — total candles in observation

If `sell_signal` returns True, the trade exits at the current candle's close price. If no sell triggers, the trade holds to end of observation (same as before).

Sell-specific metrics are printed:
- `sell_vs_hold` — average improvement of sell strategy over buy-and-hold
- `capture_ratio` — how much of peak return was captured
- Per-label sell effectiveness breakdown

**The goal: maximize `avg_return`** — the average return on tokens your classifier flags as "buy". Secondary goals: high `alpha` (excess return vs buying everything), high `precision` (avoiding false positives), reasonable `recall` (not missing all the good ones).

## Output format

The evaluator prints:

```
---
avg_return:       12.34
alpha:            8.56
win_rate:         62.5
sharpe:           0.456
num_buys:         15
num_skips:        45
total_tokens:     60
precision:        0.750
recall:           0.600
f1:               0.667
simulated_pnl:    0.1851
best_trade:       150.00
worst_trade:      -85.00
median_return:    5.20
market_avg:       3.78
```

Extract the key metric: `grep "^avg_return:" run.log`

## Logging results

Log to `results.tsv` (tab-separated):

```
commit	avg_return	alpha	win_rate	status	description
```

1. git commit hash (short, 7 chars)
2. avg_return achieved — use 0.00 for crashes
3. alpha (excess return vs market) — use 0.00 for crashes
4. win_rate — use 0.0 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	avg_return	alpha	win_rate	status	description
a1b2c3d	3.78	0.00	45.2	keep	baseline
b2c3d4e	8.56	4.78	62.5	keep	add liquidity ratio filter
c3d4e5f	2.10	-1.68	40.0	discard	aggressive security filter
d4e5f6g	0.00	0.00	0.0	crash	syntax error in score_token
```

## The experiment loop

LOOP FOREVER:

1. Look at the git state: current branch/commit
2. Modify `strategy.py` with an experimental idea
3. git commit
4. Run: `python3 evaluate.py > run.log 2>&1`
5. Read results: `grep "^avg_return:\|^alpha:\|^win_rate:" run.log`
6. If grep is empty, it crashed. Run `tail -n 20 run.log` and fix.
7. Record results in the tsv
8. If avg_return improved, keep the commit (advance the branch)
9. If avg_return is equal or worse, git reset back

## Strategy ideas to try

### Buy signal
- **Feature engineering**: compute new ratios or combine existing features
- **Threshold tuning**: adjust the thresholds for liquidity, holders, volume, etc.
- **Multi-signal scoring**: weight multiple signals and sum them
- **Negative filters**: hard filters that instantly reject tokens (security failures, too few holders, etc.)
- **Label-aware calibration**: use `calibrate()` to learn thresholds from training data
- **Selectivity**: being more selective (fewer buys, higher scores) often beats buying everything
- **Pattern recognition**: look for specific feature combinations that predict moons vs rugs
- **Volume patterns**: buy_volume vs sell_volume ratios, volume per holder, etc.
- **Time decay**: weight features differently based on token age

### Sell signal
- **Trailing stops**: sell when drawdown from peak exceeds threshold (careful: moons can dip 70%+ from peak before recovering)
- **Pump_dump detection**: sell tokens that pumped 150%+ and show sustained decline (candles_since_peak > 12, ret < 10%)
- **Feature-informed exits**: use token features to set different sell thresholds per trade
- **Volume collapse**: sell when current candle volume drops to near zero (dead market)
- **Time-based exits**: exit stagnant positions after extended holding period

**Important findings**: Moons and pump_dumps are feature-identical (same risk, social, ratios). Both can crash 70%+ from peak mid-trade. The only reliable sell signals require high specificity to avoid false moon sells. Grid search over parameters is recommended.

## NEVER STOP

Once the loop begins, do NOT pause to ask. The human might be away. You are autonomous. If you run out of ideas, re-read the feature list, try combining near-misses, try radical changes, try simplification. The loop runs until interrupted.
