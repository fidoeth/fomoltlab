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
| `evaluate.py` | FIXED — loads dataset, runs strategy, computes metrics |
| `strategy.py` | MODIFIABLE — the scoring function you iterate on |
| `data/dataset.jsonl` | Cached dataset (gitignored) |
| `results.tsv` | Experiment log |

## Experimentation

Each experiment is instant — just running the classifier on cached data. No training time, no GPU needed. Launch it as: `python3 evaluate.py`

**What you CAN do:**
- Modify `strategy.py` — this is the only file you edit. The `score_token(features)` function is the classifier. Everything is fair game: thresholds, feature combinations, weighting schemes, ensemble logic, derived features.
- Optionally implement a `calibrate(train_set)` function in strategy.py — it receives the training split and can compute statistics to inform scoring.

**What you CANNOT do:**
- Modify `collect.py` or `evaluate.py`.
- Change the evaluation metric computation.
- Look at test set labels when building the strategy (no cheating — the split is deterministic).

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

- **Feature engineering**: compute new ratios or combine existing features
- **Threshold tuning**: adjust the thresholds for liquidity, holders, volume, etc.
- **Multi-signal scoring**: weight multiple signals and sum them
- **Negative filters**: hard filters that instantly reject tokens (security failures, too few holders, etc.)
- **Label-aware calibration**: use `calibrate()` to learn thresholds from training data
- **Selectivity**: being more selective (fewer buys, higher scores) often beats buying everything
- **Pattern recognition**: look for specific feature combinations that predict moons vs rugs
- **Volume patterns**: buy_volume vs sell_volume ratios, volume per holder, etc.
- **Time decay**: weight features differently based on token age

## NEVER STOP

Once the loop begins, do NOT pause to ask. The human might be away. You are autonomous. If you run out of ideas, re-read the feature list, try combining near-misses, try radical changes, try simplification. The loop runs until interrupted.
