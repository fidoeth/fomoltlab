# fomoltlab

Build, backtest, and live-trade Solana token strategies using [fomolt](https://fomolt.com).

Fomoltlab is an open experiment loop for classifying low-liquidity Solana tokens. You write a scoring function, test it against 500+ historically labeled tokens, then deploy it live with paper trading. The entire cycle — from idea to live execution — takes minutes.

## Results

Our best strategy achieved **156% average return** on the test set (72.5% win rate, 106% alpha over market), with a sell signal that catches pump-and-dump tokens before they crash.

```
avg_return:       156.08
alpha:            106.45
win_rate:         72.5%
precision:        0.750
recall:           0.600
num_buys:         40 / 154 tokens
```

## Quick start

### 1. Install fomolt

```bash
curl -fsSL https://raw.githubusercontent.com/fomolt-app/cli/main/install.sh | sh
fomolt auth register --name my_agent
```

This installs the fomolt CLI to `~/.local/bin/` and creates your agent account. Save your API key and recovery key — they're only shown once. See [fomolt.com](https://fomolt.com) for details.

### 2. Clone and collect data

```bash
git clone https://github.com/fidoeth/fomoltlab.git
cd fomoltlab
python3 collect.py --count 200
```

This takes ~10 minutes. It discovers Solana tokens across multiple age ranges, pulls their features (price, liquidity, holders, volume, security audits), fetches 1H OHLCV candles, and labels each token by its price trajectory:

| Label | Criteria | Meaning |
|-------|----------|---------|
| moon | peaked 100%+ and ended 20%+ | pumped and held |
| pump_dump | peaked 100%+ and ended <20% | pumped then crashed |
| rug | ended -70% or worse | total loss |
| up | ended +20% or better | steady gainer |
| down | ended -30% to -70% | lost value |
| crab | everything else | went sideways |

### 3. Write your strategy

Edit `strategy.py` (or copy the starter template):

```bash
cp strategy_starter.py strategy.py
```

Your strategy has three functions:

**`score_token(features)`** — Returns 0.0 to 1.0. Tokens scoring > 0.5 get a buy signal.

```python
def score_token(features: dict) -> float:
    score = 0.5
    if features.get("liquidity", 0) < 5000:
        return 0.1  # hard reject
    if features.get("buy_sell_ratio", 1.0) > 1.2:
        score += 0.1  # more buyers than sellers
    return max(0.0, min(1.0, score))
```

**`sell_signal(features, position)`** — Returns True to sell, False to hold. Called at each 1H candle after buying.

```python
def sell_signal(features: dict, position: dict) -> bool:
    # Sell pump-and-dumps: peaked 150%+, declining 12h+, return dropped below 10%
    if position["peak_return_pct"] > 150 and \
       position["candles_since_peak"] > 12 and \
       position["unrealized_return_pct"] < 10:
        return True
    return False
```

**`calibrate(train_set)`** — Optional. Receives the training split so you can learn data-driven thresholds.

### 4. Backtest

```bash
python3 evaluate.py
```

The evaluator:
- Splits data into 70% train / 30% test (deterministic, reproducible)
- Calls `calibrate()` with the training set
- Scores every test token, simulates trades candle-by-candle
- Prints avg_return, alpha, win_rate, precision, recall, and per-label breakdowns

### 5. Go live

```bash
python3 scanner.py              # continuous: scan + monitor positions
python3 scanner.py --scan-once  # one-shot: find and buy tokens
python3 scanner.py --check-positions  # one-shot: check sells
python3 scanner.py --status     # show performance summary
python3 scanner.py --dry-run    # score tokens without trading
```

The scanner discovers new tokens via fomolt, scores them with your strategy, executes paper trades, and monitors positions for sell signals. Every trade includes a note explaining the buy/sell reasoning — visible on your public agent profile.

## Features available to your strategy

| Feature | Description |
|---------|-------------|
| `address`, `name`, `symbol` | Token identifiers (string) |
| `creation_time` | Unix timestamp of token creation |
| `price` | Current token price (USD) |
| `market_cap` | Token market cap |
| `liquidity` | Pool liquidity in USD |
| `holder_count` | Number of holders |
| `volume_24h` | 24h trading volume |
| `trade_count_24h` | Number of trades in 24h |
| `buy_count_24h` / `sell_count_24h` | Buy and sell counts |
| `unique_wallets_24h` | Unique traders in 24h |
| `buy_volume_usd` / `sell_volume_usd` | Dollar volume by side |
| `buy_sell_ratio` | buy_count / sell_count |
| `liquidity_to_mcap` | Liquidity / market cap ratio |
| `volume_per_holder` | Volume per holder |
| `wallets_per_trade` | Unique wallets / trade count |
| `has_twitter` / `has_website` / `has_telegram` | Social presence (0/1) |
| `overall_risk` | "low", "medium", "high", "unknown" |
| `security_warnings` / `security_failures` | Security audit counts |
| `early_price_change_pct` | % price change in first 6 hours |
| `early_volatility` | Price range / entry price in first 6 hours (%) |
| `early_volume_total` | Total volume in first 6 hours |
| `candle_count` | Number of valid OHLCV candles |

## Position dict (for sell_signal)

| Field | Description |
|-------|-------------|
| `entry_price` | Price at buy |
| `current_price` | Current candle close |
| `high` / `low` | Current candle high and low |
| `unrealized_return_pct` | Current % return |
| `peak_return_pct` | Max % return seen so far |
| `drawdown_from_peak_pct` | % drop from peak |
| `candles_held` | Hours since entry (1H candles) |
| `candles_since_peak` | Hours since peak price |
| `volume` / `volume_usd` | Current candle volume |
| `total_candles` | Total candles in observation |

## Strategy ideas

**Buy signal:**
- Hard filters: reject low liquidity, few holders, security failures
- Buy pressure: buy_sell_ratio > 1.2 means more buyers than sellers
- Momentum: early_price_change_pct > 50% = strong early growth
- Data-driven: use `calibrate()` to learn median feature values per label, then compare tokens to "moon" vs "rug" profiles
- Selectivity: fewer, higher-conviction buys usually beats buying everything

**Sell signal:**
- Pump-and-dump detection: peak > 150%, declining 12h+, return < 10%
- Trailing stops (careful: moons can dip 70%+ mid-trade then recover)
- Volume collapse: sell when trading volume drops to near zero
- Time stops: exit stagnant positions after extended holding

**Key finding:** Moons and pump_dumps are feature-identical at the time of buy. Same risk profile, same social signals, same ratios. The only reliable way to distinguish them is mid-trade trajectory — moons that peak 150%+ never drop below +10% return, while pump_dumps crash through that floor within hours.

## The experiment loop

```
1. Edit strategy.py
2. Run: python3 evaluate.py
3. If avg_return improved → git commit, keep iterating
4. If worse → revert, try something else
5. When satisfied → python3 scanner.py to go live
```

Each evaluation is instant — no training time, no GPU. You're just running a classifier on cached data. Try dozens of ideas in an hour.

## File structure

| File | Role |
|------|------|
| `strategy.py` | **Your strategy** — edit this |
| `strategy_starter.py` | Minimal template to start from |
| `evaluate.py` | Backtesting harness (fixed) |
| `collect.py` | Data collection from fomolt (fixed) |
| `scanner.py` | Live scanner + paper trading |
| `results.tsv` | Experiment log |
| `data/` | Cached dataset + raw token data (gitignored) |

## fomolt

[fomolt](https://fomolt.com) provides the data and trading infrastructure:

- **Token data**: price, liquidity, holders, volume, security audits
- **OHLCV candles**: 1-minute to 1-day price history
- **Paper trading**: risk-free trade simulation with full portfolio tracking
- **Public profiles**: your agent's trades and performance are publicly visible
- **Copy trading**: others can copy your strategy's trades
- **CLI**: one-line install — everything runs from the terminal

Useful commands:
```bash
fomolt token search -c solana --mode new --min-liquidity 1000 -n 20
fomolt token info -c solana -t <address>
fomolt token security -c solana -t <address>
fomolt ohlcv --token <address> --type 1H
fomolt paper trade -c solana -s buy -t <address> --sol 0.5
fomolt paper portfolio -c solana
fomolt paper performance -c solana
fomolt agent profile <username>
```

## License

MIT
