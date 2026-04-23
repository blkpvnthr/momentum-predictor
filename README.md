# Momentum Predictor

Intraday Momentum / Breakout Predictor.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Status](https://img.shields.io/badge/Status-Active-success)
![Trading](https://img.shields.io/badge/Mode-Paper%20Trading-orange)
![License](https://img.shields.io/badge/License-Private-lightgrey)

A **production-grade, regime-aware trading system** combining:

- Reinforcement Learning (RL)
- Alpha-first portfolio optimization
- Covariance-aware risk control
- Real-time execution via Alpaca

---

## 📌 Overview

This system is designed to **maximize return while maintaining strict capital discipline**.

It integrates:
- 📊 ML-based signal generation  
- 🧠 Portfolio optimization (alpha + covariance)  
- ⚙️ Execution engine (risk + capital constraints)  

---

## 🧠 Architecture


Market Data → Feature Engineering → RL Policy → Target Weights
↓
Alpha Portfolio Optimizer (Sharpe + Covariance + Momentum)
↓
Execution Engine (Volatility + Regime + Capital Constraints)
↓
Alpaca Broker (Paper Trading)


---

## ⚙️ Core Components

### 1. Reinforcement Learning Model
- Outputs **continuous actions per asset**
- Converted into **target portfolio weights**
- Regime-aware:
  - BULL
  - TRANSITION
  - BEAR

---

### 2. Alpha Portfolio Optimizer

**Goal:** Maximize return while controlling risk

**Objective Function:**

maximize: alpha - risk_penalty - turnover_penalty


**Key Features:**
- Alpha-first allocation (model-driven)
- Momentum blending:
  - short-term
  - medium-term
- Covariance used as **risk regularizer**
- Dynamic:
  - gross exposure
  - position caps
- Top-K concentration
- Gradient-based optimization

---

### 3. Execution Engine

The execution layer ensures:

- ✅ Cash never goes negative  
- ✅ Capital deployed efficiently  
- ✅ Orders executed safely  

**Core Logic:**
- Sell first → free capital
- Buy highest conviction positions
- Enforce:
  - cooldowns
  - minimum trade sizes
  - no duplicate orders

---

## 📊 Strategy Behavior

### Regime-Based Allocation

| Regime      | Behavior |
|------------|---------|
| BULL       | Aggressive, high exposure |
| TRANSITION | Controlled, cautious |
| BEAR       | Defensive, reduced risk |

---

### Volatility Scaling

- High volatility → reduce exposure  
- Strong bull + high confidence → override suppression  

---

## 💰 Risk & Capital Controls

- Max position caps enforced  
- Minimum buying power maintained  
- Turnover penalties applied  
- Full liquidation for weak signals  
- No negative cash states  

---

## 📡 Data Pipeline

- Source: Alpaca (`DataFeed.IEX`)
- Frequency: 1-minute bars
- Warmup: 500 bars
- Universe includes:
  - Core ETFs: `QQQ`, `TQQQ`, `SQQQ`
  - Extended high-beta + sector equities

---

## 🧾 Logging & Monitoring

Outputs:

### Trade Journal

live/logs/paper_trader_journal.csv


### Equity Curve

live/logs/paper_trader_equity_curve.csv


### Runtime Logs Include:
- Regime + confidence
- Portfolio value
- Volatility
- Optimizer Sharpe
- Trade decisions

---

## ▶️ Getting Started

### 1. Install Dependencies
```bash
pip install -r file.txt
```
2. Configure Environment

Create .env:
```bash
APCA_API_KEY_ID=your_key
APCA_API_SECRET_KEY=your_secret
```
3. Run Paper Trader
```bash
python trade.py
```
🔁 Execution Loop

Every ~30 seconds:

Fetch market data
Build features
Run RL model
Generate target weights
Optimize portfolio
Execute trades
Log results
🧪 Design Philosophy
Alpha > Everything

Model signal drives decisions. Risk only constrains.

Execution is the Edge

Most systems fail here — this one prioritizes:

capital efficiency
minimal churn
controlled risk
Regime Awareness

Same signal behaves differently across market conditions.

⚠️ Limitations
Alpaca symbol limits (~30 streams)
Market hours only
Requires sufficient historical data
Dependent on feature pipeline quality
🔮 Future Improvements
Multi-agent ensemble strategies
Live trading deployment
Adaptive universe selection
Cross-asset support (crypto, futures)
Online learning
📬 Summary

This is a full-stack trading system, not just a model.

It combines:

Machine learning
Portfolio theory
Real-world execution constraints

Goal:

Maximize return while maintaining strict control over risk and capital deployment.

⚠️ Disclaimer

This software is for research and educational purposes only.
Not financial advice. Use at your own risk.