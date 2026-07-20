"""
Backtest of a Merton-style dynamic asset allocation strategy vs. simple
buy-and-hold benchmarks (100% S&P 500, 100% cash/bonds, static 50/50).

Idea: each month, size the equity weight using the classic Merton
formula w* = (mu - rf) / (gamma * sigma^2), estimated from a trailing
12-month window of returns, then compare how that strategy performs
against some simple static benchmarks.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Pull data -------------------------------------------------------
# Just using SPY as a stand-in for "the market". auto_adjust=True gives
# us dividend/split-adjusted prices so returns already include dividends.
sp500 = yf.download("SPY", start="1993-01-01", end="2024-12-31", auto_adjust=True)["Close"].iloc[:, 0]
sp500_returns = sp500.pct_change().dropna()

# Assume a constant 5% annual risk-free rate for simplicity. In reality
# this would move around (T-bill yields, etc.) but a constant rate keeps
# the exercise focused on the equity-weighting logic rather than rate risk.
R_FREE = 0.05

REBAL_FREQ = "ME"      # rebalance monthly (month-end)
WINDOW_MONTHS = 12     # lookback window used to estimate mu/sigma

# Compound the daily returns up into monthly returns
monthly_sp500 = sp500_returns.resample(REBAL_FREQ).apply(lambda x: (1 + x).prod() - 1)

df = pd.DataFrame({"r_risky": monthly_sp500}).dropna()
df["r_f_current"] = R_FREE / 12  # monthly risk-free rate

# --- Estimate mu and sigma^2 from trailing 12 months ------------------
# .shift(1) is important here — we only want to use information that
# would have actually been available *before* that month's return, i.e.
# no look-ahead bias.
df["mu"] = df["r_risky"].rolling(WINDOW_MONTHS).mean().shift(1)
df["sigma2"] = df["r_risky"].rolling(WINDOW_MONTHS).var().shift(1)
df = df.dropna()

# Annualize the monthly estimates so they're on the same scale as R_FREE
df["mu_annual"] = df["r_risky"].rolling(WINDOW_MONTHS).mean().shift(1) * 12
df["sigma2_annual"] = df["r_risky"].rolling(WINDOW_MONTHS).var().shift(1) * 12

# --- Merton optimal weight ---------------------------------------------
# w* = (mu - rf) / (gamma * sigma^2)
# gamma = 3 is a fairly standard "moderate" risk-aversion assumption.
GAMMA = 3.0

df["w_merton"] = (df["mu_annual"] - R_FREE) / (GAMMA * df["sigma2_annual"])

# Clip to [0, 1] — no leverage, no shorting. Keeps things realistic for
# a retail-style implementation of the strategy.
df["w_merton_clipped"] = df["w_merton"].clip(0, 1)

# --- Build the strategy returns -----------------------------------------
df["port_return"] = df["w_merton_clipped"] * df["r_risky"] + (1 - df["w_merton_clipped"]) * df["r_f_current"]
df["bond_return"] = df["r_f_current"]
df["fifty_return"] = 0.5 * df["r_risky"] + 0.5 * df["r_f_current"]

# Turn returns into cumulative wealth curves (starting from $1)
df["wealth_merton"] = (1 + df["port_return"]).cumprod()
df["wealth_sp500"] = (1 + df["r_risky"]).cumprod()
df["wealth_bond"] = (1 + df["bond_return"]).cumprod()
df["wealth_fifty"] = (1 + df["fifty_return"]).cumprod()


def sharpe(returns, rf, periods=12):
    """Annualized Sharpe ratio from a series of periodic returns."""
    excess = returns - rf
    return excess.mean() / excess.std() * np.sqrt(periods)


def max_drawdown(wealth):
    """Worst peak-to-trough decline in a wealth series."""
    return ((wealth - wealth.cummax()) / wealth.cummax()).min()


# --- Print a quick summary table ----------------------------------------
print(f"{'Strategy':<12} {'Sharpe':>8} {'Max DD':>8} {'Ann Ret':>8}")
print("-" * 40)

strategies = [
    ("Merton", df["port_return"], df["wealth_merton"]),
    ("SP500", df["r_risky"], df["wealth_sp500"]),
    ("Bond", df["bond_return"], df["wealth_bond"]),
    ("50/50", df["fifty_return"], df["wealth_fifty"]),
]

for name, ret, wealth in strategies:
    ann_return = wealth.iloc[-1] ** (12 / len(df)) - 1
    print(f"{name:<12} {sharpe(ret, df['r_f_current']):>8.2f} {max_drawdown(wealth):>8.1%} {ann_return:>8.1%}")

# --- Plots ---------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(12, 14))

# 1) Wealth curves for each strategy
ax = axes[0]
ax.plot(df.index, df["wealth_sp500"], label="S&P 500", color="steelblue")
ax.plot(df.index, df["wealth_merton"], label="Merton", color="darkorange")
ax.plot(df.index, df["wealth_fifty"], label="50/50", color="green")
ax.plot(df.index, df["wealth_bond"], label="Bond", color="gray", linestyle="--")
ax.set_title("Cumulative Wealth ($1 invested in 1994)")
ax.set_ylabel("Portfolio Value ($)")
ax.legend()
ax.grid(alpha=0.3)

# 2) How the Merton weight moves over time (nice sanity check — should
# swing toward 0 in high-vol periods and toward 1 in calm bull markets)
ax = axes[1]
ax.plot(df.index, df["w_merton_clipped"], color="darkorange")
ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="100% equities")
ax.axhline(0.5, color="green", linestyle="--", alpha=0.5, label="50/50")
ax.set_title("Merton Optimal Weight in S&P 500 Over Time")
ax.set_ylabel("Weight")
ax.legend()
ax.grid(alpha=0.3)

# 3) Drawdown comparison — usually the most convincing chart for whether
# the dynamic strategy is actually managing risk or not
ax = axes[2]
for name, wealth, color in [
    ("S&P 500", df["wealth_sp500"], "steelblue"),
    ("Merton", df["wealth_merton"], "darkorange"),
    ("50/50", df["wealth_fifty"], "green"),
]:
    dd = (wealth - wealth.cummax()) / wealth.cummax()
    ax.plot(df.index, dd, label=name, color=color)
ax.set_title("Drawdowns")
ax.set_ylabel("Drawdown")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("backtest_results.png", dpi=150)
plt.show()