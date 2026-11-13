"""
Метрики эффективности торговой стратегии.

Обязательные:
- Total return, CAGR
- Sharpe ratio (annualized)
- Sortino ratio
- Max drawdown (abs & rel)
- Win rate, profit factor
- Average win / average loss
- Total trades, average holding period
- Calmar ratio
- Exposure time (% времени в позиции)

Опционально:
- Monthly returns
- Distribution of returns
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from src.strategy import Trade

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Структурированные метрики
# ---------------------------------------------------------------------------


@dataclass
class PerformanceMetrics:
    """Контейнер для всех метрик."""

    # --- Return ---
    total_return: float = 0.0         # Общий доходность (decimal)
    cagr: float = 0.0                 # Годовая доходность
    days_traded: int = 0              # Дней в данных

    # --- Risk ---
    sharpe_ratio: float = 0.0         # Sharpe (annualized, risk-free=0)
    sortino_ratio: float = 0.0        # Sortino
    calmar_ratio: float = 0.0         # CAGR / max_drawdown
    max_drawdown_abs: float = 0.0     # Абс. просадка
    max_drawdown_pct: float = 0.0     # Отн. просадка
    max_drawdown_duration: int = 0    # Дней просадки

    # --- Trades ---
    total_trades: int = 0
    win_trades: int = 0
    loss_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0        # gross_profit / gross_loss
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_return: float = 0.0
    avg_holding_bars: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0

    # --- Exposure ---
    exposure_pct: float = 0.0         # % времени в позиции

    # --- Other ---
    total_commissions: float = 0.0
    final_equity: float = 0.0
    initial_equity: float = 0.0


# ---------------------------------------------------------------------------
# Расчёты
# ---------------------------------------------------------------------------


def compute_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 365 * 24,  # hourly candles
) -> float:
    """
    Sharpe ratio (annualized).

    sharpe = (mean(r) - rf) / std(r) * sqrt(periods_per_year)
    """
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    std = excess.std()
    if std == 0:
        return 0.0
    return (excess.mean() / std) * np.sqrt(periods_per_year)


def compute_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 365 * 24,
) -> float:
    """
    Sortino ratio (annualized).
    Похож на Sharpe, но использует только downside deviation.
    """
    excess = returns - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    if len(downside) < 2:
        return 0.0
    downside_std = downside.std()
    if downside_std == 0:
        return 0.0
    return (excess.mean() / downside_std) * np.sqrt(periods_per_year)


def compute_max_drawdown(
    equity_curve: pd.Series,
) -> Dict:
    """
    Вычислить максимальную просадку.

    Возвращает
    ----------
    Dict: {peak, trough, drawdown, start, end, duration}
    """
    # equity никогда не должна идти ниже 0
    assert (equity_curve >= 0).all(), (
        f"Equity curve has negative values: min={equity_curve.min():.2f}. "
        "This indicates a bug in position sizing or risk management."
    )
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max

    if drawdown.min() >= 0:
        return {
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "peak_idx": 0,
            "trough_idx": 0,
            "duration": 0,
        }

    trough_idx = drawdown.idxmin()
    peak_idx = equity_curve[:trough_idx].idxmax()

    # Длительность просадки (от пика до момента восстановления)
    post_trough = equity_curve[trough_idx:]
    recovery_mask = post_trough >= equity_curve[peak_idx]
    if recovery_mask.any():
        recovery_idx = recovery_mask.idxmax()
        duration = int((recovery_idx - peak_idx).total_seconds() / 86400)
    else:
        recovery_idx = post_trough.index[-1]
        duration = int((recovery_idx - peak_idx).total_seconds() / 86400)

    dd = equity_curve[peak_idx] - equity_curve[trough_idx]
    dd_pct = (equity_curve[peak_idx] - equity_curve[trough_idx]) / equity_curve[peak_idx]

    return {
        "max_drawdown": dd,
        "max_drawdown_pct": dd_pct,
        "peak_idx": peak_idx,
        "trough_idx": trough_idx,
        "duration": duration,
    }


def compute_monthly_returns(
    equity_curve: pd.Series,
) -> pd.DataFrame:
    """
    Получить матрицу monthly returns для heatmap.

    Возвращает
    ----------
    pd.DataFrame — колонки по месяцам, строки по годам
    """
    if equity_curve.empty:
        return pd.DataFrame()

    # Daily returns
    daily_returns = equity_curve.pct_change().dropna()
    daily_returns.index.name = "date"
    daily_returns.name = "return"

    # Resample to monthly
    monthly = daily_returns.groupby(
        [daily_returns.index.year, daily_returns.index.month]
    ).apply(lambda x: (1 + x).prod() - 1)

    monthly = monthly.unstack(level=1)
    monthly.columns = [str(c) for c in monthly.columns]
    return monthly


def compute_metrics(
    trades: List[Trade],
    equity_curve: pd.DataFrame,
    initial_equity: float,
    timeframe: str = "1h",
) -> PerformanceMetrics:
    """
    Вычислить все метрики по списку сделок и equity curve.

    Параметры
    ---------
    trades : List[Trade]
    equity_curve : pd.DataFrame с колонкой 'equity'
    initial_equity : float
    timeframe : str — для расчёта periods_per_year

    Возвращает
    ----------
    PerformanceMetrics
    """
    m = PerformanceMetrics()
    m.initial_equity = initial_equity
    m.final_equity = float(equity_curve["equity"].iloc[-1]) if not equity_curve.empty else initial_equity

    # --- Return ---
    m.total_return = (m.final_equity - initial_equity) / initial_equity

    # Downtime
    if len(equity_curve) > 1:
        m.days_traded = (
            equity_curve.index[-1] - equity_curve.index[0]
        ).days
    else:
        m.days_traded = 1

    if m.days_traded < 30:
        # Слишком короткий период для годовой нормировки — берём total_return
        m.cagr = m.total_return
    else:
        annualization_factor = m.days_traded / 365.0
        if annualization_factor > 0:
            m.cagr = (
                (m.final_equity / initial_equity) ** (1 / annualization_factor) - 1
            )

    # --- Returns series ---
    if not equity_curve.empty:
        periods_per_hour = {"1h": 1, "4h": 0.25, "1d": 1/24}.get(timeframe.split("h")[0].replace("d", ""), 1)
        # Определяем periods per year
        tf_hours = 1
        if "d" in timeframe:
            tf_hours = int(timeframe.replace("d", "")) * 24
        elif "h" in timeframe:
            tf_hours = int(timeframe.replace("h", ""))
        elif "m" in timeframe:
            tf_hours = int(timeframe.replace("m", "")) / 60
        elif "w" in timeframe:
            tf_hours = 168

        periods_per_year = int(8760 / tf_hours)

        daily_eq = equity_curve["equity"]
        returns = daily_eq.pct_change().dropna()

        m.sharpe_ratio = compute_sharpe_ratio(returns, periods_per_year=periods_per_year)
        m.sortino_ratio = compute_sortino_ratio(returns, periods_per_year=periods_per_year)

        # --- Max Drawdown ---
        dd_info = compute_max_drawdown(daily_eq)
        m.max_drawdown_abs = dd_info["max_drawdown"]
        m.max_drawdown_pct = dd_info["max_drawdown_pct"]
        m.max_drawdown_duration = int(dd_info["duration"])

        # --- Calmar ---
        if m.max_drawdown_pct > 0:
            m.calmar_ratio = m.cagr / m.max_drawdown_pct

    # --- Trades ---
    closed = [t for t in trades if t.is_closed and t.exit_price is not None]
    unclosed = [t for t in trades if not t.is_closed]

    m.total_trades = len(closed)
    m.total_commissions = sum(t.commission for t in trades)

    if closed:
        pnls = [t.pnl for t in closed]
        pnl_pcts = [t.pnl_pct for t in closed if t.pnl_pct is not None]

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        m.win_trades = len(wins)
        m.loss_trades = len(losses)
        m.win_rate = len(wins) / len(closed) if closed else 0.0

        if wins:
            m.avg_win = np.mean(wins)
            m.best_trade = max(wins)

        if losses:
            m.avg_loss = np.mean(losses)
            m.worst_trade = min(losses)

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        m.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        m.avg_return = np.mean(pnl_pcts) if pnl_pcts else 0.0
        m.avg_holding_bars = np.mean([t.holding_bars for t in closed]) if closed else 0.0

    # --- Exposure ---
    if not equity_curve.empty and trades:
        active_index = set()
        for t in trades:
            if t.entry_time and t.exit_time:
                mask = (equity_curve.index >= t.entry_time) & (equity_curve.index <= t.exit_time)
                active_index.update(equity_curve.index[mask])

        m.exposure_pct = len(active_index) / len(equity_curve) * 100.0

    # --- Unclosed ---
    if unclosed:
        logger.info("%d unclosed positions at end of backtest", len(unclosed))

    return m


# ---------------------------------------------------------------------------
# Форматирование вывода
# ---------------------------------------------------------------------------


def format_metrics_report(metrics: PerformanceMetrics) -> str:
    """
    Сформировать текстовый отчёт с метриками.
    """
    import math

    pf_str = "inf" if math.isinf(metrics.profit_factor) else f"{metrics.profit_factor:.3f}"
    sharpe_str = "N/A" if pd.isna(metrics.sharpe_ratio) else f"{metrics.sharpe_ratio:.3f}"
    sortino_str = "N/A" if pd.isna(metrics.sortino_ratio) else f"{metrics.sortino_ratio:.3f}"

    lines = [
        "=" * 60,
        "  БЭКТЕСТ ОТЧЁТ — Range Trading with Trend Filter",
        "=" * 60,
        "",
        "  ДОХОДНОСТЬ",
        "-" * 40,
        f"  Начальный капитал:      ${metrics.initial_equity:>12,.2f}",
        f"  Конечный капитал:       ${metrics.final_equity:>12,.2f}",
        f"  Общая доходность:       {metrics.total_return:>11.2%}",
        f"  CAGR:                   {metrics.cagr:>11.2%}",
        f"  Дней в данных:          {metrics.days_traded:>12}",
        "",
        "  РИСК",
        "-" * 40,
        f"  Sharpe Ratio:           {sharpe_str:>11s}",
        f"  Sortino Ratio:          {sortino_str:>11s}",
        f"  Calmar Ratio:           {metrics.calmar_ratio:>11.3f}",
        f"  Max Drawdown (abs):     ${metrics.max_drawdown_abs:>11,.2f}",
        f"  Max Drawdown (pct):     {metrics.max_drawdown_pct:>11.2%}",
        f"  Max DD Duration:        {metrics.max_drawdown_duration:>12} days",
        "",
        "  ТОРГОВЛЯ",
        "-" * 40,
        f"  Всего сделок:           {metrics.total_trades:>12}",
        f"  Побед:                  {metrics.win_trades:>12}",
        f"  Поражений:              {metrics.loss_trades:>12}",
        f"  Win Rate:               {metrics.win_rate:>11.2%}",
        f"  Profit Factor:          {pf_str:>11s}",
        f"  Avg Win:                ${metrics.avg_win:>11,.2f}",
        f"  Avg Loss:               ${metrics.avg_loss:>11,.2f}",
        f"  Avg Holding (bars):     {metrics.avg_holding_bars:>12.1f}",
        f"  Best Trade:             ${metrics.best_trade:>11,.2f}",
        f"  Worst Trade:            ${metrics.worst_trade:>11,.2f}",
        "",
        "  ПРОЧЕЕ",
        "-" * 40,
        f"  Exposure Time:          {metrics.exposure_pct:>11.1f}%",
        f"  Комиссии (всего):       ${metrics.total_commissions:>11,.2f}",
        "",
        "=" * 60,
    ]
    return "\n".join(lines)
