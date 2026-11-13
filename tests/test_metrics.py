"""
Тесты для metrics.py.

Используют синтетические equity_curve и trade данные.
"""

from __future__ import annotations

import sys
import pathlib
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.metrics import (
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_max_drawdown,
    compute_monthly_returns,
    compute_metrics,
    format_metrics_report,
    PerformanceMetrics,
)
from src.strategy import Trade


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------


def _make_equity_curve(n: int = 365 * 24, start: float = 100000.0,
                       drift: float = 0.0001, volatility: float = 0.005) -> pd.DataFrame:
    """Синтетическая equity curve (daily-like)."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    eq = start
    values = [eq]
    for _ in range(n - 1):
        ret = drift + volatility * np.random.normal()
        eq *= (1 + ret)
        values.append(eq)

    return pd.DataFrame({"equity": values}, index=dates)


def _make_sample_trades(n_wins: int = 10, n_losses: int = 5) -> list:
    """Создать список сделок с известными характеристиками."""
    trades = []
    base_time = pd.Timestamp("2024-01-01", tz="UTC")

    for i in range(n_wins):
        trades.append(Trade(
            symbol="ETH/USDT",
            entry_time=base_time + pd.Timedelta(hours=i * 20),
            entry_price=100.0,
            entry_size=10.0,
            stop_price=95.0,
            take_profit_price=None,
            exit_time=base_time + pd.Timedelta(hours=i * 20 + 5),
            exit_price=110.0,
            exit_reason="take_profit",
            pnl=100.0,  # profit
            pnl_pct=10.0,
            holding_bars=5,
            commission=2.0,
            is_closed=True,
            equity_at_entry=100000.0,
        ))

    for i in range(n_losses):
        trades.append(Trade(
            symbol="BTC/USDT",
            entry_time=base_time + pd.Timedelta(hours=(n_wins + i) * 20),
            entry_price=100.0,
            entry_size=10.0,
            stop_price=95.0,
            take_profit_price=None,
            exit_time=base_time + pd.Timedelta(hours=(n_wins + i) * 20 + 3),
            exit_price=93.0,
            exit_reason="stop_loss",
            pnl=-70.0,  # loss
            pnl_pct=-7.0,
            holding_bars=3,
            commission=2.0,
            is_closed=True,
            equity_at_entry=100000.0,
        ))

    # Добавить одну unclosed
    trades.append(Trade(
        symbol="SOL/USDT",
        entry_time=base_time + pd.Timedelta(hours=1000),
        entry_price=100.0,
        entry_size=5.0,
        stop_price=95.0,
        take_profit_price=None,
        is_closed=False,
        equity_at_entry=100000.0,
    ))

    return trades


# ---------------------------------------------------------------------------
# Sharpe & Sortino
# ---------------------------------------------------------------------------


class TestSharpeSortino:
    def test_sharpe_positive_drift(self):
        """Positive drift → positive sharpe."""
        returns = pd.Series(np.random.normal(0.001, 0.01, 1000))
        sharpe = compute_sharpe_ratio(returns, periods_per_year=252)
        assert sharpe > 0

    def test_sharpe_negative_drift(self):
        """Negative drift → negative sharpe."""
        returns = pd.Series(np.random.normal(-0.001, 0.01, 1000))
        sharpe = compute_sharpe_ratio(returns, periods_per_year=252)
        assert sharpe < 0

    def test_sharpe_zero_vol(self):
        """Нулевая волатильность → sharpe = 0."""
        returns = pd.Series([0.0] * 100)
        sharpe = compute_sharpe_ratio(returns, periods_per_year=252)
        assert sharpe == 0.0

    def test_sortino_ignores_positive_vol(self):
        """Sortino должен игнорировать положительную волатильность."""
        returns = pd.Series(np.abs(np.random.normal(0.001, 0.01, 1000)))
        sortino = compute_sortino_ratio(returns, periods_per_year=252)
        assert sortino >= 0

    def test_sortino_all_positive(self):
        """Все положительные → inf или очень большое значение."""
        returns = pd.Series([0.01] * 100)
        sortino = compute_sortino_ratio(returns, periods_per_year=252)
        assert sortino >= 0


# ---------------------------------------------------------------------------
# Max Drawdown
# ---------------------------------------------------------------------------


class TestMaxDrawdown:
    def test_no_drawdown(self):
        """Растущий equity — drawdown = 0."""
        eq = pd.Series([100, 110, 120, 130, 140])
        dd = compute_max_drawdown(eq)
        assert dd["max_drawdown"] == 0.0

    def test_known_drawdown(self):
        """Известная просадка."""
        dates = pd.date_range("2024-01-01", periods=5, freq="1d", tz="UTC")
        eq = pd.Series([100, 110, 100, 90, 120], index=dates)
        dd = compute_max_drawdown(eq)
        assert dd["max_drawdown"] > 0
        assert dd["max_drawdown_pct"] > 0
        # Пик был на 110, просадка до 90
        assert dd["max_drawdown"] == 20.0

    def test_drawdown_duration(self):
        """Длительность просадки > 0."""
        dates = pd.date_range("2024-01-01", periods=6, freq="1d", tz="UTC")
        eq = pd.Series([100, 110, 100, 90, 100, 120], index=dates)
        dd = compute_max_drawdown(eq)
        assert dd["duration"] >= 0


# ---------------------------------------------------------------------------
# Monthly Returns
# ---------------------------------------------------------------------------


class TestMonthlyReturns:
    def test_monthly_returns_structure(self):
        eq = _make_equity_curve(n=365 * 24)
        monthly = compute_monthly_returns(eq)
        assert not monthly.empty
        # Должны быть колонки с месяцами
        assert monthly.shape[1] >= 1


# ---------------------------------------------------------------------------
# Compute Metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_basic_metrics(self):
        eq = _make_equity_curve(n=100)
        trades = _make_sample_trades(n_wins=5, n_losses=2)

        m = compute_metrics(trades, eq, initial_equity=100000.0)

        assert isinstance(m, PerformanceMetrics)
        assert m.total_trades == 7  # 5 + 2 (unclosed не считаются)
        assert m.win_trades == 5
        assert m.loss_trades == 2
        assert m.win_rate > 0

    def test_total_return_positive(self):
        """При росте equity return должен быть положительным."""
        eq = _make_equity_curve(n=100, drift=0.001)
        trades = _make_sample_trades(n_wins=3, n_losses=1)
        m = compute_metrics(trades, eq, initial_equity=100000.0)
        assert m.total_return > 0

    def test_total_return_negative(self):
        """При падении equity return отрицательный."""
        eq = _make_equity_curve(n=100, drift=-0.001)
        trades = _make_sample_trades(n_wins=1, n_losses=3)
        m = compute_metrics(trades, eq, initial_equity=100000.0)
        assert m.total_return < 0

    def test_all_unclosed(self):
        """Все сделки unclosed — total_trades = 0."""
        eq = _make_equity_curve(n=100)
        trades = [Trade(
            symbol="TEST/USDT",
            entry_time=pd.Timestamp("2024-01-01", tz="UTC"),
            entry_price=100, entry_size=1,
            stop_price=95, take_profit_price=None,
            is_closed=False,
        )]
        m = compute_metrics(trades, eq, initial_equity=100000.0)
        assert m.total_trades == 0

    def test_metrics_has_final_equity(self):
        eq = _make_equity_curve(n=100)
        m = compute_metrics([], eq, initial_equity=100000.0)
        assert m.final_equity > 0


# ---------------------------------------------------------------------------
# Format Report
# ---------------------------------------------------------------------------


class TestFormatReport:
    def test_report_contains_key_sections(self):
        m = PerformanceMetrics(
            total_return=0.15,
            cagr=0.12,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown_pct=0.08,
            total_trades=100,
            win_rate=0.55,
            profit_factor=1.3,
        )
        report = format_metrics_report(m)
        assert "ДОХОДНОСТЬ" in report
        assert "РИСК" in report
        assert "ТОРГОВЛЯ" in report
        assert "15.00%" in report
        assert "12.00%" in report

    def test_report_contains_metrics_values(self):
        m = PerformanceMetrics(
            total_return=0.25,
            cagr=0.20,
            sharpe_ratio=1.8,
            total_trades=50,
            win_rate=0.60,
            max_drawdown_pct=0.10,
            final_equity=125000.0,
            initial_equity=100000.0,
        )
        report = format_metrics_report(m)
        assert "25.00%" in report
        assert "20.00%" in report
        assert "60.00%" in report
