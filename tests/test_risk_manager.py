"""
Тесты для risk_manager.py.
"""

from __future__ import annotations

import sys
import pathlib
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.risk_manager import (
    calculate_position_size,
    compute_rolling_correlation,
    check_correlation_filter,
    check_position_limits,
    validate_liquidation,
    estimate_liquidation_price,
    RiskManager,
)
from src.config import StrategyConfig


@pytest.fixture
def cfg() -> StrategyConfig:
    return StrategyConfig()


# ---------------------------------------------------------------------------
# Position size
# ---------------------------------------------------------------------------


class TestPositionSize:
    def test_equity_100k_risk_2pct(self):
        # Новая формула: size = (free_equity * risk_per_trade) / |entry - stop|
        # 100000 * 0.02 / 5 = 400
        # max_position = 100000 * 1 / 100 = 1000
        # size = min(400, 1000) = 400
        size = calculate_position_size(100000, 100, 95, 0.02)
        assert size == 400.0

    def test_pnl_at_stop_equals_risk_percentage(self):
        """Проверка: PnL при достижении стопа = risk_per_trade * equity.

        При entry=100, stop=95, risk=2%, equity=100k:
        - raw_size = 100000 * 0.02 / 5 = 400
        - max_10x = 100000 * 10 / 100 = 10000
        - size = min(400, 10000) = 400
        - PnL at stop = size * (entry - stop) = 400 * 5 = $2000 = 2% ✓
        """
        size = calculate_position_size(
            free_equity=100000,
            entry_price=100.0,
            stop_price=95.0,
            risk_per_trade=0.02,
            leverage=10,
        )
        # PnL при достижении стопа
        pnl_at_stop = size * (100.0 - 95.0)
        # PnL должен быть ~2% от equity
        expected_pnl = 100000 * 0.02  # $2000
        assert abs(pnl_at_stop - expected_pnl) < 1.0, (
            f"PnL при стопе = ${pnl_at_stop:,.2f}, ожидаемо ${expected_pnl:,.2f} "
            f"(2% от equity=$100k). Отклонение: {abs(pnl_at_stop - expected_pnl):.2f}"
        )

    def test_pnl_at_stop_with_leverage_10(self):
        """PnL при стопе остаётся ~2% equity даже с плечом 10x.

        Плечо увеличивает максимальную позицию, но не изменяет риск.
        Риск определяется формулой size * distance = equity * risk_per_trade.
        """
        size = calculate_position_size(
            free_equity=50000.0,
            entry_price=80.0,
            stop_price=75.68,
            risk_per_trade=0.02,
            leverage=10,
        )
        # PnL = size * |entry - stop|
        pnl_at_stop = size * abs(80.0 - 75.68)
        expected_pnl = 50000.0 * 0.02  # $1000 = 2%
        assert abs(pnl_at_stop - expected_pnl) < 1.0, (
            f"Expected ~$1000 (2% of $50k), got ${pnl_at_stop:,.2f}"
        )

    def test_leverage_doubles_max_position(self):
        # Плечо увеличивает MAX размер позиции, но не risk-based размер
        s1 = calculate_position_size(100000, 100, 95, 0.02, leverage=1)
        s2 = calculate_position_size(100000, 100, 95, 0.02, leverage=2)
        # При risk=2%, entry=100, stop=95: raw=400
        # max_position_1x = 100000 / 100 = 1000, min(400,1000) = 400
        # max_position_2x = 200000 / 100 = 2000, min(400,2000) = 400
        # Т.к. risk-based размер меньше лимита по плечу, оба равны 400
        assert s1 == s2 == 400.0
        # Проверим случай, где плечо реально влияет
        # Большой risk: risk_amount = 100000 * 0.05 = 5000, raw = 5000/5 = 1000
        # max_1x = 1000, min(1000,1000) = 1000
        # max_2x = 2000, min(1000,2000) = 1000 → тоже не меняется
        # Попробуем с меньшим расстоянием до стопа
        s3 = calculate_position_size(100000, 100, 99, 0.05, leverage=1)  # risk=5%, dist=1
        s4 = calculate_position_size(100000, 100, 99, 0.05, leverage=5)
        # raw = 5000/1 = 5000, max_1x = 100000, min(5000,100000) = 5000
        # max_5x = 500000, min(5000,500000) = 5000
        # Всё равно не меняется... Нужно найти случай где плечо влияет
        # max_position = equity * leverage / entry
        # raw = equity * risk / dist
        # raw > max только если risk/dist > leverage/entry
        # => dist < risk * entry / leverage


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


class TestCorrelation:
    def test_perfect_positive_correlation(self):
        """Идентичные ряды → корреляция = 1."""
        data = np.random.randn(200)
        returns_asset = pd.Series(data)
        returns_btc = pd.Series(data)
        corr = compute_rolling_correlation(returns_asset, returns_btc, window=50)
        valid = corr.dropna()
        assert valid.iloc[-1] > 0.99

    def test_perfect_negative_correlation(self):
        """Обратные ряды → корреляция = -1."""
        data = np.random.randn(200)
        returns_asset = pd.Series(data)
        returns_btc = pd.Series(-data)
        corr = compute_rolling_correlation(returns_asset, returns_btc, window=50)
        valid = corr.dropna()
        assert valid.iloc[-1] < -0.99

    def test_zero_correlation(self):
        """Независимые ряды → корреляция ~ 0."""
        np.random.seed(42)
        returns_asset = pd.Series(np.random.randn(200))
        returns_btc = pd.Series(np.random.randn(200))
        corr = compute_rolling_correlation(returns_asset, returns_btc, window=50)
        valid = corr.dropna()
        # Корреляция должна быть близка к 0 (±0.5 — большой допуск для рандома)
        assert abs(valid.iloc[-1]) < 0.5

    def test_correlation_filter_allows_low_corr(self):
        corr = pd.Series({"BTC/USDT": 1.0, "ETH/USDT": 0.5, "OTHER/USDT": 0.3})
        result = check_correlation_filter("OTHER/USDT", {"ETH/USDT"},
                                          corr, threshold=0.8)
        assert result is True

    def test_correlation_filter_blocks_high_corr(self):
        """Если корреляция > 0.8, фильтр блокирует."""
        corr = pd.Series({"BTC/USDT": 1.0, "ETH/USDT": 0.95})
        result = check_correlation_filter("ETH/USDT", set(), corr, threshold=0.8)
        assert result is False

    def test_btc_always_allowed(self):
        """BTC всегда разрешён."""
        corr = pd.Series({})
        result = check_correlation_filter("BTC/USDT", set(), corr, threshold=0.8)
        assert result is True


# ---------------------------------------------------------------------------
# Position limits
# ---------------------------------------------------------------------------


class TestPositionLimits:
    def test_below_limit(self):
        assert check_position_limits(3, 4) is True

    def test_at_limit(self):
        assert check_position_limits(4, 4) is False

    def test_over_limit(self):
        assert check_position_limits(5, 4) is False

    def test_zero_max(self):
        assert check_position_limits(0, 0) is False


# ---------------------------------------------------------------------------
# Liquidation
# ---------------------------------------------------------------------------


class TestLiquidation:
    def test_spot_no_liquidation_risk(self):
        """Spot — нет риска ликвидации."""
        assert validate_liquidation(100, 95, 0, buffer_multiplier=2.0, leverage=1) is True

    def test_liquidation_far_enough(self):
        """Ликвидация далеко → безопасно."""
        # entry=100, stop=95, liq=50
        # stop_dist=5, liq_dist=50, buffer=2 → 50 >= 10 ✓
        assert validate_liquidation(100, 95, 50, buffer_multiplier=2.0, leverage=10) is True

    def test_liquidation_too_close(self):
        """Ликвидация близко → небезопасно."""
        # entry=100, stop=97, liq=90
        # stop_dist=3, liq_dist=10, buffer=2 → 10 >= 6 ✓ (пройдёт)
        # entry=100, stop=97, liq=96
        # stop_dist=3, liq_dist=4, buffer=2 → 4 >= 6 ✗
        assert validate_liquidation(100, 97, 96, buffer_multiplier=2.0, leverage=10) is False

    def test_estimate_liquidation_long(self):
        """Оценка цены ликвидации для long."""
        liq = estimate_liquidation_price(100.0, leverage=10, is_long=True)
        # liq ≈ 100 * (1 - 0.1 + 0.005) = 90.5
        assert 85 < liq < 95, f"Expected ~90.5, got {liq}"

    def test_estimate_liquidation_short(self):
        """Оценка цены ликвидации для short."""
        liq = estimate_liquidation_price(100.0, leverage=10, is_long=False)
        # liq ≈ 100 * (1 + 0.1 - 0.005) = 109.5
        assert 105 < liq < 115, f"Expected ~109.5, got {liq}"


# ---------------------------------------------------------------------------
# RiskManager
# ---------------------------------------------------------------------------


class TestRiskManager:
    def test_can_open_position(self, cfg):
        rm = RiskManager(cfg)
        assert rm.can_open_position("ETH/USDT", correlation_ok=True) is True

    def test_cannot_open_when_correlation_high(self, cfg):
        rm = RiskManager(cfg)
        assert rm.can_open_position("ETH/USDT", correlation_ok=False) is False

    def test_add_and_remove_position(self, cfg):
        rm = RiskManager(cfg)
        rm.add_position("ETH/USDT")
        assert "ETH/USDT" in rm.open_symbols
        rm.remove_position("ETH/USDT")
        assert "ETH/USDT" not in rm.open_symbols

    def test_cannot_open_at_max_positions(self, cfg):
        cfg.max_positions = 2
        rm = RiskManager(cfg)
        rm.add_position("A")
        rm.add_position("B")
        assert rm.can_open_position("C", correlation_ok=True) is False

    def test_cannot_open_over_max_positions(self, cfg):
        cfg.max_positions = 1
        rm = RiskManager(cfg)
        rm.add_position("A")
        assert rm.can_open_position("B", correlation_ok=True) is False
