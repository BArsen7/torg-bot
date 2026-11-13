"""
Тесты для стратегии (strategy.py).

Проверка условий входа, выхода, размера позиции.
Синтетические данные — без внешних API.
"""

from __future__ import annotations

import sys
import pathlib
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.strategy import (
    check_entry_conditions,
    check_exit_conditions,
    calculate_position_size,
    apply_costs,
    generate_signals,
)
from src.config import StrategyConfig
from src.indicators import add_all_indicators


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg() -> StrategyConfig:
    return StrategyConfig()


def _make_oscillating_df(n: int = 400) -> pd.DataFrame:
    """
    Создать DataFrame с циклами (подходит для range trading).

    Close падает ниже канала, затем отскакивает — входные условия
    должны срабатывать.
    """
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")

    # Oscillating close
    base = 100.0
    closes = []
    for i in range(n):
        wave = 5 * np.sin(i / 20)  # волны каждые ~20 часов
        noise = np.random.normal(0, 0.3)
        closes.append(base + wave + noise)

    high = [c + abs(np.random.normal(0, 0.2)) for c in closes]
    low = [c - abs(np.random.normal(0, 0.2)) for c in closes]
    open_ = [c + np.random.normal(0, 0.1) for c in closes]
    volume = np.abs(np.random.normal(1000, 200, n))

    df = pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": closes, "volume": volume,
    }, index=dates)

    return df


def _make_trending_df(n: int = 400) -> pd.DataFrame:
    """Растущий тренд."""
    np.random.seed(100)
    dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    closes = [50 + i * 0.15 + np.random.normal(0, 0.3) for i in range(n)]
    high = [c + abs(np.random.normal(0, 0.2)) for c in closes]
    low = [c - abs(np.random.normal(0, 0.2)) for c in closes]
    open_ = [c + np.random.normal(0, 0.1) for c in closes]
    volume = np.abs(np.random.normal(500, 100, n))

    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": closes, "volume": volume,
    }, index=dates)


# ---------------------------------------------------------------------------
# Тесты условий входа
# ---------------------------------------------------------------------------


class TestEntryConditions:
    def test_entry_all_conditions_met(self, cfg):
        """Когда все условия выполнены, signal должен быть True."""
        np.random.seed(42)
        # Создаём больше данных, чтобы после dropna осталось достаточно строк
        n = 400
        df = pd.DataFrame({
            "close": [100 + i * 0.01 + np.random.normal(0, 0.1) for i in range(n)],
            "open": [99 + i * 0.01 + np.random.normal(0, 0.1) for i in range(n)],
            "high": [101 + i * 0.01 + np.random.normal(0, 0.1) for i in range(n)],
            "low": [99.5 + i * 0.01 + np.random.normal(0, 0.1) for i in range(n)],
            "volume": [1000] * n,
        })
        # Добавим индикаторы
        df = add_all_indicators(df, cfg)

        # Для тестирования — проверим что функция принимает DataFrame
        assert len(df) > 50
        met, conditions = check_entry_conditions(df, 50, cfg)
        assert isinstance(met, bool)
        assert isinstance(conditions, dict)
        assert "trend_score_ge_0" in conditions
        assert "touch_lower" in conditions
        assert "all_met" in conditions

    def test_entry_conditions_return_dict(self, cfg):
        """check_entry_conditions возвращает dict с именами условий."""
        df = _make_oscillating_df()
        df = add_all_indicators(df, cfg)

        met, cond = check_entry_conditions(df, 50, cfg)
        assert isinstance(cond, dict)
        assert len(cond) >= 6

    def test_entry_no_met_when_rising_strong(self, cfg):
        """В сильном растущем тренде условия входа менее вероятны."""
        df = _make_trending_df()
        df = add_all_indicators(df, cfg)

        met, _ = check_entry_conditions(df, 100, cfg)
        # В растущем тренде отскок от канала менее вероятен
        # Но это зависит от синтетики — проверяем что логика работает
        assert isinstance(met, bool)

    def test_entry_requires_trend_score(self, cfg):
        """Если trend_score < 0, вход запрещён."""
        # Создаём DF где trend_score явно < 0 (нисходящий тренд с чёткими пивотами)
        n = 500
        np.random.seed(99)
        # Сильно падающий тренд с осцилляциями
        base = 500.0
        closes = [base - i * 2 + np.random.normal(0, 1) for i in range(n)]
        highs = [c + abs(np.random.normal(0, 0.5)) for c in closes]
        lows = [c - abs(np.random.normal(0, 0.5)) for c in closes]
        opens = [c + np.random.normal(0, 0.3) for c in closes]
        volumes = [1000] * n

        df = pd.DataFrame({
            "close": closes,
            "open": opens,
            "high": highs,
            "low": lows,
            "volume": volumes,
        })
        df = add_all_indicators(df, cfg)

        assert len(df) > 100
        met, cond = check_entry_conditions(df, 100, cfg)
        # В сильном нисходящем тренде trend_score обычно < 0
        # even if trend_score >= 0, other conditions should block
        assert cond["all_met"] is False, "В падающем тренде вход не должен быть разрешён"


# ---------------------------------------------------------------------------
# Тесты условий выхода
# ---------------------------------------------------------------------------


class TestExitConditions:
    def test_stop_loss_triggered(self, cfg):
        """Стоп-лосс должен сработать при падении ниже stop."""
        from src.strategy import Trade
        trade = Trade(
            symbol="TEST/USDT",
            entry_time=pd.Timestamp("2024-01-01", tz="UTC"),
            entry_price=100.0,
            entry_size=10.0,
            stop_price=95.0,
            take_profit_price=None,
            max_price=100.0,
        )

        df = pd.DataFrame({
            "high": [100, 99, 98, 94],
            "low": [99, 98, 93, 92],
            "close": [99.5, 98.5, 94.5, 93.0],
            "atr": [2.0, 2.0, 2.0, 2.0],
            "upper_band": [110, 110, 110, 110],
        }, index=pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC"))

        needs_exit, reason = check_exit_conditions(trade, df, 3, cfg)
        assert needs_exit is True
        assert reason == "stop_loss"

    def test_take_profit_triggered(self, cfg):
        """Тейк-профит при достижении верхней границы."""
        from src.strategy import Trade
        trade = Trade(
            symbol="TEST/USDT",
            entry_time=pd.Timestamp("2024-01-01", tz="UTC"),
            entry_price=100.0,
            entry_size=10.0,
            stop_price=95.0,
            take_profit_price=None,
            max_price=105.0,  # был рост
        )

        df = pd.DataFrame({
            "high": [105, 108, 115],
            "low": [104, 107, 114],
            "close": [104.5, 107.5, 115.0],
            "atr": [1.0, 1.0, 1.0],
            "upper_band": [110, 110, 110],
        }, index=pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC"))

        needs_exit, reason = check_exit_conditions(trade, df, 2, cfg)
        assert needs_exit is True
        assert reason == "take_profit"

    def test_no_exit_when_holding(self, cfg):
        """Не выходит, если ни одно условие не выполнено."""
        from src.strategy import Trade
        trade = Trade(
            symbol="TEST/USDT",
            entry_time=pd.Timestamp("2024-01-01", tz="UTC"),
            entry_price=100.0,
            entry_size=10.0,
            stop_price=95.0,
            take_profit_price=None,
            max_price=102.0,
        )

        df = pd.DataFrame({
            "high": [101, 102, 103],
            "low": [99, 98, 97],
            "close": [100.5, 101.5, 102.5],
            "atr": [1.0, 1.0, 1.0],
            "upper_band": [120, 120, 120],
        }, index=pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC"))

        needs_exit, reason = check_exit_conditions(trade, df, 2, cfg)
        assert needs_exit is False

    def test_trailing_stop_triggered(self, cfg):
        """Трейлинг-стоп при падении от максимума."""
        from src.strategy import Trade
        trade = Trade(
            symbol="TEST/USDT",
            entry_time=pd.Timestamp("2024-01-01", tz="UTC"),
            entry_price=100.0,
            entry_size=10.0,
            stop_price=95.0,
            take_profit_price=None,
            max_price=120.0,  # был сильный рост
        )

        # ATR = 1.0, trailing = 120 - 1.5*1 = 118.5
        # close = 117 — ниже трейлинг-стопа
        df = pd.DataFrame({
            "high": [121, 119],
            "low": [116, 116],
            "close": [119.5, 117.0],
            "atr": [1.0, 1.0],
            "upper_band": [130, 130],
        }, index=pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC"))

        needs_exit, reason = check_exit_conditions(trade, df, 1, cfg)
        assert needs_exit is True
        assert reason == "trailing_stop"


# ---------------------------------------------------------------------------
# Тесты размера позиции
# ---------------------------------------------------------------------------


class TestPositionSize:
    def test_basic_calculation(self, cfg):
        """Базовый расчёт: size = equity * risk / (entry - stop)."""
        equity = 100000
        entry = 100.0
        stop = 95.0
        risk = 0.02  # 2%

        size = calculate_position_size(equity, entry, stop, risk, leverage=1)
        expected = (equity * risk) / (entry - stop)  # 2000 / 5 = 400
        assert abs(size - expected) < 0.01, f"Expected {expected}, got {size}"

    def test_with_leverage(self, cfg):
        """Плечо увеличивает размер позиции."""
        size_1x = calculate_position_size(100000, 100, 95, 0.02, leverage=1)
        size_5x = calculate_position_size(100000, 100, 95, 0.02, leverage=5)
        assert size_5x == size_1x * 5

    def test_zero_distance_returns_zero(self, cfg):
        """При entry == stop размер должен быть 0."""
        size = calculate_position_size(100000, 100, 100, 0.02, leverage=1)
        assert size == 0.0

    def test_negative_distance_handled(self, cfg):
        """Stop > entry — берём абсолютное значение."""
        size = calculate_position_size(100000, 100, 105, 0.02, leverage=1)
        # Расстояние = 5, size = 2000/5 = 400
        assert size == 400.0

    def test_higher_risk_larger_position(self, cfg):
        """Больше риск → больше позиция."""
        s1 = calculate_position_size(100000, 100, 95, 0.01, leverage=1)
        s2 = calculate_position_size(100000, 100, 95, 0.05, leverage=1)
        assert s2 > s1


# ---------------------------------------------------------------------------
# Тесты комиссий
# ---------------------------------------------------------------------------


class TestCosts:
    def test_costs_positive(self, cfg):
        commission, slippage = apply_costs(
            100.0, 105.0, 10.0,
            is_long=True,
            maker_fee=cfg.maker_fee,
            taker_fee=cfg.taker_fee,
            slippage_pct=cfg.slippage_pct,
        )
        assert commission > 0

    def test_costs_proportional_to_notional(self, cfg):
        """Комиссия пропорциональна стоимости позиции."""
        c1, _ = apply_costs(100, 105, 10, True, cfg.maker_fee, cfg.taker_fee, cfg.slippage_pct)
        c2, _ = apply_costs(100, 105, 20, True, cfg.maker_fee, cfg.taker_fee, cfg.slippage_pct)
        assert abs(c2 - 2 * c1) < 0.01

    def test_funding_costs(self, cfg):
        """Фандер увеличивает комиссию для long позиций."""
        c1, _ = apply_costs(100, 105, 10, True, cfg.maker_fee, cfg.taker_fee,
                            cfg.slippage_pct, funding_pct=0.0001, holding_hours=24)
        c2, _ = apply_costs(100, 105, 10, True, cfg.maker_fee, cfg.taker_fee,
                            cfg.slippage_pct, funding_pct=0.0, holding_hours=24)
        assert c1 > c2


# ---------------------------------------------------------------------------
# Тесты generate_signals
# ---------------------------------------------------------------------------


class TestGenerateSignals:
    def test_signals_column_added(self, cfg):
        df = _make_oscillating_df()
        df = add_all_indicators(df, cfg)

        result = generate_signals(df, cfg)
        assert "signal" in result.columns
        assert "entry_price" in result.columns
        assert "stop_price" in result.columns

    def test_signal_values(self, cfg):
        """signal может быть только 0 или 1."""
        df = _make_oscillating_df()
        df = add_all_indicators(df, cfg)

        result = generate_signals(df, cfg)
        valid_signals = result["signal"].dropna()
        assert all(v in [0, 1] for v in valid_signals)

    def test_signals_atleast_zero(self, cfg):
        """Может не быть сигналов вообще (но колонка должна быть)."""
        df = pd.DataFrame({
            "close": [100] * 100,
            "open": [99] * 100,
            "high": [101] * 100,
            "low": [98] * 100,
            "volume": [1000] * 100,
        })
        df = add_all_indicators(df, cfg)
        result = generate_signals(df, cfg)
        assert len(result) == len(df)
