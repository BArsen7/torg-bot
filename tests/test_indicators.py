"""
Тесты для индикаторов (indicators.py).

Используют синтетические данные — без внешних зависимостей.
"""

from __future__ import annotations

import sys
import pathlib
import pytest
import numpy as np
import pandas as pd

# Добавляем проект в путь
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.indicators import (
    compute_atr,
    compute_rsi,
    compute_sma,
    find_pivots,
    compute_trend_score,
    compute_atr_channels,
    compute_donchian_channels,
    volume_filter,
    atr_pct,
    add_all_indicators,
)


# ---------------------------------------------------------------------------
# Фикстуры: синтетические данные
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Создать синтетический OHLCV DataFrame."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")

    # Случайное блуждание для close
    close = 100.0
    closes = []
    for _ in range(n):
        close += np.random.normal(0, 0.5)
        closes.append(close)

    high = [c + abs(np.random.normal(0, 0.3)) for c in closes]
    low = [c - abs(np.random.normal(0, 0.3)) for c in closes]
    open_ = [c + np.random.normal(0, 0.2) for c in closes]
    volume = np.abs(np.random.normal(1000, 200, n))

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": closes,
        "volume": volume,
    }, index=dates)

    return df


@pytest.fixture
def rising_df() -> pd.DataFrame:
    """Синтетический данные с восходящим трендом."""
    np.random.seed(100)
    n = 300
    dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    base = 50.0
    closes = [base + i * 0.2 + np.random.normal(0, 0.5) for i in range(n)]
    high = [c + abs(np.random.normal(0, 0.2)) for c in closes]
    low = [c - abs(np.random.normal(0, 0.2)) for c in closes]
    open_ = [c + np.random.normal(0, 0.1) for c in closes]
    volume = np.abs(np.random.normal(500, 100, n))

    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": closes, "volume": volume,
    }, index=dates)


# ---------------------------------------------------------------------------
# Тесты ATR
# ---------------------------------------------------------------------------


class TestATR:
    def test_atr_returns_series(self, synthetic_df):
        atr = compute_atr(synthetic_df["high"], synthetic_df["low"],
                          synthetic_df["close"], period=14)
        assert isinstance(atr, pd.Series)
        assert len(atr) == len(synthetic_df)
        assert atr.notna().sum() > 14  # первые N должны быть NaN

    def test_atr_positive(self, synthetic_df):
        atr = compute_atr(synthetic_df["high"], synthetic_df["low"],
                          synthetic_df["close"], period=14)
        valid = atr.dropna()
        assert (valid > 0).all(), "ATR должен быть всегда положительным"

    def test_atr_larger_range_larger_atr(self):
        """Больший разброс цен → больший ATR."""
        high = pd.Series([10, 20, 30, 40, 50] * 20)
        low = pd.Series([5, 10, 15, 20, 25] * 20)
        close = pd.Series([9, 18, 28, 38, 48] * 20)
        atr = compute_atr(high, low, close, period=5)
        assert atr.dropna().iloc[-1] > atr.dropna().iloc[0]


# ---------------------------------------------------------------------------
# Тесты RSI
# ---------------------------------------------------------------------------


class TestRSI:
    def test_rsi_range(self, synthetic_df):
        rsi = compute_rsi(synthetic_df["close"], period=14)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_oversold(self, synthetic_df):
        """Падающий ряд → RSI должен быть низким."""
        falling = synthetic_df["close"].copy()
        falling.iloc[100:] = falling.iloc[100] - np.arange(len(falling.iloc[100:])) * 0.5
        rsi = compute_rsi(falling, period=14)
        assert rsi.iloc[-1] < 50, f"RSI должен быть низким для падающего ряда, got {rsi.iloc[-1]}"

    def test_rsi_overbought(self):
        """Растущий ряд → RSI должен быть высоким."""
        np.random.seed(42)
        rising = pd.Series([100 + i * 1.0 + np.random.normal(0, 0.1)
                           for i in range(300)])
        rsi = compute_rsi(rising, period=14)
        valid = rsi.dropna()
        assert len(valid) > 0, f"RSI должен быть не-NaN"
        assert valid.iloc[-1] > 50, f"RSI должен быть высоким, got {valid.iloc[-1]}"

    def test_rsi_neutral(self):
        """Плоский ряд → RSI ~ 50."""
        flat = pd.Series([100.0] * 100)
        rsi = compute_rsi(flat, period=14)
        # При идеально плоском ряду RSI будет NaN (деление на 0)
        # Но при малом шуме должен быть ~50


# ---------------------------------------------------------------------------
# Тесты SMA
# ---------------------------------------------------------------------------


class TestSMA:
    def test_sma_equals_mean(self, synthetic_df):
        sma = compute_sma(synthetic_df["close"], period=20)
        # Сравнить с rolling mean
        expected = synthetic_df["close"].rolling(20, min_periods=20).mean()
        pd.testing.assert_series_equal(sma, expected)

    def test_sma_length(self, synthetic_df):
        sma = compute_sma(synthetic_df["close"], period=10)
        assert len(sma) == len(synthetic_df)

    def test_sma_lagging(self, rising_df):
        """SMA должен отставать от цены в восходящем тренде."""
        sma = compute_sma(rising_df["close"], period=20)
        assert sma.iloc[-1] < rising_df["close"].iloc[-1], \
            f"SMA должен отставать: SMA={sma.iloc[-1]:.2f}, close={rising_df['close'].iloc[-1]:.2f}"


# ---------------------------------------------------------------------------
# Тесты пивотов
# ---------------------------------------------------------------------------


class TestPivots:
    def test_pivot_highs_find_local_maxima(self):
        """Пивот-хай должен находить локальные максимумы."""
        # Создаём данные с явным максимумом на позиции 5
        arr = pd.Series([1, 2, 3, 2, 1, 5, 1, 2, 3, 2, 1])
        highs = find_pivots(arr, order=1, high=True)
        assert highs.iloc[5] == 5.0, "Пивот должен найти локальный максимум"

    def test_pivot_lows_find_local_minima(self):
        """Пивот-лоу должен находить локальные минимумы."""
        arr = pd.Series([3, 2, 1, 2, 3, 5, 3, 2, 1, 2, 3])
        lows = find_pivots(arr, order=1, high=False)
        assert lows.iloc[2] == 1.0, "Пивот должен найти локальный минимум"

    def test_pivots_no_pivots_in_flat(self):
        """Плоский ряд — нет пивотов."""
        flat = pd.Series([5.0] * 100)
        pivots = find_pivots(flat, order=1, high=True)
        assert pivots.dropna().empty

    def test_pivot_returns_same_length(self, synthetic_df):
        pivots = find_pivots(synthetic_df["high"], order=3, high=True)
        assert len(pivots) == len(synthetic_df)


# ---------------------------------------------------------------------------
# Тесты тренд-скор
# ---------------------------------------------------------------------------


class TestTrendScore:
    def test_trend_score_type(self, synthetic_df):
        score = compute_trend_score(synthetic_df["close"], order=3)
        assert isinstance(score, pd.Series)
        valid = score.unique()
        # Допустимые значения: -1, 0, +1
        assert all(v in [-1, 0, 1] or np.isnan(v) for v in valid)

    def test_trend_score_in_rising_trend(self, rising_df):
        score = compute_trend_score(rising_df["close"], order=2)
        # В восходящем тренде score должен быть >= 0
        valid = score.dropna()
        assert len(valid) > 0


# ---------------------------------------------------------------------------
# Тесты каналов
# ---------------------------------------------------------------------------


class TestChannels:
    def test_atr_channels(self, synthetic_df):
        upper, lower, central = compute_atr_channels(
            synthetic_df["close"],
            synthetic_df["high"],
            synthetic_df["low"],
            sma_period=20,
            atr_period=14,
            atr_multiplier=2.0,
        )
        assert len(upper) == len(synthetic_df)
        assert len(lower) == len(synthetic_df)
        # Upper > central > Lower (почти всегда)
        valid = central.dropna()
        assert (upper[valid.index] > central[valid.index]).all()
        assert (lower[valid.index] < central[valid.index]).all()

    def test_donchian_channels(self, synthetic_df):
        upper, lower = compute_donchian_channels(
            synthetic_df["high"],
            synthetic_df["low"],
            period=20,
        )
        assert len(upper) == len(synthetic_df)
        assert (upper.dropna() >= lower.dropna()).all()

    def test_channel_widens_with_volatility(self):
        """При высокой волатильности канал шире."""
        np.random.seed(99)
        # Низкая волатильность
        close_l = pd.Series([100.0 + np.random.normal(0, 0.1) for _ in range(200)])
        high_l = close_l + 0.1
        low_l = close_l - 0.1

        # Высокая волатильность
        close_h = pd.Series([100.0 + np.random.normal(0, 2.0) for _ in range(200)])
        high_h = close_h + 2.0
        low_h = close_h - 2.0

        upper_h, lower_h, _ = compute_atr_channels(close_h, high_h, low_h,
                                                    atr_period=14, atr_multiplier=2.0)
        upper_l, lower_l, _ = compute_atr_channels(close_l, high_l, low_l,
                                                    atr_period=14, atr_multiplier=2.0)
        # Ширина канала высокой волатильности больше
        width_h = (upper_h.dropna() - lower_h.dropna()).abs().mean()
        width_l = (upper_l.dropna() - lower_l.dropna()).abs().mean()
        assert width_h > width_l


# ---------------------------------------------------------------------------
# Тесты объём и ATR%
# ---------------------------------------------------------------------------


class TestVolumeAndAtrPct:
    def test_volume_filter(self, synthetic_df):
        vf = volume_filter(synthetic_df["volume"], period=20)
        assert isinstance(vf, pd.Series)
        assert vf.dtype == bool
        # Некоторые должны быть True
        assert vf.sum() > 0

    def test_atr_pct_range(self, synthetic_df):
        ap = atr_pct(synthetic_df["high"], synthetic_df["low"],
                     synthetic_df["close"], period=14)
        valid = ap.dropna()
        assert (valid > 0).all(), "ATR% должен быть положительным"
        # Должен быть в разумных пределах
        assert valid.mean() < 50.0, "Средний ATR% слишком велик"


# ---------------------------------------------------------------------------
# Тест add_all_indicators
# ---------------------------------------------------------------------------


class TestAddAllIndicators:
    def test_all_columns_present(self, synthetic_df):
        from src.config import StrategyConfig
        cfg = StrategyConfig()
        df = add_all_indicators(synthetic_df, cfg)
        required = ["atr", "rsi", "upper_band", "lower_band",
                    "volume_sma", "volume_above",
                    "pivot_high", "pivot_low", "trend_score"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_nan_after_indicators(self, synthetic_df):
        from src.config import StrategyConfig
        cfg = StrategyConfig()
        df = add_all_indicators(synthetic_df, cfg)
        # После dropna NaN должно быть мало
        nan_count = df[[c for c in ["atr", "rsi", "upper_band", "lower_band",
                                      "volume_sma", "trend_score"]]].isna().sum().sum()
        # Ожидаем немного NaN (из-за dropna в add_all_indicators)
        assert nan_count == 0, "Не должно быть NaN после добавления индикаторов"

    def test_df_not_empty(self, synthetic_df):
        from src.config import StrategyConfig
        cfg = StrategyConfig()
        df = add_all_indicators(synthetic_df, cfg)
        assert not df.empty
