"""
Технические индикаторы для Range Trading стратегии.

Все функции векторизованы через pandas/numpy — циклы только там,
где без них нельзя (свинг-пивоты).

Не использовать `.shift(-1)` для данных, которые будут использованы
в расчёте — это causa look-ahead bias.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Вычислить ATR (Average True Range) через Wilder's smoothing.

    Параметры
    ---------
    high, low, close : pd.Series
    period : int

    Возвращает
    ----------
    pd.Series — ATR значения
    """
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(span=period, adjust=False, min_periods=period).mean()
    return atr


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Вычислить RSI (Relative Strength Index) через Wilder's smoothing.

    Параметры
    ---------
    close : pd.Series
    period : int

    Возвращает
    ----------
    pd.Series — RSI значения (0..100)
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(span=period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(span=period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


# ---------------------------------------------------------------------------
# SMA
# ---------------------------------------------------------------------------


def compute_sma(series: pd.Series, period: int) -> pd.Series:
    """Простая скользящая средняя."""
    return series.rolling(window=period, min_periods=period).mean()


# ---------------------------------------------------------------------------
# Свинг-пивоты
# ---------------------------------------------------------------------------


def find_pivots(
    series: pd.Series,
    order: int = 3,
    high: bool = True,
) -> pd.Series:
    """
    Найти локальные максимумы или минимумы.

    Используем скользящее окно 2*k+1:
    pivot_high[i] = max(high[i-k:i+k]) если i — локальный максимум.

    Для flat series (все значения в окне равны) пивотов нет.

    Параметры
    ---------
    series : pd.Series — цены (high для максимумов, low для минимумов)
    order : int — k, размер окна
    high : bool — True для pivot highs, False для pivot lows

    Возвращает
    ----------
    pd.Series — те же индексы, NaN где пивота нет
    """
    result = pd.Series(np.nan, index=series.index, dtype=float)
    k = order
    arr = series.values

    for i in range(k, len(arr) - k):
        window_start = i - k
        window_end = i + k
        window = arr[window_start:window_end]
        window_min = window.min()
        window_max = window.max()

        # Если все значения в окне равны — не пивот (flat series)
        if window_min == window_max:
            continue

        if high:
            # Локальный максимум: current > все остальные в окне
            if arr[i] == window_max and arr[i] > window_min:
                result.iloc[i] = arr[i]
        else:
            # Локальный минимум: current < все остальные в окне
            if arr[i] == window_min and arr[i] < window_max:
                result.iloc[i] = arr[i]

    return result


def compute_trend_score(
    close: pd.Series,
    order: int = 3,
    fast_period: int = 20,
    slow_period: int = 50,
) -> pd.Series:
    """
    Определить trend_score скользящим окном.

    Для каждого бара i:
    1. Берём EMA(close, fast_period) и EMA(close, slow_period)
    2. Если EMA_fast > EMA_slow → бычий тренд (+1)
    3. Если EMA_fast < EMA_slow → медвежий тренд (-1)
    4. Если близко к равенству → нейтральный (0)

    Это векторизованная версия, работает для каждого бара отдельно.

    Параметры
    ---------
    close : pd.Series
    order : int — игнорируется, оставлен для совместимости
    fast_period : int — быстрая EMA
    slow_period : int — медленная EMA

    Возвращает
    ----------
    pd.Series — trend_score (+1, 0, или -1) для каждого бара
    """
    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()

    # Разница между EMA
    diff = ema_fast - ema_slow

    # Если разница > 0.1% от цены — считаем трендом
    threshold = close * 0.001
    trend_score = np.where(
        diff > threshold, 1,
        np.where(diff < -threshold, -1, 0)
    )
    return pd.Series(trend_score, index=close.index)


# ---------------------------------------------------------------------------
# ATR-каналы
# ---------------------------------------------------------------------------


def compute_atr_channels(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    sma_period: int = 30,
    atr_period: int = 14,
    atr_multiplier: float = 2.2,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Построить ATR-каналы.

    central = SMA(close, p)
    upper  = central + k * ATR
    lower  = central - k * ATR

    Параметры
    ---------
    close, high, low : pd.Series
    sma_period : int
    atr_period : int
    atr_multiplier : float (k)

    Возвращает
    ----------
    (upper_band, lower_band, central)
    """
    atr = compute_atr(high, low, close, period=atr_period)
    sma = compute_sma(close, sma_period)

    upper = sma + atr_multiplier * atr
    lower = sma - atr_multiplier * atr

    return upper, lower, sma


# ---------------------------------------------------------------------------
# Donchian канал
# ---------------------------------------------------------------------------


def compute_donchian_channels(
    high: pd.Series,
    low: pd.Series,
    period: int = 30,
) -> Tuple[pd.Series, pd.Series]:
    """
    Построить Donchian канал.

    upper = max(high[-period:])
    lower = min(low[-period:])
    """
    upper = high.rolling(window=period, min_periods=period).max()
    lower = low.rolling(window=period, min_periods=period).min()
    return upper, lower


# ---------------------------------------------------------------------------
# Объёмная проверка
# ---------------------------------------------------------------------------


def volume_filter(
    volume: pd.Series,
    period: int = 50,
) -> pd.Series:
    """
    Проверка: объём > SMA(volume, period).

    Возвращает
    ----------
    pd.Series — булев маск (True если объём достаточный)
    """
    avg_vol = compute_sma(volume, period)
    return volume > avg_vol


# ---------------------------------------------------------------------------
# ATR % (волатильность)
# ---------------------------------------------------------------------------


def atr_pct(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    ATR в процентах от цены: ATR / close * 100.

    Параметры
    ---------
    high, low, close : pd.Series
    period : int

    Возвращает
    ----------
    pd.Series — ATR_pct
    """
    atr = compute_atr(high, low, close, period=period)
    return atr / close * 100.0




def add_sma(df: pd.DataFrame, period: int = 200, col: str = "close") -> pd.DataFrame:
    """Добавляет Simple Moving Average для фильтра долгосрочного тренда."""
    df[f"sma_{period}"] = df[col].rolling(window=period).mean()
    return df

def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Рассчитывает ADX (Average Directional Index).
    ADX > 25 указывает на наличие тренда, < 20 — на боковик.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # +DM и -DM
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Сглаживание (Wilder's smoothing аналогичен EMA с alpha=1/period)
    atr = tr.ewm(alpha=1/period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)

    # DX и ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    df["adx"] = dx.ewm(alpha=1/period, min_periods=period).mean()
    return df




# ---------------------------------------------------------------------------
# Применить все индикаторы к DataFrame
# ---------------------------------------------------------------------------


def add_all_indicators(
    df: pd.DataFrame,
    strategy_cfg,
) -> pd.DataFrame:
    """
    Добавить все технические индикаторы к OHLCV DataFrame.

    Колонки:
    - atr: ATR
    - rsi: RSI(14)
    - atr_pct: ATR/close*100
    - sma_central: SMA(close, p)
    - upper_band: SMA ± k*ATR (или Donchian upper)
    - lower_band: SMA − k*ATR (или Donchian lower)
    - volume_above: True если volume > SMA(volume, 50)
    - pivot_high: свинг-максимумы
    - pivot_low: свинг-минимумы
    - trend_score: тренд (+1, 0, -1)

    Параметры
    ---------
    df : pd.DataFrame — OHLCV
    strategy_cfg : StrategyConfig

    Возвращает
    ----------
    pd.DataFrame — с добавленными колонками
    """
    df = df.copy()

    # --- ATR + RSI ---
    df["atr"] = compute_atr(df["high"], df["low"], df["close"],
                            period=strategy_cfg.atr_period)
    df["rsi"] = compute_rsi(df["close"], period=strategy_cfg.rsi_period)
    df["atr_pct"] = atr_pct(df["high"], df["low"], df["close"],
                            period=strategy_cfg.atr_period)

    # --- Каналы ---
    if strategy_cfg.channel_method == "atr":
        upper, lower, central = compute_atr_channels(
            df["close"], df["high"], df["low"],
            sma_period=strategy_cfg.sma_period,
            atr_period=strategy_cfg.atr_period,
            atr_multiplier=strategy_cfg.atr_multiplier,
        )
    else:
        upper, lower = compute_donchian_channels(
            df["high"], df["low"],
            period=strategy_cfg.donchian_period,
        )
        central = (upper + lower) / 2.0

    df["upper_band"] = upper
    df["lower_band"] = lower
    df["sma_central"] = central

    # --- Объём ---
    df["volume_sma"] = compute_sma(df["volume"], strategy_cfg.volume_sma_period)
    df["volume_above"] = volume_filter(df["volume"], strategy_cfg.volume_sma_period)
    # Новый SMA(20) для фильтра объёма на входе
    df["volume_sma_20"] = compute_sma(df["volume"], 20)

    # --- Свинг-пивоты ---
    df["pivot_high"] = find_pivots(df["high"], strategy_cfg.pivot_order, high=True)
    df["pivot_low"] = find_pivots(df["low"], strategy_cfg.pivot_order, high=False)

    # --- Trend score ---
    df["trend_score"] = compute_trend_score(df["close"], order=strategy_cfg.pivot_order)

    # --- DropNA: после всех вычислений ---
    drop_cols = ["atr", "rsi", "upper_band", "lower_band", "volume_sma", "trend_score"]
    initial_len = len(df)
    df.dropna(subset=drop_cols, inplace=True)
    dropped = initial_len - len(df)
    if dropped > 0:
        logger.debug("Dropped %d rows due to NaN indicators", dropped)

    df = add_sma(df, period=200, col="close")
    df = add_adx(df, period=14)
    return df
