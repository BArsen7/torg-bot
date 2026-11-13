"""
Риск-менеджмент: управление позициями, корреляционный фильтр, лимиты.

Функции:
- calculate_position_size — размер позиции на основе риска
- check_correlation — фильтрация по корреляции к BTC
- check_position_limits — проверка лимитов на одновременные позиции
- validate_liquidation — проверка расстояния до ликвидации (для futures)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from src.config import StrategyConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Размер позиции
# ---------------------------------------------------------------------------


def calculate_position_size(
    free_equity: float,
    entry_price: float,
    stop_price: float,
    risk_per_trade: float,
    leverage: int = 1,
) -> float:
    """
    Рассчитать размер позиции (M5).

    position_size = (free_equity * risk_per_trade) / |entry - stop|

    ВАЖНО: используем free_equity, а не общий equity, чтобы избежать
    оверрага. Плечо применяется только если остаток маржи достаточен.

    Параметры
    ---------
    free_equity : float — доступная маржа (не общий капитал!)
    entry_price : float — цена входа
    stop_price : float — цена стопа
    risk_per_trade : float — доля риска (0.02 = 2%)
    leverage : int — плечо

    Возвращает
    ----------
    float — размер позиции в базовом активе (0 если риск слишком велик)
    """
    if free_equity <= 0 or entry_price <= 0 or stop_price <= 0:
        logger.warning("Invalid parameters: free_equity=%.2f, entry=%.4f, stop=%.4f",
                       free_equity, entry_price, stop_price)
        return 0.0

    # Стоимость маржи, требуемой для позиции
    # Для long: маржа = position_size * entry_price / leverage
    # Мы хотим: position_size * |entry - stop| / entry_price = free_equity * risk
    # => position_size = free_equity * risk * entry_price / |entry - stop|

    price_distance = abs(entry_price - stop_price)

    if price_distance <= 0 or np.isnan(price_distance):
        logger.warning("Invalid price distance: %.6f", price_distance)
        return 0.0

    # Размер позиции с учётом риска
    position_size = (free_equity * risk_per_trade * entry_price) / price_distance

    # Ограничиваем: стоимость позиции не может превышать equity * leverage
    max_notional = free_equity * leverage
    max_position_size = max_notional / entry_price

    position_size = min(position_size, max_position_size)

    return max(0.0, position_size)


# ---------------------------------------------------------------------------
# Корреляционный фильтр
# ---------------------------------------------------------------------------


def compute_rolling_correlation(
    returns_asset: pd.Series,
    returns_btc: pd.Series,
    window: int = 60,
) -> pd.Series:
    """
    Вычислить rolling корреляцию доходностей актива с BTC.

    ВАЖНО: используем rolling window (60 дней), не всю историю.
    Для безопасности используем min_periods=window, чтобы не получить
    корреляцию из-за малого количества наблюдений.

    Параметры
    ---------
    returns_asset : pd.Series — доходности актива (close.pct_change())
    returns_btc : pd.Series — доходности BTC
    window : int — размер rolling окна (дни)

    Возвращает
    ----------
    pd.Series — rolling корреляция
    """
    # Выравниваем индексы
    merged = pd.merge_asof(
        returns_asset.to_frame("asset"),
        returns_btc.to_frame("btc"),
        left_index=True,
        right_index=True,
        direction="backward",
    )

    if merged.shape[1] < 2:
        return pd.Series(np.nan, index=returns_asset.index)

    corr = merged["asset"].rolling(window=window, min_periods=window).corr(
        merged["btc"]
    )

    return corr


def check_correlation_filter(
    symbol: str,
    open_positions: Set[str],
    corr_values: pd.Series,
    threshold: float = 0.8,
    btc_symbol: str = "BTC/USDT",
) -> bool:
    """
    Проверить, не превышает ли корреляция актива к BTC порог.

    Возвращает
    ----------
    True — можно открывать позицию (корреляция допустима)
    False — нельзя (корреляция слишком высокая)
    """
    if symbol == btc_symbol:
        # BTC — всегда можно (референс)
        return True

    if corr_values is not None and not corr_values.empty:
        corr = corr_values.get(symbol, np.nan)
        if not np.isnan(corr) and corr > threshold:
            logger.info("Correlation filter: %s corr=%.3f > %.1f — skipped",
                        symbol, corr, threshold)
            return False

    return True


# ---------------------------------------------------------------------------
# Лимиты позиций
# ---------------------------------------------------------------------------


def check_position_limits(
    open_count: int,
    max_positions: int,
) -> bool:
    """
    Проверить, можно ли открыть ещё одну позицию.

    Параметры
    ---------
    open_count : int — текущее количество открытых позиций
    max_positions : int — максимум одновременных позиций (M5)

    Возвращает
    ----------
    True — можно, False — лимит исчерпан
    """
    if open_count >= max_positions:
        logger.debug("Position limit reached: %d/%d", open_count, max_positions)
        return False
    return True


# ---------------------------------------------------------------------------
# Проверка ликвидации (для futures)
# ---------------------------------------------------------------------------


def validate_liquidation(
    entry_price: float,
    stop_price: float,
    estimated_liquidation_price: float,
    buffer_multiplier: float = 2.0,
    leverage: int = 1,
) -> bool:
    """
    Проверить, что расстояние до ликвидации > buffer * расстояние до стопа (M5).

    Для long-позиции: liquidation_price < stop_price (если stop выше ликвидации — хорошо).
    Если плечо высокое, нужно убедиться что ликвидация далеко.

    Параметры
    ---------
    entry_price : float
    stop_price : float
    estimated_liquidation_price : float — приблизительная цена ликвидации
    buffer_multiplier : float — множитель буфера (по умолчанию 2x)
    leverage : int — плечо

    Возвращает
    ----------
    True — безопасно, False — риск ликвидации слишком высок
    """
    if leverage <= 1:
        # Spot или без плеча — не может быть ликвидации
        return True

    # Расстояние до стопа
    stop_distance = abs(entry_price - stop_price)

    # Расстояние до ликвидации
    liq_distance = abs(entry_price - estimated_liquidation_price)

    if liq_distance <= 0 or stop_distance <= 0:
        return False

    # Ликвидация должна быть дальше чем buffer * стоп
    safe = liq_distance >= buffer_multiplier * stop_distance

    if not safe:
        logger.warning(
            "Liquidation risk: entry=%.4f, stop=%.4f, liq=%.4f, "
            "stop_dist=%.4f, liq_dist=%.4f, buffer=%.1fx",
            entry_price, stop_price, estimated_liquidation_price,
            stop_distance, liq_distance, buffer_multiplier,
        )

    return safe


def estimate_liquidation_price(
    entry_price: float,
    leverage: int,
    is_long: bool = True,
    maintenance_margin_pct: float = 0.005,  # 0.5% для Binance
) -> float:
    """
    Приблизительно оценить цену ликвидации для фьючерсного контракта.

    Формула (упрощённая для long):
    liq_price = entry * (1 - 1/leverage + maintenance_margin)

    Параметры
    ---------
    entry_price : float
    leverage : int
    is_long : bool
    maintenance_margin_pct : float

    Возвращает
    ----------
    float — приблизительная цена ликвидации
    """
    margin = 1.0 / leverage
    mm = maintenance_margin_pct

    if is_long:
        liq = entry_price * (1.0 - margin + mm)
    else:
        liq = entry_price * (1.0 + margin - mm)

    return liq


# ---------------------------------------------------------------------------
# Класс RiskManager
# ---------------------------------------------------------------------------


class RiskManager:
    """
    Менеджер риска: объединяет все проверки и расчёты.
    """

    def __init__(self, config: StrategyConfig):
        self.cfg = config
        self.open_symbols: Set[str] = set()
        self.corr_cache: Dict[str, pd.Series] = {}

    def calculate_size(
        self,
        equity: float,
        entry_price: float,
        stop_price: float,
    ) -> float:
        """Вычислить размер позиции."""
        return calculate_position_size(
            equity, entry_price, stop_price,
            self.cfg.risk_per_trade,
            self.cfg.leverage,
        )

    def can_open_position(
        self,
        symbol: str,
        correlation_ok: bool = True,
    ) -> bool:
        """
        Проверить все условия для открытия позиции.

        Возвращает
        ----------
        True — все проверки пройдены
        """
        if not check_position_limits(len(self.open_symbols), self.cfg.max_positions):
            return False

        if not correlation_ok:
            logger.info("Correlation filter blocked position for %s", symbol)
            return False

        return True

    def add_position(self, symbol: str) -> None:
        self.open_symbols.add(symbol)

    def remove_position(self, symbol: str) -> None:
        self.open_symbols.discard(symbol)

    def liquidation_safe(self, entry: float, stop: float, estimate_liq: float) -> bool:
        """Проверить безопасность от ликвидации."""
        return validate_liquidation(
            entry, stop, estimate_liq,
            self.cfg.liquidation_buffer,
            self.cfg.leverage,
        )
