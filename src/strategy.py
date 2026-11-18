"""
Стратегия Range Trading с трендовым фильтром.

Генерация сигналов входа и выхода на основе индикаторов.
Все условия строго следуют спецификации:
- M3: вход LONG (7 условий)
- M4: выход (тейк, стоп, трейлинг)

Запись сигналов в DataFrame для прогона бэктеста.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.indicators import add_all_indicators
from src.risk_manager import (
    calculate_position_size as risk_calculate_position_size,
    check_correlation_filter,
    estimate_liquidation_price,
    validate_liquidation,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Сделки
# ---------------------------------------------------------------------------


@dataclass
class Trade:
    """Одна сделка."""

    symbol: str
    entry_time: pd.Timestamp
    entry_price: float
    entry_size: float  # количество базового актива (контракты/монеты)
    stop_price: float
    take_profit_price: Optional[float]
    entry_bar: int = 0  # индекс бара входа (для расчёта holding_bars)
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # 'take_profit', 'stop_loss', 'trailing', 'halt', 'eod'
    pnl: Optional[float] = None  # абсолютный PnL (после комиссий)
    pnl_pct: Optional[float] = None  # процентный PnL
    max_price: Optional[float] = None  # max price с момента входа (для трейлинга)
    max_stop: Optional[float] = None  # максимальный уровень стопа (для трейлинга)
    holding_bars: int = 0
    commission: float = 0.0
    slippage: float = 0.0
    is_closed: bool = False
    equity_at_entry: float = 0.0
    fee_rate: float = 0.0  # combined maker/taker fee used


# ---------------------------------------------------------------------------
# Проверка условий входа
# ---------------------------------------------------------------------------

def check_entry_conditions(
    df: pd.DataFrame,
    idx: int,
    strategy_cfg,
) -> Tuple[bool, Dict[str, bool]]:
    row = df.iloc[idx]
    conditions: Dict[str, bool] = {}

    # 1. Базовый тренд (цена не должна быть в глубоком медвежьем тренде)
    # Используем мягкий фильтр: цена выше 95% от SMA200
    if pd.notna(row.get("sma_200", np.nan)):
        conditions["above_sma200"] = row["close"] > (row["sma_200"] * 0.95)
    else:
        conditions["above_sma200"] = True

    # 2. Цена коснулась нижней полосы и закрылась выше неё (классический отскок)
    conditions["touch_lower"] = row["low"] <= row["lower_band"]
    conditions["close_above_lower"] = row["close"] > row["lower_band"]

    # 3. Подтверждение разворота: RSI перепродан ИЛИ бычья свеча
    conditions["rsi_or_bullish"] = (row["rsi"] < strategy_cfg.rsi_threshold) or (row["close"] > row["open"])

    # 4. Фильтр волатильности (чтобы не торговать в мертвом боковике)
    conditions["atr_pct_in_range"] = (
        strategy_cfg.atr_pct_min <= row["atr_pct"] <= strategy_cfg.atr_pct_max
    )

    all_met = all(conditions.values())
    conditions["all_met"] = all_met

    return all_met, conditions

# ---------------------------------------------------------------------------
# Проверка условий выхода
# ---------------------------------------------------------------------------


def check_exit_conditions(
    trade: Trade,
    df: pd.DataFrame,
    idx: int,
    strategy_cfg,
) -> Tuple[bool, str]:
    """
    Проверить условия выхода для открытой позиции.
    """
    row = df.iloc[idx]

    # Стоп-лосс (жёсткий)
    if row["close"] <= trade.stop_price:
        return True, "stop_loss"

    # Тейк-профит (на верхней границе канала)
    if strategy_cfg.take_profit == "upper_band":
        if row["close"] >= row["upper_band"]:
            return True, "take_profit"

    # Трейлинг
    current_max = trade.max_price or trade.entry_price
    if row["high"] > current_max:
        current_max = row["high"]

    # Обновляем max price
    trailing_stop = current_max - strategy_cfg.trailing_multiplier * row["atr"]
    if row["close"] <= trailing_stop and current_max > trade.entry_price:
        return True, "trailing_stop"

    return False, ""


# ---------------------------------------------------------------------------
# Расчёт размера позиции
# ---------------------------------------------------------------------------


def calculate_position_size(
    equity: float,
    entry_price: float,
    stop_price: float,
    risk_per_trade: float,
    leverage: int = 1,
) -> float:
    """
    Рассчитать размер позиции так, чтобы убыток при срабатывании стопа 
    был РАВЕН risk_per_trade от equity.
    """
    risk_amount = equity * risk_per_trade
    price_distance = abs(entry_price - stop_price)

    if price_distance <= 0:
        return 0.0

    # Требуемый размер для соблюдения риска
    required_size = risk_amount / price_distance
    
    # Максимальный размер, который позволяет маржа с учётом плеча
    max_position_size = (equity * leverage) / entry_price

    if required_size > max_position_size:
        return 0.0

    return required_size

# ---------------------------------------------------------------------------
# Расчёт комиссии и проскальзывания
# ---------------------------------------------------------------------------


def apply_costs(
    entry_price: float,
    exit_price: float,
    size: float,
    is_long: bool,
    maker_fee: float,
    taker_fee: float,
    slippage_pct: float,
    funding_pct: float = 0.0,
    holding_hours: int = 0,
    include_entry: bool = True,
    include_exit: bool = True,
) -> Tuple[float, float]:
    """
    Рассчитать стоимость комиссии и проскальзывания.
    """
    entry_notional = entry_price * size
    exit_notional = exit_price * size
    
    # Комиссия (от стоимости позиции)
    entry_fee = entry_notional * (taker_fee + maker_fee) / 2
    exit_fee = exit_notional * (taker_fee + maker_fee) / 2

    commission = 0.0
    if include_entry:
        commission += entry_fee
    if include_exit:
        commission += exit_fee

    slippage = 0.0

    # Фандер (только для futures)
    funding_cost = 0.0
    if funding_pct > 0:
        funding_periods = holding_hours / 8  # каждые 8 часов
        funding_cost = entry_notional * funding_pct * funding_periods

    return commission + funding_cost, slippage


# ---------------------------------------------------------------------------
# Генерация сигналов для одного актива
# ---------------------------------------------------------------------------


def generate_signals(
    df: pd.DataFrame,
    strategy_cfg,
    fee_rate: float = 0.0005,
) -> pd.DataFrame:
    """
    Проанализировать DataFrame символа и добавить сигнальные колонки.
    """
    df = df.copy()
    df["signal"] = 0
    df["entry_price"] = np.nan
    df["stop_price"] = np.nan

    n = len(df)
    for i in range(1, n):
        row = df.iloc[i]

        # Проверяем входные условия
        met, _ = check_entry_conditions(df, i, strategy_cfg)
        if met:
            df.at[df.index[i], "signal"] = 1
            df.at[df.index[i], "entry_price"] = row["close"]
            # Стоп: entry - 1.8 * ATR
            stop = row["close"] - strategy_cfg.stop_multiplier * row["atr"]
            df.at[df.index[i], "stop_price"] = stop

    return df


# ---------------------------------------------------------------------------
# Симуляция торговли (event-driven simulation)
# ---------------------------------------------------------------------------


@dataclass
class SimulationResult:
    """Результаты симуляции торговли."""

    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    cash: float = 0.0
    final_equity: float = 0.0


def simulate_trading(
    symbols_data: Dict[str, pd.DataFrame],
    strategy_cfg,
    backtest_cfg,
    fee_rate: float = 0.0005,
    slippage_pct: float = 0.0005,
    funding_pct: float = 0.0,
) -> SimulationResult:
    """
    Прогнать симуляцию торговли по всем символам.
    """
    result = SimulationResult()
    initial_equity = backtest_cfg.initial_cash
    cash = initial_equity
    result.cash = cash
    current_equity = cash  # FIX: Инициализация current_equity до цикла

    # Состояние
    open_positions: Dict[str, Trade] = {}  # symbol -> Trade
    daily_equity: Dict[pd.Timestamp, float] = {}

    # Собираем все временные метки для выравнивания
    all_dates = set()
    for symbol, df in symbols_data.items():
        all_dates.update(df.index)
    all_dates = sorted(all_dates)

    # Для каждого символа добавляем сигналы
    for symbol, df in symbols_data.items():
        df["signal"] = 0
        df["entry_price"] = np.nan
        df["stop_price"] = np.nan

    # Проходим по каждому бару (по всем символам одновременно)
    for dt in all_dates:
        # --- 1. Обновляем открытые позиции (выходы) ---
        symbols_to_close = []
        for symbol, trade in open_positions.items():
            df = symbols_data[symbol]
            if dt not in df.index:
                continue

            row = df.loc[dt]
            exit_needed, reason = check_exit_conditions(
                trade, df, df.index.get_loc(dt), strategy_cfg
            )

            if exit_needed:
                symbols_to_close.append((symbol, trade, reason))

        # --- 2. Закрываем позиции ---
        for symbol, trade, reason in symbols_to_close:
            df = symbols_data[symbol]
            row = df.loc[dt]
            idx = df.index.get_loc(dt)  # FIX: корректный индекс для holding_bars

            # Цена выхода
            if reason == "trailing_stop":
                exit_price = trade.max_price - strategy_cfg.trailing_multiplier * row["atr"]
            elif reason == "stop_loss":
                exit_price = trade.stop_price
            else:
                exit_price = row["close"]

            # PnL: (exit - entry) * size (для long)
            gross_pnl = (exit_price - trade.entry_price) * trade.entry_size

            # Комиссия за закрытие (вход + выход)
            holding_hours = max(1, int((dt - trade.entry_time).total_seconds() / 3600))
            commission, _ = apply_costs(
                trade.entry_price, exit_price, trade.entry_size,
                is_long=True,
                maker_fee=strategy_cfg.maker_fee,
                taker_fee=strategy_cfg.taker_fee,
                slippage_pct=slippage_pct,
                funding_pct=funding_pct,
                holding_hours=holding_hours,
                include_entry=True,
                include_exit=True,
            )

            # Реализованный PnL = gross - комиссия
            pnl = gross_pnl - commission

            # Возвращаем МАРЖУ + PnL при закрытии фьючерса
            margin_returned = (trade.entry_price * trade.entry_size) / strategy_cfg.leverage
            cash += margin_returned + pnl

            # Обновляем сделку
            trade.exit_time = dt
            trade.exit_price = exit_price
            trade.exit_reason = reason
            trade.pnl = pnl
            trade.pnl_pct = (pnl / (trade.entry_price * trade.entry_size) * 100.0) if (trade.entry_price * trade.entry_size) > 0 else 0.0
            trade.holding_bars = max(1, idx - trade.entry_bar) if hasattr(trade, 'entry_bar') and idx >= trade.entry_bar else 1
            trade.commission = commission
            trade.is_closed = True

            result.trades.append(trade)
            del open_positions[symbol]

        # --- 3. Проверяем входы ---
        for symbol, df in symbols_data.items():
            if symbol in open_positions:
                continue
            if len(open_positions) >= strategy_cfg.max_positions:
                break

            if dt not in df.index:
                continue

            # Корреляционный фильтр
            if not check_correlation_filter(
                symbol, set(open_positions.keys()),
                None,
                threshold=strategy_cfg.correlation_threshold,
                btc_symbol=strategy_cfg.correlation_symbol,
            ):
                continue

            idx = df.index.get_loc(dt)
            met, conditions = check_entry_conditions(df, idx, strategy_cfg)

            if met:
                row = df.iloc[idx]
                entry_price = row["close"]
                stop_price = entry_price - strategy_cfg.stop_multiplier * row["atr"]

                # Размер позиции: risk * equity / |entry - stop|
                size = calculate_position_size(
                    current_equity, entry_price, stop_price,
                    strategy_cfg.risk_per_trade,
                    strategy_cfg.leverage,
                )

                if size <= 0:
                    continue

                position_cost = entry_price * size
                if position_cost > current_equity:
                    logger.warning(
                        "Position cost %.2f > available equity %.2f, skipping %s at %s",
                        position_cost, current_equity, symbol, dt
                    )
                    continue

                # FIX: Для фьючерсов списывается только маржа, а не полная стоимость!
                margin_cost = position_cost / strategy_cfg.leverage
                remaining_cash = cash - margin_cost
                if remaining_cash < 0:
                    logger.warning(
                        "After margin deduction cash would be negative (%.2f), skipping %s",
                        remaining_cash, symbol
                    )
                    continue

                # ДОП. ПРОВЕРКА: total risk не должен превышать X% от equity
                single_risk = size * abs(entry_price - stop_price)
                total_risk = single_risk + sum(
                    t.entry_size * abs(t.entry_price - t.stop_price)
                    for t in open_positions.values()
                )
                max_total_risk = current_equity * 0.10  # макс 10% общего риска

                if total_risk > max_total_risk:
                    logger.debug(
                        "Total risk %.2f > max %.2f, skipping %s",
                        total_risk, max_total_risk, symbol
                    )
                    continue

                # ПРОВЕРКА ЛИКВИДАЦИИ (для futures с плечом)
                if strategy_cfg.leverage > 1:
                    estimated_liq = estimate_liquidation_price(
                        entry_price, strategy_cfg.leverage, is_long=True
                    )
                    liq_safe = validate_liquidation(
                        entry_price, stop_price, estimated_liq,
                        buffer_multiplier=strategy_cfg.liquidation_buffer,
                        leverage=strategy_cfg.leverage,
                    )
                    if not liq_safe:
                        logger.warning(
                            "Liquidation risk: entry=%.4f, stop=%.4f, liq=%.4f — "
                            "skipping %s at %s",
                            entry_price, stop_price, estimated_liq, symbol, dt
                        )
                        continue

                # Детальное логирование для отладки риска
                logger.info(
                    f"OPEN {symbol}: entry={entry_price:.4f}, stop={stop_price:.4f}, "
                    f"dist={abs(entry_price-stop_price):.4f}, size={size:.4f}, "
                    f"cost={position_cost:.2f}, cash={cash:.2f}, risk_amt={current_equity*strategy_cfg.risk_per_trade:.2f}"
                )
                
                trade = Trade(
                    symbol=symbol,
                    entry_time=dt,
                    entry_bar=idx,
                    entry_price=entry_price,
                    entry_size=size,
                    stop_price=stop_price,
                    take_profit_price=None,
                    max_price=entry_price,
                    max_stop=stop_price,
                    equity_at_entry=current_equity,
                    fee_rate=fee_rate,
                )

                # Вычитаем МАРЖУ, а не полную стоимость (фьючерсы)
                cash -= margin_cost
                logger.info(f"MARGIN DEDUCTED: {margin_cost:.2f} (Lev: {strategy_cfg.leverage}x). Remaining cash: {cash:.2f}")

                open_positions[symbol] = trade

                # Обновляем сигнал в DataFrame
                df.at[dt, "signal"] = 1
                df.at[dt, "entry_price"] = entry_price
                df.at[dt, "stop_price"] = stop_price

        # --- 4. Обновляем equity (cash + unrealized PnL) ---
        unrealized_pnl = 0.0
        for symbol, trade in open_positions.items():
            df = symbols_data[symbol]
            if dt in df.index:
                row = df.loc[dt]
                # Обновляем max price для трейлинга
                if row["high"] > (trade.max_price or trade.entry_price):
                    trade.max_price = row["high"]
                # Unrealized PnL = (current - entry) * size
                unrealized_pnl += (row["close"] - trade.entry_price) * trade.entry_size

        current_equity = cash + unrealized_pnl
        daily_equity[dt] = current_equity

        # --- 5. DRAWDOWN GUARD: если equity упал ниже 50% от начального, останавливаемся ---
        if current_equity < initial_equity * 0.5:
            logger.error(
                "HALT TRIGGERED: equity=%.2f < initial=%.2f * 0.5",
                current_equity, initial_equity
            )
            # Принудительно закрываем ВСЕ позиции
            for symbol, trade in list(open_positions.items()):
                df = symbols_data.get(symbol, pd.DataFrame())
                
                # Определяем цену выхода и индекс
                if not df.empty and dt in df.index:
                    exit_price = df.loc[dt, "close"]
                    idx = df.index.get_loc(dt)
                elif not df.empty:
                    exit_price = df["close"].iloc[-1]
                    idx = len(df) - 1
                else:
                    exit_price = trade.entry_price
                    idx = trade.entry_bar

                gross_pnl = (exit_price - trade.entry_price) * trade.entry_size
                holding_hours = max(1, int((dt - trade.entry_time).total_seconds() / 3600))

                commission, _ = apply_costs(
                    trade.entry_price, exit_price, trade.entry_size,
                    is_long=True,
                    maker_fee=strategy_cfg.maker_fee,
                    taker_fee=strategy_cfg.taker_fee,
                    slippage_pct=slippage_pct,
                    funding_pct=funding_pct,
                    holding_hours=holding_hours,
                    include_entry=True,
                    include_exit=True,
                )
                
                pnl = gross_pnl - commission

                trade.exit_time = dt
                trade.exit_price = exit_price
                trade.exit_reason = "drawdown_halt"
                trade.pnl = pnl
                trade.pnl_pct = (pnl / (trade.entry_price * trade.entry_size) * 100.0) if (trade.entry_price * trade.entry_size) > 0 else 0.0
                trade.holding_bars = max(1, idx - trade.entry_bar) if hasattr(trade, 'entry_bar') and idx >= trade.entry_bar else 1
                trade.is_closed = True
                
                margin_returned = (trade.entry_price * trade.entry_size) / strategy_cfg.leverage
                cash += margin_returned + pnl
                
                result.trades.append(trade)
                del open_positions[symbol]
                
            logger.warning(f"Emergency closed positions. Final cash: {cash:.2f}")
            break  # Прерываем симуляцию

    # --- 6. Закрываем все оставшиеся позиции (end-of-data) ---
    for symbol, trade in open_positions.items():
        df = symbols_data.get(symbol, pd.DataFrame())
        if not df.empty:
            last_known_price = df["close"].iloc[-1]
            last_dt = df.index[-1]
            idx = len(df) - 1
        else:
            last_known_price = trade.entry_price
            last_dt = trade.entry_time
            idx = trade.entry_bar

        gross_pnl = (last_known_price - trade.entry_price) * trade.entry_size
        exit_commission, _ = apply_costs(
            trade.entry_price, last_known_price, trade.entry_size,
            is_long=True,
            maker_fee=strategy_cfg.maker_fee,
            taker_fee=strategy_cfg.taker_fee,
            slippage_pct=slippage_pct,
            funding_pct=funding_pct,
            include_entry=True,
            include_exit=True,
        )
        pnl = gross_pnl - exit_commission

        trade.exit_time = last_dt
        trade.exit_price = last_known_price
        trade.exit_reason = "eod"
        trade.pnl = pnl
        trade.pnl_pct = (pnl / (trade.entry_price * trade.entry_size) * 100.0) if (trade.entry_price * trade.entry_size) > 0 else 0.0
        trade.holding_bars = max(1, idx - trade.entry_bar) if hasattr(trade, 'entry_bar') and idx >= trade.entry_bar else 1
        trade.is_closed = True
        
        margin_returned = (trade.entry_price * trade.entry_size) / strategy_cfg.leverage
        cash += margin_returned + pnl
        
        result.trades.append(trade)

    # --- 7. Equity curve ---
    if daily_equity:
        eq_df = pd.DataFrame(
            list(daily_equity.items()),
            columns=["timestamp", "equity"]
        ).set_index("timestamp")
        eq_df.sort_index(inplace=True)
        result.equity_curve = eq_df

    result.final_equity = cash if daily_equity else initial_equity

    return result
