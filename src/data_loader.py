"""
Загрузка и кэширование исторических данных.

Источники: Binance (spot / futures perpetual) через ccxt.
Формат хранения: Apache Parquet для скорости и экономии места.
"""

from __future__ import annotations

import asyncio
import glob
import logging
import os
import pathlib
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set, Tuple

import ccxt.async_support as ccxt_async
import numpy as np
import pandas as pd

from src.config import Config

logger = logging.getLogger(__name__)

# Символы, которые нужно исключить (стейблкоины и т.п.)
ALL_SYMBOLS_TF: Dict[str, str] = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m",
    "30m": "30m", "1h": "1h", "2h": "2h", "4h": "4h",
    "6h": "6h", "8h": "8h", "12h": "12h", "1d": "1d",
    "1w": "1w", "1M": "1M",
}

# Для свинг-пивотов на недельном ТФ — backtrader использует daily candles,
# поэтому 1w конвертируется в '1w' для ccxt
WEEKLY_MAP = {"1w": "1w"}


# Топ-монеты для fallback, если API недоступен
_FALLBACK_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "MATIC/USDT",
    "DOT/USDT", "UNI/USDT", "ATOM/USDT", "LTC/USDT", "ETC/USDT",
    "FIL/USDT", "APT/USDT", "ARBN/USDT", "NEAR/USDT", "ALGO/USDT",
    "FTM/USDT", "SAND/USDT", "MANA/USDT", "AXS/USDT", "THETA/USDT",
    "ICP/USDT", "VET/USDT", "HBAR/USDT", "EGLD/USDT", "XTZ/USDT",
]


def _parse_tz_datetime(dt_str: str) -> datetime:
    """Парсить timestamp в datetime с UTC timezone."""
    dt = datetime.fromtimestamp(dt_str / 1000.0, tz=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# Топ-монеты
# ---------------------------------------------------------------------------


async def _get_top_symbols(
    exchange: ccxt_async.binance,
    top_n: int,
    market_type: str,
    exclude: List[str],
    quote: str = "USDT",
) -> List[str]:
    """
    Получить топ-N монет по объёму за 24ч, исключая стейблкоины.

    Параметры
    ---------
    exchange : ccxt exchange instance
    top_n : int — сколько символов вернуть
    market_type : 'spot' или 'futures'
    exclude : List[str] — префиксы/суффиксы для исключения
    quote : str — базовая валюта (по умолчанию USDT)

    Возвращает
    ----------
    List[str] — список символов вида 'BTC/USDT', 'ETH/USDT', ...
    """
    try:
        await exchange.load_markets()
    except Exception as e:
        logger.error("Failed to load markets: %s", e)
        return []

    exclude_upper = {e.upper() for e in exclude}
    exclude_upper.add(quote.upper())

    candidates: List[dict] = []

    # Загружаем тикеры для объёмов
    try:
        tickers = await exchange.fetch_tickers()
    except Exception as e:
        logger.warning("Failed to fetch tickers: %s — using fallback symbols", e)
        tickers = {}

    # Нормализуем ключи тикеров: futures формат 'BTC/USDT:USDT' → 'BTC/USDT'
    normalized_tickers: Dict[str, dict] = {}
    for sym, data in tickers.items():
        # Убираем ':USDT' или ':BTC' и т.д. из конца символа
        normalized = sym.split(":")[0] if ":" in sym else sym
        normalized_tickers[normalized] = data
    
    # Для futures также добавляем оригинальные ключи (futures format)
    # чтобы можно было сопоставить с markets
    for sym in list(tickers.keys()):
        if ":" not in sym:
            # Это спотовый символ, добавляем как есть
            normalized_tickers[sym] = tickers[sym]
    
    tickers = normalized_tickers

    # Проверка: если tickers пустой для futures — используем fallback
    if not tickers and market_type == "futures":
        logger.warning("No tickers returned for futures — using fallback symbols")
        fallback_symbols = _FALLBACK_SYMBOLS[:top_n]
        logger.info("Using fallback: %d symbols", len(fallback_symbols))
        return fallback_symbols

    # Если tickers есть, но объёмы нулевые — это тоже плохо
    if tickers and market_type == "futures":
        non_zero_count = sum(1 for t in tickers.values() if (t.get("quoteVolume", 0) or 0) > 1_000_000)
        if non_zero_count == 0:
            logger.warning("All futures tickers have zero volume — using fallback")
            fallback_symbols = _FALLBACK_SYMBOLS[:top_n]
            logger.info("Using fallback: %d symbols", len(fallback_symbols))
            return fallback_symbols

    # Проверка: если market_type == futures, но в tickers нет swap-символов — fallback
    if market_type == "futures" and tickers:
        swap_tickers = 0
        for sym, data in tickers.items():
            # Проверяем формат символа — futures имеет ':USDT' или похожий суффикс
            if ":" in sym or sym.endswith(":USDT"):
                swap_tickers += 1
        
        if swap_tickers == 0:
            logger.warning("No futures tickers found (all appear to be spot) — using fallback")
            fallback_symbols = _FALLBACK_SYMBOLS[:top_n]
            logger.info("Using fallback: %d symbols", len(fallback_symbols))
            return fallback_symbols

    for sym, m in exchange.markets.items():
        # Фильтр по типу рынка
        if market_type == "futures":
            if not (m.get("swap") and m.get("linear")):
                continue
        else:
            if not m.get("spot"):
                continue

        base = m.get("base", "").upper()
        quote_curr = m.get("quote", "").upper()

        if quote_curr != quote:
            continue

        # Исключаем стейблкоины и исключённые префиксы
        if any(ex in base for ex in exclude_upper):
            continue

        # Объём
        volume_24h = 0
        if sym in tickers:
            volume_24h = tickers[sym].get("quoteVolume", 0) or 0

        if volume_24h < 1_000_000:  # минимум 1M USD за 24ч
            continue

        candidates.append({"symbol": sym, "volume": volume_24h, "base": base})

    logger.info("Found %d candidates from %d markets after filtering",
                len(candidates), len(exchange.markets))

    # Сортировка по объёму и выбор топ-N
    candidates.sort(key=lambda x: x["volume"], reverse=True)
    
    # Логгируем отброшенные символы
    excluded_count = 0
    for c in candidates[top_n:]:
        logger.warning("Excluded %s from top-%d: volume=%.2f (too low or not enough slots)",
                       c["symbol"], top_n, c["volume"])
    
    selected = candidates[:top_n]

    symbols = [c["symbol"] for c in selected]
    logger.info("Selected %d symbols from %d candidates (top-%d by volume)",
                len(symbols), len(candidates), top_n)
    return symbols


# ---------------------------------------------------------------------------
# Загрузка candle data
# ---------------------------------------------------------------------------


def _ohlcv_to_df(
    ohlcv: List[list],
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:
    """
    Преобразовать сырой OHLCV от ccxt в DataFrame.

    Параметры
    ----------
    ohlcv : List[list]
        [[timestamp, open, high, low, close, volume], ...]
    symbol : str
        'BTC/USDT'
    timeframe : str
        '1h', '4h', etc.

    Возвращает
    ----------
    pd.DataFrame с индексом DatetimeIndex и колонками O, H, L, C, V.
    """
    if not ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["open", "high", "low", "close"], inplace=True)
    df.sort_index(inplace=True)
    df.symbol = symbol  # метка
    return df


async def _fetch_ohlcv_paginated(
    exchange: ccxt_async.binance,
    symbol: str,
    timeframe: str,
    since_ms: int,
    duration_ms: Optional[int] = None,
    max_retries: int = 3,
) -> List[list]:
    """
    Загрузить все свечи с пагинацией (ccxt возвращает по 1000 за запрос).

    Параметры
    ---------
    exchange : ccxt exchange
    symbol : str
    timeframe : str
    since_ms : int — timestamp начала периода (ms)
    duration_ms : int — продолжительность загрузки (ms). Если None — берём сколько возможно до now.
    max_retries : int

    Возвращает
    ----------
    List[list] — все свечи [[ts, o, h, l, c, v], ...]
    """
    all_ohlcv: List[list] = []
    current_since = since_ms

    # Для futures limit может быть до 1000, для spot тоже ~1000
    max_per_request = 1000

    retries = 0
    while True:
        if retries > max_retries:
            logger.error("Max retries reached for %s", symbol)
            break

        try:
            if duration_ms is not None:
                # Ограничиваем конечную дату
                limit_since = max(current_since, since_ms)
                end_ms = since_ms + duration_ms
                if limit_since >= end_ms:
                    break
            else:
                limit_since = current_since

            raw = await exchange.fetch_ohlcv(
                symbol, timeframe,
                since=limit_since,
                limit=max_per_request,
            )

            if not raw:
                break

            all_ohlcv.extend(raw)
            last_ts = raw[-1][0]
            # Сдвигаемся дальше
            current_since = last_ts + 1
            retries = 0

            # Лог для отслеживания прогресса (раз в ~10000 свечей)
            if len(all_ohlcv) % 5000 == 0:
                logger.debug(
                    "  %s: downloaded %d candles (last ts=%s)",
                    symbol, len(all_ohlcv),
                    datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).isoformat(),
                )

        except ccxt_async.errors.RateLimit as e:
            retries += 1
            wait_time = 2 ** retries
            logger.warning("Rate limit for %s: %s. Waiting %ds", symbol, e, wait_time)
            await asyncio.sleep(wait_time)
        except Exception as e:
            retries += 1
            logger.error("Error fetching %s: %s", symbol, e)
            await asyncio.sleep(2)
            # Продолжаем с того же места
            continue

        # Если не получили новых данных — конец
        if not raw or len(raw) < max_per_request:
            break

    return all_ohlcv


async def fetch_all_symbols(
    symbols: List[str],
    timeframe: str,
    since_days: int,
    market_type: str,
    min_required_rows: int = 100,
) -> Dict[str, pd.DataFrame]:
    """
    Загрузить OHLCV для списка символов.

    Параметры
    ---------
    symbols : List[str]
    timeframe : str
    since_days : int — сколько дней назад загружать
    market_type : 'spot' или 'futures'
    min_required_rows : int — минимальное количество свечей для анализа

    Возвращает
    ----------
    Dict[str, pd.DataFrame] — символ -> DataFrame
    """
    since_ts = int((datetime.now(tz=timezone.utc) - timedelta(days=since_days * 365)).timestamp() * 1000)
    exchange = getattr(ccxt_async, "binance")({
        "enableRateLimit": True,
    })

    # Для futures указываем тип рынка
    if market_type == "futures":
        exchange.options["defaultType"] = "swap"

    data: Dict[str, pd.DataFrame] = {}

    try:
        await exchange.load_markets()
    except Exception as e:
        logger.error("Failed to load markets: %s", e)
        return data

    total = len(symbols)
    for i, symbol in enumerate(symbols):
        logger.info("[%d/%d] Fetching %s (%s)", i + 1, total, symbol, timeframe)
        try:
            ohlcv = await _fetch_ohlcv_paginated(
                exchange, symbol, timeframe,
                since_ms=since_ts,
            )
            if not ohlcv:
                logger.warning("Skipped %s: no data returned from API", symbol)
                continue

            df = _ohlcv_to_df(ohlcv, symbol, timeframe)
            if df.empty:
                logger.warning("Skipped %s: empty DataFrame after processing", symbol)
                continue

            if len(df) < min_required_rows:
                logger.warning("Skipped %s: insufficient data. Got %d rows, required min %d",
                               symbol, len(df), min_required_rows)
                continue

            data[symbol] = df
            logger.info("  ✓ %s: %d candles (%s → %s)",
                        symbol, len(df),
                        df.index[0].isoformat(), df.index[-1].isoformat())

        except Exception as e:
            logger.error("Failed to fetch %s: %s", symbol, e)

    await exchange.close()
    return data


# ---------------------------------------------------------------------------
# Кэширование
# ---------------------------------------------------------------------------


def _cache_path(cache_dir: str, symbol: str, timeframe: str) -> pathlib.Path:
    """Сформировать путь к кэш-файлу для символа и таймфрейма."""
    # 'BTC/USDT' -> 'BTC_USDT'
    safe = symbol.replace("/", "_")
    return pathlib.Path(cache_dir) / f"{safe}_{timeframe}.parquet"


def _is_cache_fresh(cache_file: pathlib.Path, max_age_days: int) -> bool:
    """Проверить, не истёк ли кэш."""
    if not cache_file.exists():
        return False
    age = time.time() - cache_file.stat().st_mtime
    max_age = max_age_days * 86400  # секунд
    return age < max_age


def save_cache(data: Dict[str, pd.DataFrame], cache_dir: str, max_age_days: int) -> None:
    """Сохранить данные в parquet-файлы."""
    pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)
    saved = 0
    for symbol, df in data.items():
        path = _cache_path(cache_dir, symbol, df.attrs.get("timeframe", "1h"))
        df.to_parquet(path, engine="pyarrow")
        saved += 1
    logger.info("Cached %d symbol datasets to %s/", saved, cache_dir)


def load_cache(
    cache_dir: str,
    symbols: List[str],
    timeframe: str,
) -> Dict[str, pd.DataFrame]:
    """
    Загрузить данные из кэша.

    Возвращает
    ----------
    Dict[str, pd.DataFrame] — только те символы, что есть в cache и свежи.
    """
    data: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = _cache_path(cache_dir, symbol, timeframe)
        if path.exists():
            try:
                df = pd.read_parquet(path)
                df.attrs["timeframe"] = timeframe
                df.attrs["cached"] = True
                data[symbol] = df
            except Exception as e:
                logger.warning("Failed to read cache %s: %s", path, e)
        else:
            logger.debug("Cache not found: %s", path)
    logger.info("Loaded %d datasets from cache", len(data))
    return data


# ---------------------------------------------------------------------------
# Главный метод: загрузка + кэширование
# ---------------------------------------------------------------------------


class DataLoader:
    """
    Загрузка и кэширование исторических данных с Binance.

    Использует ccxt async API для параллельной загрузки.
    Данные сохраняются в parquet для ускорения последующих прогонов.
    """

    def __init__(self, config: Config):
        self.cfg = config
        self.cache_dir = pathlib.Path(config.data.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_symbols(self) -> List[str]:
        """Получить список символов для торговли."""
        # Сначала пробуем из кэша, если он есть
        cached_files = list(self.cache_dir.glob("*.parquet"))
        if cached_files and not self.cfg.download_only:
            # Извлекаем символы из имён файлов: BTC_USDT_1h → BTC/USDT
            existing_symbols: List[str] = []
            for f in cached_files:
                parts = f.stem.split("_")
                # Берём все части, кроме последней (timeframe)
                base_parts = parts[:-1]
                symbol = "/".join(base_parts)
                existing_symbols.append(symbol)
            logger.info("%d cached datasets found", len(existing_symbols))
            return existing_symbols

        # Запускаем async функцию для получения топ-символов
        return self._run_sync(self._get_top_symbols_async())

    @staticmethod
    def _run_sync(coro):
        """Запустить async-корутину из синхронного контекста."""
        return asyncio.run(coro)

    async def _get_top_symbols_async(self) -> List[str]:
        """Async-версия получения топ-символов."""
        max_retries = 3
        for attempt in range(max_retries):
            exchange = ccxt_async.binance({
                "enableRateLimit": True,
            })
            # Для futures обязательно указываем тип рынка
            if self.cfg.data.market_type == "futures":
                exchange.options["defaultType"] = "swap"
            try:
                await exchange.load_markets()
                return await _get_top_symbols(
                    exchange,
                    top_n=self.cfg.data.top_coins,
                    market_type=self.cfg.data.market_type,
                    exclude=self.cfg.data.exclude,
                    quote=self.cfg.data.quote_currency,
                )
            except Exception as e:
                logger.warning(
                    "Attempt %d/%d failed: %s", attempt + 1, max_retries, e
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

        # Fallback: если API не доступен — используем хардкод
        logger.warning("All attempts failed, using fallback symbols")
        top = self.cfg.data.top_coins
        symbols = _FALLBACK_SYMBOLS[:top]
        logger.info("Using fallback: %d symbols", len(symbols))
        return symbols

    def _filter_cached(
        self, symbols: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Разделить символы на cached и need_download.

        Возвращает
        ----------
        (to_download, already_cached)
        """
        to_download: List[str] = []
        already_cached: List[str] = []

        for symbol in symbols:
            path = _cache_path(str(self.cache_dir), symbol, self.cfg.data.timeframe)
            if path.exists() and _is_cache_fresh(path, self.cfg.data.cache_max_age_days):
                already_cached.append(symbol)
            else:
                to_download.append(symbol)

        logger.info("Already cached: %d, need download: %d", len(already_cached), len(to_download))
        return to_download, already_cached

    def load(self) -> Dict[str, pd.DataFrame]:
        """
        Загрузить OHLCV данные для всех символов.

        Алгоритм:
        1. Получить список топ-символов
        2. Проверить кэш
        3. Скачать недостающее
        4. Слить и сохранить кэш

        Возвращает
        ----------
        Dict[str, pd.DataFrame] — символ -> OHLCV DataFrame
        """
        symbols = self._get_symbols()
        if not symbols:
            logger.warning("No symbols found")
            return {}

        # Фильтрация по кэшу
        to_download, already_cached = self._filter_cached(symbols)

        # Загрузка из кэша
        data: Dict[str, pd.DataFrame] = {}
        if already_cached:
            data = load_cache(str(self.cache_dir), already_cached, self.cfg.data.timeframe)

        # Скачивание нового
        if to_download and not self.cfg.skip_download:
            logger.info("Downloading %d datasets...", len(to_download))
            fetched = self._run_sync(
                fetch_all_symbols(to_download, self.cfg.data.timeframe,
                                  self.cfg.data.data_years, self.cfg.data.market_type,
                                  min_required_rows=self.cfg.data.min_required_rows)
            )
            data.update(fetched)

        if not data:
            logger.error("No data loaded. Aborting.")
            return {}

        # Сохраняем кэш (один раз)
        save_cache(data, str(self.cache_dir), self.cfg.data.cache_max_age_days)

        logger.info("Total loaded: %d symbols", len(data))
        return data

    def load_for_backtest(self) -> Dict[str, pd.DataFrame]:
        """
        Загрузить данные, готовые для бэктеста.

        Делает дополнительную обработку:
        - Убедимся что есть нужные колонки
        - Сортировка по индексу
        - Добавляем attrs (timeframe, symbol)
        - Сохраняем в кэш
        """
        data = self.load()
        for symbol, df in data.items():
            df = df[["open", "high", "low", "close", "volume"]].copy()
            df.sort_index(inplace=True)
            df.dropna(subset=["open", "high", "low", "close"], inplace=True)
            df.attrs["timeframe"] = self.cfg.data.timeframe
            df.attrs["symbol"] = symbol

        # Повторно кэшируем уже отфильтрованные данные
        save_cache(data, str(self.cache_dir), self.cfg.data.cache_max_age_days)

        return data


# ---------------------------------------------------------------------------
# Загрузка данных для нескольких таймфреймов (тренд на 1d/1w)
# ---------------------------------------------------------------------------


class MultiTimeframeDataLoader:
    """
    Загрузка данных для нескольких таймфреймов.

    Нужен для:
    - 1h / 4h — основная торговля
    - 1d / 1w — определение тренда
    """

    def __init__(self, config: Config):
        self.cfg = config

    def load(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Загрузить данные для всех нужных ТФ.

        Возвращает
        ----------
        Dict[timeframe, Dict[symbol, DataFrame]]
        """
        timeframes = set()
        timeframes.add(self.cfg.data.timeframe)
        timeframes.update(self.cfg.strategy.trend_timeframes)

        result: Dict[str, Dict[str, pd.DataFrame]] = {}
        for tf in timeframes:
            loader = DataLoader(self.cfg)
            # Временно переопределяем таймфрейм
            orig_tf = loader.cfg.data.timeframe
            loader.cfg.data.timeframe = tf
            data = loader.load_for_backtest()
            loader.cfg.data.timeframe = orig_tf
            result[tf] = data

        logger.info("MultiTimeframe loaded: %d timeframes", len(result))
        return result
