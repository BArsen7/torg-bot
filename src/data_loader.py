"""
Загрузка и кэширование исторических данных.

Источник: Bybit API V5 (прямая асинхронная интеграция через httpx).
Формат хранения: Apache Parquet для скорости и экономии места.
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
import time
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd

from src.config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Настройки Bybit V5 API
# ---------------------------------------------------------------------------

# Базовый URL API V5 (без ключей — публичные эндпоинты)
BYBIT_BASE_URL = "https://api.bybit.com"

# Словарь маппинга таймфреймов проекта → значения interval Bybit V5.
# Bybit использует свои обозначения:
#   1m, 3m, 5m, 15m, 30m → одинаковые
#   1h → "60", 2h → "120", 4h → "240"
#   6h → "360", 1d → "D", 1w → "W", 1M → "M"
# 8h и 12h не поддерживаются напрямую → маппим на ближайшие (480 и 720 минут)
TIMEFRAME_TO_INTERVAL: Dict[str, str] = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "6h": "360",
    "8h": "480",   # Bybit не имеет 8h, используем 480 минут
    "12h": "720",  # Bybit не имеет 12h, используем 720 минут
    "1d": "D",
    "1w": "W",
    "1M": "M",
}

# Фallback-список топ-30 USDT-перпетуалов на случай недоступности API.
# Отсортированы по примерному объёму и ликвидности.
_FALLBACK_SYMBOLS: List[str] = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "TRX/USDT",
    "DOT/USDT", "UNI/USDT", "ATOM/USDT", "LTC/USDT", "ETC/USDT",
    "FIL/USDT", "APT/USDT", "ARB/USDT", "NEAR/USDT", "ALGO/USDT",
    "FTM/USDT", "SAND/USDT", "MANA/USDT", "AXS/USDT", "THETA/USDT",
    "ICP/USDT", "VET/USDT", "HBAR/USDT", "EGLD/USDT", "XTZ/USDT",
]


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def _ms_to_dt(ms: int) -> datetime:
    """Конвертировать Unix-таймстамп в миллисекундах в datetime (UTC)."""
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def _symbol_to_bybit(symbol: str) -> str:
    """
    Преобразовать символ проекта в формат Bybit API.

    Проект:  "BTC/USDT"  →  Bybit: "BTCUSDT"
    Проект:  "ETH/USDT"  →  Bybit: "ETHUSDT"
    """
    return symbol.replace("/", "")


def _bybit_to_symbol(symbol: str) -> str:
    """
    Преобразовать символ Bybit API в формат проекта.

    Bybit: "BTCUSDT"  →  проект: "BTC/USDT"
    """
    if symbol.endswith("USDT"):
        return symbol[:-4] + "/USDT"
    return symbol


# ---------------------------------------------------------------------------
# Декоратор retry с экспоненциальной задержкой (exponential backoff)
# ---------------------------------------------------------------------------


def _retry_with_backoff(
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
):
    """
    Декоратор для повторных попыток с экспоненциальной задержкой.

    Повторяет вызов при:
    - HTTP 429 (Rate Limit)
    - HTTP 5xx (Server Error)
    - Timeout / ConnectionError

    Задержка: base_delay * (2 ** attempt), но не больше max_delay.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error: Optional[Exception] = None

            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except httpx.RemoteProtocolError as e:
                    last_error = e
                    wait = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(
                        "Protocol error in %s (attempt %d/%d): %s. "
                        "Retrying in %.1fs",
                        func.__name__, attempt + 1, max_retries, e, wait,
                    )
                    await asyncio.sleep(wait)
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        wait = min(base_delay * (2 ** attempt), max_delay)
                        # Попробуем прочитать Retry-After из заголовка
                        retry_after = e.response.headers.get("retry-after")
                        if retry_after:
                            try:
                                wait = float(retry_after)
                            except (ValueError, TypeError):
                                pass
                        logger.warning(
                            "Rate limited (429) in %s (attempt %d/%d). "
                            "Waiting %.1fs",
                            func.__name__, attempt + 1, max_retries, wait,
                        )
                        await asyncio.sleep(wait)
                    elif e.response.status_code >= 500:
                        wait = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(
                            "Server error %d in %s (attempt %d/%d): %s. "
                            "Retrying in %.1fs",
                            e.response.status_code,
                            func.__name__, attempt + 1, max_retries,
                            e, wait,
                        )
                        await asyncio.sleep(wait)
                    else:
                        # Код 4xx — это не ошибка сети, пробрасываем сразу
                        raise
                except (httpx.TimeoutException, httpx.ConnectError) as e:
                    last_error = e
                    wait = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(
                        "Network error in %s (attempt %d/%d): %s. "
                        "Retrying in %.1fs",
                        func.__name__, attempt + 1, max_retries, e, wait,
                    )
                    await asyncio.sleep(wait)

            logger.error(
                "All %d retries exhausted for %s. Last error: %s",
                max_retries, func.__name__, last_error,
            )
            raise last_error  # type: ignore[misc]

        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# 1. Получение списка топ-монет
# ---------------------------------------------------------------------------


@_retry_with_backoff(max_retries=3, base_delay=1.0)
async def _get_top_symbols(
    client: Optional[httpx.AsyncClient],
    top_n: int = 30,
    quote: str = "USDT",
) -> List[str]:
    """
    Получить топ-N монет по объёму за 24ч через Bybit V5.

    Если client=None, создаётся автономно и закрывается внутри.
    """
    own_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=30.0)
        own_client = True

    try:
        return await _get_top_symbols_impl(client, top_n, quote)
    finally:
        if own_client:
            await client.aclose()


async def _get_top_symbols_impl(
    client: httpx.AsyncClient,
    top_n: int = 30,
    quote: str = "USDT",
) -> List[str]:
    """
    Реальная логика получения топ-N монет через Bybit V5.

    Эндпоинт: GET /v5/market/tickers с параметром category=linear.
    Фильтр: только USDT-перпетуалы с turnover24h > 1M USD.
    """
    url = f"{BYBIT_BASE_URL}/v5/market/tickers"
    params = {"category": "linear"}

    response = await client.get(url, params=params, timeout=30.0)
    response.raise_for_status()
    data = response.json()

    # Проверяем статус API
    if data.get("retCode") != 0:
        logger.error(
            "Bybit API error retCode=%d retMsg=%s — fallback",
            data["retCode"], data.get("retMsg"),
        )
        return _fallback_symbols(top_n)

    tickers_list = data.get("result", {}).get("list", [])
    if not tickers_list:
        logger.warning("No tickers from Bybit — using fallback")
        return _fallback_symbols(top_n)

    # Фильтрация: USDT-перпетуалы с turnover24h > 1M USD
    candidates: List[Dict[str, Any]] = []
    for t in tickers_list:
        symbol = t.get("symbol", "")
        if not symbol.endswith(quote):
            continue

        base = symbol.replace(quote, "")
        if not base:
            continue

        turnover = float(t.get("turnover24h", 0) or 0)
        if turnover < 1_000_000:
            continue

        candidates.append({"symbol": symbol, "volume": turnover})

    logger.info("Found %d candidates out of %d Bybit tickers",
                len(candidates), len(tickers_list))

    # Сортировка по убыванию объёма
    candidates.sort(key=lambda x: x["volume"], reverse=True)
    selected = candidates[:top_n]

    symbols = [_bybit_to_symbol(c["symbol"]) for c in selected]
    logger.info("Top symbols: %s", ", ".join(symbols[:10]))
    return symbols


def _fallback_symbols(top_n: int = 30) -> List[str]:
    """
    Возвращает fallback-список топ-символов.

    Используется, когда API недоступен или вернул пустой результат.
    """
    return _FALLBACK_SYMBOLS[:top_n]


# ---------------------------------------------------------------------------
# 2. Загрузка исторических свечей с пагинацией
# ---------------------------------------------------------------------------


@_retry_with_backoff(max_retries=5, base_delay=1.0, max_delay=60.0)
async def _fetch_kline_page(
    client: httpx.AsyncClient,
    symbol: str,  # в формате Bybit: "BTCUSDT"
    interval: str,  # в формате Bybit: "60", "D", "W", ...
    start_ms: int,
    limit: int = 1000,
) -> Optional[Dict]:
    """
    Загрузить одну страницу свечей с Bybit V5.

    Эндпоинт: GET /v5/market/kline

    Параметры
    ---------
    symbol : str — символ в формате Bybit (например, "BTCUSDT")
    interval : str — интервал в формате Bybit (например, "60", "D")
    start_ms : int — стартовый timestamp в миллисекундах
    limit : int — количество свечей (максимум 1000)

    Возвращает
    ----------
    Dict с полем 'list': [[timestamp_ms, open, high, low, close, volume], ...]
    или None при ошибке.
    """
    url = f"{BYBIT_BASE_URL}/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "start": start_ms,
        "limit": limit,
    }

    response = await client.get(url, params=params, timeout=30.0)
    response.raise_for_status()
    data = response.json()

    if data.get("retCode") != 0:
        raise httpx.HTTPStatusError(
            f"Bybit error retCode={data['retCode']}, retMsg={data.get('retMsg')}",
            request=response.request,
            response=response,
        )

    return data


async def _fetch_ohlcv_paginated(
    client: httpx.AsyncClient,
    symbol: str,  # в формате проекта: "BTC/USDT"
    interval: str,  # в формате Bybit: "60", "240", "D", ...
    since_ms: int,
    until_ms: Optional[int] = None,
    max_candles: int = 200_000,
) -> Optional[pd.DataFrame]:
    """
    Асинхронная загрузка всех свечей с пагинацией назад во времени.

    ВАЖНО: Bybit V5 возвращает свечи в ПРЯМОМ порядке (от старых к новым),
    но принимает start_timestamp и возвращает max limit=1000 за запрос.

    Алгоритм пагинации:
    1. Начинаем с until_ms (или текущего времени).
    2. Запрашиваем свечи, где start <= candle_timestamp <= until_ms.
    3. Берём самый старый timestamp из ответа.
    4. Следующий запрос: start = самый_старый_timestamp - 1 мс.
    5. Повторяем, пока не достигнем since_ms.

    Почему начинаем с конца и двигаемся назад?
    - Bybit V5 kline endpoint не поддерживает end параметр, только start.
    - Мы запрашиваем "свечи начиная с start" и получаем limit последних из них.
    - Чтобы получить данные до определённой даты, используем хитрость:
      запрашиваем с большим start и берём последние limit свечей.
    - Альтернатива: запрашиваем весь диапазон, затем фильтруем.

    Более надёжный подход:
    - Запрос "с since_ms" → получаем первые 1000 свечей.
    - Берём последний timestamp, следующий запрос: start = last_ts + 1.
    - Двигаемся ВПЕРЁД от начала к концу периода.

    Параметры
    ---------
    client : httpx.AsyncClient
    symbol : str — символ в формате проекта ("BTC/USDT")
    interval : str — интервал в формате Bybit ("60", "240", "D", ...)
    since_ms : int — начальная дата периода (ms)
    until_ms : int | None — конечная дата (ms). Если None — до текущего времени.
    max_candles : int — защитный лимит на общее число свечей.

    Возвращает
    ----------
    pd.DataFrame | None — OHLCV DataFrame или None при ошибке.
    """
    bybit_symbol = _symbol_to_bybit(symbol)
    all_ohlcv: List[List] = []
    current_start = since_ms
    total_fetched = 0

    if until_ms is None:
        until_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)

    while total_fetched < max_candles:
        try:
            page = await _fetch_kline_page(
                client, bybit_symbol, interval,
                current_start, limit=1000,
            )
        except Exception as e:
            logger.error("Failed to fetch kline page for %s: %s", symbol, e)
            return None

        result_list = page.get("result", {}).get("list", [])
        if not result_list:
            # Нет больше данных — конец
            logger.debug(
                "Empty response for %s at start=%d. Pagination done.",
                symbol, current_start,
            )
            break

        # Bybit возвращает свечи в прямом порядке:
        # [timestamp, open, high, low, close, volume, turnover, adjChange]
        for candle in result_list:
            all_ohlcv.append([
                int(candle[0]),   # timestamp (ms)
                float(candle[1]), # open
                float(candle[2]), # high
                float(candle[3]), # low
                float(candle[4]), # close
                float(candle[5]), # volume
            ])

        total_fetched = len(all_ohlcv)
        last_ts = int(result_list[-1][0])
        first_ts = int(result_list[0][0])

        # Логирование прогресса
        if total_fetched % 5000 == 0 or total_fetched <= 1000:
            start_dt = _ms_to_dt(first_ts).strftime("%Y-%m-%d")
            end_dt = _ms_to_dt(last_ts).strftime("%Y-%m-%d")
            logger.info(
                "  ✓ %s: %d свечей (диапазон: %s → %s, next_start=%d)",
                symbol, total_fetched, start_dt, end_dt,
                current_start,
            )

        # Останавливаемся, если достигли target-диапазона
        if last_ts >= until_ms:
            break

        # Если получено меньше лимита (1000) — это конец доступных данных
        if len(result_list) < 1000:
            logger.debug(
                "Partial page (%d candles) for %s. No more data.",
                len(result_list), symbol,
            )
            break

        # Продвигаемся: следующий start = последний timestamp + 1 мс
        # Это обеспечивает отсутствие пропусков и дубликатов.
        current_start = last_ts + 1

    if not all_ohlcv:
        logger.warning("No data fetched for %s", symbol)
        return None

    return _ohlcv_to_df(all_ohlcv, symbol)


def _ohlcv_to_df(
    ohlcv: List[List],
    symbol: str,
) -> pd.DataFrame:
    """
    Преобразовать сырой OHLCV в DataFrame.

    Параметры
    ----------
    ohlcv : List[List]
        [[timestamp_ms, open, high, low, close, volume], ...]
    symbol : str
        "BTC/USDT"

    Возвращает
    ----------
    pd.DataFrame с DatetimeIndex (UTC) и колонками
    ['open', 'high', 'low', 'close', 'volume'].
    """
    if not ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )

    # Конвертация timestamp (мс) в DatetimeIndex с UTC
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)

    # Приведение типов к float
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Удаляем строки с NaN в ключевых полях
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)

    # Сортировка по индексу и удаление дубликатов (на всякий случай)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="last")]

    df.attrs["symbol"] = symbol
    return df


# ---------------------------------------------------------------------------
# 3. Параллельная загрузка для нескольких символов
# ---------------------------------------------------------------------------


async def fetch_all_symbols(
    symbols: List[str],
    interval: str,  # в формате Bybit: "60", "240", "D"
    since_ms: int,
    until_ms: Optional[int] = None,
    max_candles_per_symbol: int = 200_000,
    fetch_limit: int = 10,
) -> Dict[str, pd.DataFrame]:
    """
    Загрузить OHLCV для списка символов параллельно.

    Использует asyncio.Semaphore для ограничения параллельных запросов
    и избежания rate-лимитов.

    Параметры
    ---------
    symbols : List[str] — символы в формате "BTC/USDT"
    interval : str — интервал в формате Bybit
    since_ms : int — начальная дата (ms)
    until_ms : int | None — конечная дата
    max_candles_per_symbol : int — защитный лимит свечей на символ
    fetch_limit : int — макс. параллельных загрузок символов

    Возвращает
    ----------
    Dict[str, pd.DataFrame] — символ → DataFrame
    """
    # Создаём один httpx.AsyncClient для всех запросов.
    # Это гарантирует корректное закрытие соединения и переиспользование
    # TCP/TLS-сессий (no "Unclosed client session").
    async with httpx.AsyncClient(
        timeout=30.0,
        follow_redirects=True,
        headers={"User-Agent": "torg-bot/1.0"},
    ) as client:
        semaphore = asyncio.Semaphore(fetch_limit)
        data: Dict[str, pd.DataFrame] = {}

        async def _fetch_one(sym: str) -> Tuple[str, pd.DataFrame | None, int]:
            """Загрузить данные для одного символа с ограничением parallelism."""
            async with semaphore:
                try:
                    df = await _fetch_ohlcv_paginated(
                        client, sym, interval,
                        since_ms, until_ms, max_candles_per_symbol,
                    )
                    count = len(df) if df is not None else 0
                    return sym, df, count
                except Exception as e:
                    logger.error("Failed to fetch %s: %s", sym, e)
                    return sym, None, 0

        tasks = [_fetch_one(sym) for sym in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error("Task exception: %s", result)
                continue

            if isinstance(result, tuple) and len(result) == 3:
                sym, df, count = result
                if df is not None and count > 0:
                    data[sym] = df

    return data


# ---------------------------------------------------------------------------
# 4. Кэширование (совместимо со старым интерфейсом)
# ---------------------------------------------------------------------------


def _cache_path(cache_dir: str, symbol: str, timeframe: str) -> pathlib.Path:
    """Сформировать путь к кэш-файлу для символа и таймфрейма."""
    safe = symbol.replace("/", "_")
    return pathlib.Path(cache_dir) / f"{safe}_{timeframe}.parquet"


def _is_cache_fresh(cache_file: pathlib.Path, max_age_days: int) -> bool:
    """Проверить, не истёк ли кэш по возрасту."""
    if not cache_file.exists():
        return False
    age = time.time() - cache_file.stat().st_mtime
    max_age = max_age_days * 86400  # секунд в днях
    return age < max_age


def save_cache(
    data: Dict[str, pd.DataFrame],
    cache_dir: str,
    max_age_days: int = 1,
) -> None:
    """Сохранить данные в parquet-файлы."""
    pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)
    saved = 0
    for symbol, df in data.items():
        # Таймфрейм берём из attrs, установленного при загрузке
        timeframe = df.attrs.get("timeframe", "1h")
        path = _cache_path(cache_dir, symbol, timeframe)
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
    Dict[str, pd.DataFrame] — только те символы, кэш которых существует и свеж.
    """
    data: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = _cache_path(cache_dir, symbol, timeframe)
        if path.exists() and _is_cache_fresh(path, max_age_days=365):
            try:
                df = pd.read_parquet(path)
                df.attrs["timeframe"] = timeframe
                df.attrs["cached"] = True
                df.attrs["symbol"] = symbol
                data[symbol] = df
            except Exception as e:
                logger.warning("Failed to read cache %s: %s", path, e)
        else:
            logger.debug("Cache not found or expired: %s", path)
    logger.info("Loaded %d datasets from cache", len(data))
    return data


# ---------------------------------------------------------------------------
# Главный класс загрузки данных
# ---------------------------------------------------------------------------


class DataLoader:
    """
    Загрузка и кэширование исторических данных с Bybit.

    Использует прямую асинхронную интеграцию с Bybit API V5
    (без ccxt), что устраняет проблемы с "Unclosed client session",
    неверным определением фьючерсов и лимитами CCXT.
    """

    def __init__(self, config: Config):
        self.cfg = config
        self.cache_dir = pathlib.Path(config.data.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # --- Символы ----------------------------------------------------------

    def _get_symbols(self) -> List[str]:
        """Получить список символов для торговли."""
        cached_files = list(self.cache_dir.glob("*.parquet"))
        if cached_files and not self.cfg.download_only:
            # Извлекаем символы из имён файлов: BTC_USDT_1h → BTC/USDT
            existing_symbols: List[str] = []
            for f in cached_files:
                parts = f.stem.split("_")
                base_parts = parts[:-1]  # последняя часть — таймфрейм
                symbol = "/".join(base_parts)
                existing_symbols.append(symbol)
            logger.info("%d cached datasets found", len(existing_symbols))
            return existing_symbols

        return self._run_sync(_get_top_symbols(
            None,  # будет создан внутри
            top_n=self.cfg.data.top_coins,
            quote=self.cfg.data.quote_currency,
        ))

    @staticmethod
    def _run_sync(coro) -> Any:
        """Запустить async-корутину из синхронного контекста."""
        return asyncio.run(coro)

    # --- Фильтрация по кэшу -----------------------------------------------

    def _filter_cached(
        self,
        symbols: List[str],
    ) -> Tuple[List[str], List[str]]:
        """
        Разделить символы на те, что нужно загрузить, и уже кэшированные.

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

        logger.info("Already cached: %d, need download: %d",
                    len(already_cached), len(to_download))
        return to_download, already_cached

    # --- Основная точка загрузки -------------------------------------------

    def load(self) -> Dict[str, pd.DataFrame]:
        """
        Загрузить OHLCV данные для всех символов.

        Алгоритм:
        1. Получить список топ-символов
        2. Проверить кэш
        3. Скачать недостающее через Bybit V5 API
        4. Слить и сохранить кэш

        Возвращает
        ----------
        Dict[str, pd.DataFrame] — символ → OHLCV DataFrame
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
            data = load_cache(
                str(self.cache_dir),
                already_cached,
                self.cfg.data.timeframe,
            )

        # Скачивание нового
        if to_download and not self.cfg.skip_download:
            logger.info("Downloading %d datasets from Bybit...", len(to_download))

            # Вычисляем даты
            since_dt = datetime.now(tz=timezone.utc) - timedelta(
                days=self.cfg.data.data_years * 365
            )
            since_ms = int(since_dt.timestamp() * 1000)
            until_ms = None  # до текущего времени

            # Маппинг таймфрейма
            interval = TIMEFRAME_TO_INTERVAL.get(
                self.cfg.data.timeframe, "60",
            )

            fetched = self._run_sync(
                fetch_all_symbols(
                    to_download,
                    interval,
                    since_ms,
                    until_ms,
                    max_candles_per_symbol=200_000,
                    fetch_limit=10,
                ),
            )
            data.update(fetched)

        if not data:
            logger.error("No data loaded. Aborting.")
            return {}

        # Сохраняем кэш (один раз — все символы)
        # Пока не устанавливаем timeframe, сделаем это в load_for_backtest
        save_cache(data, str(self.cache_dir), self.cfg.data.cache_max_age_days)

        logger.info("Total loaded: %d symbols", len(data))
        return data

    def load_for_backtest(self) -> Dict[str, pd.DataFrame]:
        """
        Загрузить данные, готовые для бэктеста.

        Делает дополнительную обработку:
        - Оставляет только нужные колонки [O, H, L, C, V]
        - Устанавливает attrs (timeframe, symbol)
        - Сортирует по индексу
        - Повторно кэширует уже отфильтрованные данные
        """
        data = self.load()
        for symbol, df in data.items():
            df = df[["open", "high", "low", "close", "volume"]].copy()
            df.sort_index(inplace=True)
            df.dropna(
                subset=["open", "high", "low", "close"], inplace=True,
            )
            df.attrs["timeframe"] = self.cfg.data.timeframe
            df.attrs["symbol"] = symbol

        # Повторный кэш уже с правильным timeframe
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
        for tf in sorted(timeframes):  # sorted для детерминированного порядка
            loader = DataLoader(self.cfg)
            # Временно переопределяем таймфрейм
            orig_tf = loader.cfg.data.timeframe
            loader.cfg.data.timeframe = tf
            data = loader.load_for_backtest()
            loader.cfg.data.timeframe = orig_tf
            result[tf] = data

        logger.info("MultiTimeframe loaded: %d timeframes", len(result))
        return result
