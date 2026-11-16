"""
Загрузка и кэширование исторических данных.
Источник: Bybit API V5 (прямая асинхронная интеграция через httpx).
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pandas as pd

from src.config import Config

logger = logging.getLogger(__name__)

BYBIT_BASE_URL = "https://api.bybit.com"

# Разрешенные базовые активы (исключаем токенизированные акции типа NVDA, SOXL, AKE)
ALLOWED_CRYPTO_BASES = {
    "BTC", "ETH", "SOL", "XRP", "ZEC", "HYPE", "NEAR", "DOGE", "ADA", "AVAX",
    "LINK", "DOT", "UNI", "ATOM", "LTC", "ETC", "FIL", "APT", "ARB", "ALGO",
    "FTM", "SAND", "MANA", "AXS", "THETA", "ICP", "VET", "HBAR", "EGLD", "XTZ",
    "BNB", "MATIC", "TRX", "SUI", "PEPE", "WLD", "ONDO", "TAO", "INJ", "AAVE", "MSTR"
}

def _ohlcv_to_df(ohlcv: List[List], symbol: str) -> pd.DataFrame:
    if not ohlcv:
        return pd.DataFrame()
    
    # Bybit возвращает 7 столбцов (добавлен 'turnover'), берем только первые 6 для OHLCV
    clean_ohlcv = [row[:6] for row in ohlcv]
    df = pd.DataFrame(
        clean_ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="last")]
    df.attrs["symbol"] = symbol
    return df

async def _fetch_ohlcv_paginated(
    client: httpx.AsyncClient,
    symbol_bybit: str,
    interval: str,
    since_ms: int,
    limit: int = 1000,
) -> List[List]:
    all_ohlcv: List[List] = []
    # Bybit V5 возвращает данные от новых к старым.
    # Начинаем с текущего времени и двигаемся назад к since_ms.
    current_end = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    
    while current_end > since_ms:
        params = {
            "category": "linear",
            "symbol": symbol_bybit,
            "interval": interval,
            "end": current_end,
            "limit": limit,
        }

        response = await client.get(f"{BYBIT_BASE_URL}/v5/market/kline", params=params)
        response.raise_for_status()
        data = response.json()

        if data.get("retCode") != 0:
            logger.error(f"Bybit API error for {symbol_bybit}: {data.get('retMsg')}")
            break

        raw_list = data.get("result", {}).get("list", [])
        if not raw_list:
            break

        all_ohlcv.extend(raw_list)
        
        # raw_list отсортирован от новых к старым.
        # Последний элемент - самая старая свеча в этом батче.
        oldest_ts = int(raw_list[-1][0])
        
        # Сдвигаем конец окна времени на 1 мс перед самой старой свечей
        current_end = oldest_ts - 1
        
        # Если получили меньше лимита, значит это самые старые доступные данные
        if len(raw_list) < limit:
            break
            
        # Защита от бесконечного цикла (макс ~50k свечей = ~5.7 лет для 1h)
        if len(all_ohlcv) > 50000:
            logger.warning(f"Reached max safety limit for {symbol_bybit}, stopping fetch.")
            break

    # all_ohlcv сейчас идет от новых к старым. Переворачиваем, чтобы было от старых к новым.
    all_ohlcv.reverse()
    
    # Удаляем возможные дубликаты на стыке батчей
    unique_ohlcv = []
    seen_ts = set()
    for candle in all_ohlcv:
        ts = int(candle[0])
        if ts not in seen_ts:
            seen_ts.add(ts)
            unique_ohlcv.append(candle)

    return unique_ohlcv

async def fetch_all_symbols(
    symbols: List[str],
    interval: str,
    since_ms: int,
    fetch_limit: int = 10,
) -> Dict[str, pd.DataFrame]:
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        semaphore = asyncio.Semaphore(fetch_limit)
        data: Dict[str, pd.DataFrame] = {}

        async def _fetch_one(sym: str) -> Tuple[str, Optional[pd.DataFrame], int]:
            async with semaphore:
                try:
                    symbol_bybit = sym.replace("/", "")
                    ohlcv = await _fetch_ohlcv_paginated(client, symbol_bybit, interval, since_ms)
                    df = _ohlcv_to_df(ohlcv, sym)
                    count = len(df) if not df.empty else 0
                    if count > 0:
                        logger.info(f"  ✓ {sym}: {count} свечей (диапазон: {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')})")
                    return sym, df if not df.empty else None, count
                except Exception as e:
                    logger.error(f"Failed to fetch {sym}: {e}")
                    return sym, None, 0

        tasks = [_fetch_one(sym) for sym in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task exception: {result}")
                continue
            if isinstance(result, tuple) and len(result) == 3:
                sym, df, count = result
                if df is not None and isinstance(df, pd.DataFrame) and count > 0:
                    data[sym] = df

    return data

async def _get_top_symbols_impl(client: httpx.AsyncClient, top_n: int, quote: str = "USDT") -> List[str]:
    response = await client.get(f"{BYBIT_BASE_URL}/v5/market/tickers", params={"category": "linear"})
    response.raise_for_status()
    data = response.json()
    
    if data.get("retCode") != 0:
        logger.error(f"Bybit API error: {data.get('retMsg')}")
        return []
        
    tickers_list = data.get("result", {}).get("list", [])
    candidates = []
    
    for t in tickers_list:
        symbol = t.get("symbol", "")
        if not symbol.endswith(quote):
            continue
            
        base = symbol.replace(quote, "")
        # Фильтр: оставляем только известные крипто-активы
        if base not in ALLOWED_CRYPTO_BASES:
            continue
            
        turnover = float(t.get("turnover24h", 0) or 0)
        if turnover < 1_000_000:
            continue
            
        candidates.append({"symbol": f"{base}/{quote}", "volume": turnover})
        
    candidates.sort(key=lambda x: x["volume"], reverse=True)
    selected = candidates[:top_n]
    
    symbols = [c["symbol"] for c in selected]
    logger.info(f"Top symbols: {', '.join(symbols)}")
    return symbols

def _cache_path(cache_dir: str, symbol: str, timeframe: str) -> pathlib.Path:
    safe = symbol.replace("/", "_")
    return pathlib.Path(cache_dir) / f"{safe}_{timeframe}.csv"

def _is_cache_fresh(cache_file: pathlib.Path, max_age_days: int) -> bool:
    if not cache_file.exists():
        return False
    age = time.time() - cache_file.stat().st_mtime
    return age < (max_age_days * 86400)

def save_cache(data: Dict[str, pd.DataFrame], cache_dir: str, max_age_days: int = 1) -> None:
    pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)
    saved = 0
    for symbol, df in data.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            logger.warning(f"Пропуск кэширования {symbol}: не DataFrame или пуст")
            continue
        path = _cache_path(cache_dir, symbol, df.attrs.get("timeframe", "1h"))
        df.to_csv(path)
        saved += 1
    logger.info(f"Cached {saved} symbol datasets to {cache_dir}/")

def load_cache(cache_dir: str, symbols: List[str], timeframe: str) -> Dict[str, pd.DataFrame]:
    data: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = _cache_path(cache_dir, symbol, timeframe)
        if path.exists() and _is_cache_fresh(path, max_age_days=365):
            try:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                df.attrs["timeframe"] = timeframe
                df.attrs["cached"] = True
                df.attrs["symbol"] = symbol
                data[symbol] = df
            except Exception as e:
                logger.warning(f"Failed to read cache {path}: {e}")
    logger.info(f"Loaded {len(data)} datasets from cache")
    return data

class DataLoader:
    def __init__(self, config: Config):
        self.cfg = config
        self.cache_dir = pathlib.Path(config.data.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _run_sync(self, coro):
        return asyncio.run(coro)

    def load(self) -> Dict[str, pd.DataFrame]:
        async def _load_async():
            async with httpx.AsyncClient(timeout=30.0) as client:
                symbols = await _get_top_symbols_impl(client, self.cfg.data.top_coins, self.cfg.data.quote_currency)
            
            if not symbols:
                return {}
                
            to_download, already_cached = self._filter_cached(symbols)
            data = load_cache(str(self.cache_dir), already_cached, self.cfg.data.timeframe) if already_cached else {}
            
            if to_download and not self.cfg.skip_download:
                logger.info(f"Downloading {len(to_download)} datasets from Bybit...")
                since_dt = datetime.now(tz=timezone.utc) - timedelta(days=self.cfg.data.data_years * 365)
                since_ms = int(since_dt.timestamp() * 1000)
                interval = {"1h": "60", "4h": "240", "1d": "D"}.get(self.cfg.data.timeframe, "60")
                
                fetched = await fetch_all_symbols(to_download, interval, since_ms, fetch_limit=10)
                data.update(fetched)
                
            if not data:
                logger.error("No data loaded. Aborting.")
                return {}
                
            save_cache(data, str(self.cache_dir), self.cfg.data.cache_max_age_days)
            logger.info(f"Total loaded: {len(data)} symbols")
            return data
            
        return self._run_sync(_load_async())

    def _filter_cached(self, symbols: List[str]) -> Tuple[List[str], List[str]]:
        to_download, already_cached = [], []
        for symbol in symbols:
            path = _cache_path(str(self.cache_dir), symbol, self.cfg.data.timeframe)
            if path.exists() and _is_cache_fresh(path, self.cfg.data.cache_max_age_days):
                already_cached.append(symbol)
            else:
                to_download.append(symbol)
        logger.info(f"Already cached: {len(already_cached)}, need download: {len(to_download)}")
        return to_download, already_cached

    def load_for_backtest(self) -> Dict[str, pd.DataFrame]:
        data = self.load()
        for symbol, df in data.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df = df[["open", "high", "low", "close", "volume"]].copy()
                df.sort_index(inplace=True)
                df.dropna(subset=["open", "high", "low", "close"], inplace=True)
                df.attrs["timeframe"] = self.cfg.data.timeframe
                df.attrs["symbol"] = symbol
                data[symbol] = df
        save_cache(data, str(self.cache_dir), self.cfg.data.cache_max_age_days)
        return data
