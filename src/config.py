"""
Конфигурация бэктест-фреймворка.

Загрузка параметров из YAML, валидация, предоставление через dataclass.
Все магические числа — в одном месте.
"""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass, field
from typing import List

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses для каждого раздела конфига
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    """Параметры загрузки данных."""

    exchange: str = "binance"
    market_type: str = "futures"  # 'spot' или 'futures'
    timeframe: str = "1h"
    data_years: int = 2
    top_coins: int = 30
    quote_currency: str = "USDT"  # базовая валюта для фильтрации символов
    exclude: List[str] = field(default_factory=lambda: [
        "USDT", "BUSD", "FDUSD", "TUSD", "USDC", "DAI", "EUR", "GBP", "AUD"
    ])
    cache_dir: str = "data_cache"
    cache_max_age_days: int = 1


@dataclass
class StrategyConfig:
    """Параметры стратегии Range Trading + Trend Filter."""

    # --- Trend detection (S1) ---
    pivot_order: int = 3
    trend_timeframes: List[str] = field(default_factory=lambda: ["1d", "1w"])

    # --- Channels (M2) ---
    sma_period: int = 30
    atr_period: int = 14
    atr_multiplier: float = 2.2
    channel_method: str = "atr"  # 'atr' или 'donchian'
    donchian_period: int = 30

    # --- Entry (M3) ---
    rsi_period: int = 14
    rsi_threshold: float = 35
    atr_pct_min: float = 1.0
    atr_pct_max: float = 6.0
    volume_sma_period: int = 50

    # --- Exit (M4) ---
    stop_multiplier: float = 1.8
    take_profit: str = "upper_band"
    trailing_multiplier: float = 1.5

    # --- Risk (M5) ---
    risk_per_trade: float = 0.02
    max_positions: int = 4
    correlation_symbol: str = "BTC/USDT"
    correlation_window: int = 60
    correlation_threshold: float = 0.8
    leverage: int = 1
    liquidation_buffer: float = 2.0

    # --- Costs (M6) ---
    maker_fee: float = 0.0004
    taker_fee: float = 0.0006
    slippage_pct: float = 0.0005
    use_real_funding: bool = False
    fixed_funding_pct: float = 0.0001
    funding_interval_hours: int = 8


@dataclass
class BacktestConfig:
    """Параметры прогона бэктеста."""

    initial_cash: float = 100000.0
    quote_currency: str = "USDT"
    print_table: bool = True
    plot: bool = True
    plot_style: str = "bar"


@dataclass
class OutputConfig:
    """Параметры вывода результатов."""

    results_dir: str = "results"
    generate_html: bool = True
    save_trades_csv: bool = True
    save_equity_curve: bool = True


@dataclass
class LoggingConfig:
    """Параметры логирования."""

    level: str = "INFO"
    file: str = "backtest.log"


# ---------------------------------------------------------------------------
# Главный конфиг
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Единый конфиг — обёртка над всеми разделами."""

    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # CLI-переопределения
    download_only: bool = False
    skip_download: bool = False
    timeframe: str | None = None
    top_coins: int | None = None
    years: int | None = None


# ---------------------------------------------------------------------------
# Загрузка и валидация
# ---------------------------------------------------------------------------

# Пути поиска конфигов (от более приоритетных)
_DEFAULT_CONFIG_PATHS = [
    pathlib.Path("config.yaml"),
    pathlib.Path("config.yml"),
]


def _resolve_config_path(cli_path: str | None = None) -> pathlib.Path | None:
    """Найти файл конфига: CLI > default paths > cwd."""
    if cli_path:
        p = pathlib.Path(cli_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Config file not found: {cli_path}")

    for p in _DEFAULT_CONFIG_PATHS:
        if p.exists():
            return p

    return None


def _deep_merge(base: dict, override: dict) -> dict:
    """Рекурсивный merge override поверх base."""
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(
    cli_path: str | None = None,
    cli_overrides: dict | None = None,
) -> Config:
    """
    Загрузить конфиг из YAML, применить CLI-переопределения.

    Параметры
    ---------
    cli_path : str | None
        Путь к YAML-файлу. Если None — ищутся стандартные имена.
    cli_overrides : dict | None
        Словарь с параметрами для переопределения (как в YAML-структуре).

    Возвращает
    ---------
    Config
        Валидированный объект конфигурации.
    """
    config_path = _resolve_config_path(cli_path)
    base: dict = {}
    if config_path:
        logger.info("Loading config from: %s", config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            base = yaml.safe_load(f) or {}

    if cli_overrides:
        base = _deep_merge(base, cli_overrides)

    cfg = Config()
    cfg.data = _dataclass_from_dict(DataConfig, base.get("data", {}))
    cfg.strategy = _dataclass_from_dict(StrategyConfig, base.get("strategy", {}))
    cfg.backtest = _dataclass_from_dict(BacktestConfig, base.get("backtest", {}))
    cfg.output = _dataclass_from_dict(OutputConfig, base.get("output", {}))
    cfg.logging = _dataclass_from_dict(LoggingConfig, base.get("logging", {}))

    _validate(cfg)
    return cfg


def _dataclass_from_dict(cls, d: dict):
    """Создать dataclass из словаря, игнорируя лишние ключи."""
    fields = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in d.items() if k in fields}
    return cls(**filtered)


def _validate(cfg: Config) -> None:
    """Базовая валидация конфигурации."""
    # Таймфрейм
    valid_tfs = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "1w", "1M"}
    if cfg.data.timeframe not in valid_tfs:
        raise ValueError(f"Invalid timeframe: {cfg.data.timeframe}. Must be one of {valid_tfs}")

    # Канал
    if cfg.strategy.channel_method not in ("atr", "donchian"):
        raise ValueError(
            f"Invalid channel_method: {cfg.strategy.channel_method}. "
            "Must be 'atr' or 'donchian'."
        )

    # Рынок
    if cfg.data.market_type not in ("spot", "futures"):
        raise ValueError(f"Invalid market_type: {cfg.data.market_type}. Must be 'spot' or 'futures'")

    # Риски
    if not (0 < cfg.strategy.risk_per_trade <= 0.10):
        raise ValueError(f"risk_per_trade must be in (0, 0.10], got {cfg.strategy.risk_per_trade}")

    if cfg.strategy.max_positions < 1:
        raise ValueError(f"max_positions must be >= 1, got {cfg.strategy.max_positions}")

    if cfg.strategy.leverage < 1:
        raise ValueError(f"leverage must be >= 1, got {cfg.strategy.leverage}")

    # Период
    if cfg.data.data_years < 1:
        raise ValueError(f"data_years must be >= 1, got {cfg.data.data_years}")

    logger.info("Configuration validated successfully.")
