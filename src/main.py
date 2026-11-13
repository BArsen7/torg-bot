#!/usr/bin/env python3
"""
Точка входа бэктест-фреймворка.

Запуск:
    python main.py                          # по умолчанию
    python main.py --config config.yaml     # свой конфиг
    python main.py --timeframe 4h           # переопределить ТФ
    python main.py --download-only          # только загрузка данных
    python main.py --skip-download          # использовать кэш
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

# Добавляем корень проекта в sys.path, чтобы работали импорты src.X
_project_root = pathlib.Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import load_config
from src.backtest_engine import BacktestEngine


def setup_logging(cfg) -> None:
    """Настроить логирование."""
    log_cfg = cfg.logging
    level = getattr(logging, log_cfg.level.upper(), logging.INFO)

    handlers = [
        logging.StreamHandler(sys.stdout),
    ]

    if log_cfg.file:
        handlers.append(logging.FileHandler(log_cfg.file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def parse_args() -> argparse.Namespace:
    """Парсинг CLI-аргументов."""
    parser = argparse.ArgumentParser(
        description="Backtest: Range Trading with Trend Filter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python main.py
  python main.py --config my_config.yaml
  python main.py --timeframe 4h --top-coins 20
  python main.py --download-only --skip-download
        """,
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Путь к YAML-конфигу (default: config.yaml)",
    )
    parser.add_argument(
        "--timeframe", "-t",
        type=str,
        default=None,
        help="Таймфрейм: 1h, 4h, 1d (переопределяет config)",
    )
    parser.add_argument(
        "--top-coins", "-n",
        type=int,
        default=None,
        help="Количество монет (переопределяет config)",
    )
    parser.add_argument(
        "--years", "-y",
        type=int,
        default=None,
        help="Период данных в годах (переопределяет config)",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Только загрузить данные, не запускать бэктест",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Пропустить загрузку, использовать кэш",
    )
    parser.add_argument(
        "--market-type",
        type=str,
        choices=["spot", "futures"],
        default=None,
        help="Тип рынка (переопределяет config)",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=None,
        help="Начальный капитал (переопределяет config)",
    )

    return parser.parse_args()


def build_cli_overrides(args: argparse.Namespace) -> dict:
    """Собрать CLI-переопределения в формате конфига."""
    overrides = {}
    if args.timeframe:
        overrides.setdefault("data", {})["timeframe"] = args.timeframe
    if args.top_coins is not None:
        overrides.setdefault("data", {})["top_coins"] = args.top_coins
    if args.years is not None:
        overrides.setdefault("data", {})["data_years"] = args.years
    if args.market_type:
        overrides.setdefault("data", {})["market_type"] = args.market_type
    if args.initial_cash is not None:
        overrides.setdefault("backtest", {})["initial_cash"] = args.initial_cash

    # Флаги
    flags = {}
    if args.download_only:
        flags["download_only"] = True
    if args.skip_download:
        flags["skip_download"] = True

    return overrides, flags


def main() -> None:
    """Точка входа."""
    args = parse_args()
    overrides, flags = build_cli_overrides(args)

    # Загрузка конфига
    cfg = load_config(cli_path=args.config, cli_overrides=overrides)

    # Применяем CLI-флаги
    if args.download_only:
        cfg.download_only = True
    if args.skip_download:
        cfg.skip_download = True

    # Настройка логирования
    setup_logging(cfg)

    logger = logging.getLogger(__name__)

    logger.info("Config loaded: timeframe=%s, coins=%d, years=%d, market=%s",
                cfg.data.timeframe, cfg.data.top_coins, cfg.data.data_years,
                cfg.data.market_type)

    # Запуск движка
    engine = BacktestEngine(cfg)

    if cfg.download_only:
        logger.info("Download-only mode: loading data...")
        from src.data_loader import DataLoader
        dl = DataLoader(cfg)
        data = dl.load_for_backtest()
        logger.info("Data loaded: %d symbols", len(data))
    else:
        engine.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        logging.getLogger(__name__).error("FATAL: %s", e, exc_info=True)
        sys.exit(1)
