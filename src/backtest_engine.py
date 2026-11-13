"""
Движок бэктеста.

Объединяет все модули в единую пайплайн:
1. Загрузка данных (DataLoader)
2. Расчёт индикаторов (indicators)
3. Генерация сигналов (strategy)
4. Симуляция торговли с риск-менеджментом
5. Расчёт метрик (metrics)
6. Визуализация (visualization)
7. Экспорт результатов

Использует backtrader как основу, но с кастомной логикой сигналов
из strategy.py для полного контроля над исполнением ордеров.
"""

from __future__ import annotations

import logging
import pathlib
import time
from typing import Dict, List, Optional

import pandas as pd

from src.config import Config
from src.data_loader import DataLoader
from src.indicators import add_all_indicators
from src.metrics import PerformanceMetrics, compute_metrics, format_metrics_report
from src.risk_manager import RiskManager
from src.strategy import SimulationResult, Trade, generate_signals, simulate_trading
from src.visualization import export_all_visualizations

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------


class BacktestEngine:
    """
    Основной движок бэктеста.

    Пайплайн:
    data → indicators → signals → simulation → metrics → visualization

    Атрибуты
    --------
    config : Config
    risk_manager : RiskManager
    trades : List[Trade]
    equity_curve : pd.DataFrame
    metrics : PerformanceMetrics
    """

    def __init__(self, config: Config):
        self.cfg = config
        self.risk_manager = RiskManager(config.strategy)
        self.trades: List[Trade] = []
        self.equity_curve: pd.DataFrame = pd.DataFrame()
        self.metrics: PerformanceMetrics = PerformanceMetrics()

    def run(self) -> None:
        """
        Полный прогон бэктеста.

        1. Загрузка данных
        2. Индикаторы
        3. Сигналы
        4. Симуляция
        5. Метрики
        6. Визуализация
        7. Экспорт
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("BACKTEST START")
        logger.info("=" * 60)

        # === 1. Загрузка данных ===
        logger.info("[%s/6] Loading data...", 1)
        data_loader = DataLoader(self.cfg)
        symbols_data = data_loader.load_for_backtest()

        if not symbols_data:
            logger.error("No data loaded. Aborting.")
            return

        logger.info("Loaded %d symbols", len(symbols_data))

        # === 2. Индикаторы ===
        logger.info("[%s/6] Computing indicators...", 2)
        for symbol, df in symbols_data.items():
            symbols_data[symbol] = add_all_indicators(df, self.cfg.strategy)

        # Проверка на пустые DF после dropna
        symbols_data = {
            k: v for k, v in symbols_data.items()
            if not v.empty and "atr" in v.columns
        }
        logger.info("%d symbols ready for backtest", len(symbols_data))

        # === 3. Сигналы ===
        logger.info("[%s/6] Generating signals...", 3)
        for symbol, df in symbols_data.items():
            symbols_data[symbol] = generate_signals(
                df, self.cfg.strategy,
                fee_rate=self.cfg.strategy.maker_fee,
            )

        # Подсчёт сигналов
        total_signals = sum(df["signal"].sum() for df in symbols_data.values())
        logger.info("Total entry signals: %d", int(total_signals))

        # === 4. Симуляция торговли ===
        logger.info("[%s/6] Running simulation...", 4)
        funding_pct = 0.0
        if self.cfg.data.market_type == "futures" and not self.cfg.strategy.use_real_funding:
            funding_pct = self.cfg.strategy.fixed_funding_pct

        result = simulate_trading(
            symbols_data,
            self.cfg.strategy,
            self.cfg.backtest,
            fee_rate=self.cfg.strategy.maker_fee,
            slippage_pct=self.cfg.strategy.slippage_pct,
            funding_pct=funding_pct,
        )

        self.trades = result.trades
        self.equity_curve = result.equity_curve

        logger.info("Simulation complete: %d trades, %d unclosed",
                     len([t for t in self.trades if t.is_closed]),
                     len([t for t in self.trades if not t.is_closed]))

        # === 5. Метрики ===
        logger.info("[%s/6] Computing metrics...", 5)
        tf = self.cfg.data.timeframe
        self.metrics = compute_metrics(
            self.trades,
            self.equity_curve,
            self.cfg.backtest.initial_cash,
            timeframe=tf,
        )

        # === 6. Визуализация ===
        logger.info("[%s/6] Generating visualizations...", 6)
        export_all_visualizations(
            self.equity_curve,
            self.trades,
            self.metrics,
            self.cfg.output.results_dir,
            generate_html=self.cfg.output.generate_html,
            save_png=True,
        )

        # === 7. Экспорт ===
        self._export_results()

        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info("BACKTEST COMPLETE — %.2f seconds", elapsed)
        logger.info("=" * 60)

        # Консольный отчёт
        report = format_metrics_report(self.metrics)
        print(report)

    # ------------------------------------------------------------------
    # Экспорт
    # ------------------------------------------------------------------

    def _export_results(self) -> None:
        """Экспорт результатов в CSV и текстовый отчёт."""
        out = pathlib.Path(self.cfg.output.results_dir)
        out.mkdir(parents=True, exist_ok=True)

        # --- Trades CSV ---
        if self.cfg.output.save_trades_csv:
            trades_rows = []
            for t in self.trades:
                trades_rows.append({
                    "symbol": t.symbol,
                    "entry_time": t.entry_time,
                    "entry_price": t.entry_price,
                    "exit_time": t.exit_time,
                    "exit_price": t.exit_price,
                    "exit_reason": t.exit_reason,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "holding_bars": t.holding_bars,
                    "commission": t.commission,
                    "is_closed": t.is_closed,
                })

            if trades_rows:
                trades_df = pd.DataFrame(trades_rows)
                trades_df.to_csv(out / "trades.csv", index=False)
                logger.info("Trades CSV saved: %d rows", len(trades_df))
            else:
                pd.DataFrame(columns=[
                    "symbol", "entry_time", "entry_price",
                    "exit_time", "exit_price", "exit_reason",
                    "pnl", "pnl_pct", "holding_bars",
                    "commission", "is_closed",
                ]).to_csv(out / "trades.csv", index=False)

        # --- Equity curve CSV ---
        if self.cfg.output.save_equity_curve and not self.equity_curve.empty:
            self.equity_curve.to_csv(out / "equity_curve.csv")
            logger.info("Equity curve CSV saved")

        # --- Metrics text report ---
        from src.metrics import format_metrics_report
        report = format_metrics_report(self.metrics)
        with open(out / "metrics_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
            f.write("\n\nTrade list:\n")
            for t in self.trades[:20]:
                pnl = f"${t.pnl:,.2f}" if t.pnl is not None else "N/A"
                exit_str = f"{t.exit_price:.2f}" if t.exit_price else "N/A"
                f.write(
                    f"  {t.symbol:15s} {str(t.entry_time)[:10]:10s} "
                    f"entry={t.entry_price:.2f} → exit={exit_str} "
                    f"pnl={pnl} [{t.exit_reason}]\n"
                )

        logger.info("Text report saved to %s", out / "metrics_report.txt")
