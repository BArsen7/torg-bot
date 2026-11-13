"""
Визуализация результатов бэктеста.

Возможности:
- Equity curve + drawdown
- Распределение доходностей
- Heatmap monthly returns
- Графики отдельных сделок
- HTML-отчёт (plotly, интерактивный)
"""

from __future__ import annotations

import logging
import pathlib
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.metrics import PerformanceMetrics
from src.strategy import Trade

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Matplotlib графики (статичные PNG)
# ---------------------------------------------------------------------------


def plot_equity_curve(
    equity_curve: pd.DataFrame,
    metrics: PerformanceMetrics,
    save_path: Optional[pathlib.Path] = None,
) -> None:
    """
    Нарисовать equity curve + drawdown под ней.

    Параметры
    ---------
    equity_curve : pd.DataFrame с колонками ['equity']
    metrics : PerformanceMetrics
    save_path : Path — куда сохранить
    """
    if equity_curve.empty or "equity" not in equity_curve.columns:
        logger.warning("Cannot plot equity curve: empty data")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                              gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("Backtest Results", fontsize=16, fontweight="bold")

    eq = equity_curve["equity"]
    dates = equity_curve.index

    # --- Equity curve ---
    ax1 = axes[0]
    ax1.fill_between(dates, eq.values, alpha=0.3, color="steelblue")
    ax1.plot(dates, eq.values, color="steelblue", linewidth=1.5)
    ax1.set_ylabel("Equity ($)")
    ax1.set_title(
        f"Return: {metrics.total_return:.1%} | "
        f"CAGR: {metrics.cagr:.1%} | "
        f"Sharpe: {metrics.sharpe_ratio:.2f}"
    )
    ax1.grid(True, alpha=0.3)

    # --- Drawdown ---
    ax2 = axes[1]
    rolling_max = eq.cummax()
    dd = (eq - rolling_max) / rolling_max * 100
    ax2.fill_between(dates, dd.values, alpha=0.5, color="coral")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.set_title(f"Max DD: {metrics.max_drawdown_pct:.1%}")
    ax2.grid(True, alpha=0.3)

    # Форматирование дат
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=30)

    fig.autofmt_xdate()
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Equity curve saved to %s", save_path)
    else:
        plt.show()

    plt.close(fig)


def plot_trades_on_equity(
    equity_curve: pd.DataFrame,
    trades: List[Trade],
    save_path: Optional[pathlib.Path] = None,
) -> None:
    """
    Нарисовать equity curve с пометками сделок.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    eq = equity_curve["equity"]

    ax.plot(eq.index, eq.values, color="steelblue", linewidth=1.5, label="Equity")

    # Покупки
    buys = [t for t in trades if t.entry_time]
    for t in buys:
        color = "green" if (t.pnl is not None and t.pnl > 0) else "red"
        ax.plot(t.entry_time, eq.loc[t.entry_time] if t.entry_time in eq.index else np.nan,
                marker="v", color=color, markersize=10, alpha=0.7)

    ax.set_title(f"Trades over Equity ({len(buys)} entries)")
    ax.set_ylabel("Equity ($)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plotly (интерактивный HTML)
# ---------------------------------------------------------------------------


def create_plotly_report(
    equity_curve: pd.DataFrame,
    trades: List[Trade],
    metrics: PerformanceMetrics,
    save_path: pathlib.Path,
) -> None:
    """
    Создать интерактивный HTML-отчёт с plotly.

    Включает:
    - Equity curve + drawdown
    - Monthly returns heatmap
    - Returns distribution
    - Trade details table
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Subplots ---
    # Явно указываем типы сабплотов: xy, xy, xy, table
    # Это предотвращает ошибку "Trace type 'table' is not compatible with subplot type 'xy'"
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            "Equity Curve & Drawdown",
            "Monthly Returns Heatmap",
            "Returns Distribution",
            "Trade Details",
        ),
        vertical_spacing=0.08,
        row_heights=[0.35, 0.2, 0.2, 0.25],
        specs=[
            [{"type": "xy"}],       # Row 1: Equity + heatmap
            [{"type": "xy"}],       # Row 2: Drawdown
            [{"type": "xy"}],       # Row 3: Distribution (histogram)
            [{"type": "table"}],    # Row 4: Trade table
        ],
    )

    eq = equity_curve["equity"]
    dates = equity_curve.index

    # 1. Equity curve
    fig.add_trace(
        go.Scatter(
            x=dates, y=eq,
            name="Equity",
            line=dict(color="steelblue", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(70,130,180,0.1)",
        ),
        row=1, col=1,
    )

    # Drawdown
    rolling_max = eq.cummax()
    dd = (eq - rolling_max) / rolling_max * 100
    fig.add_trace(
        go.Scatter(
            x=dates, y=dd,
            name="Drawdown",
            fill="tozeroy",
            fillcolor="rgba(255,99,72,0.3)",
            line=dict(color="coral", width=1),
        ),
        row=2, col=1,
    )

    # 2. Monthly heatmap (отдельный subplot в строке 1)
    # Генерируем отдельно
    monthly = _get_monthly_returns(equity_curve)
    if not monthly.empty:
        heatmap = go.Heatmap(
            z=monthly.values,
            x=monthly.columns,
            y=monthly.index.astype(str),
            colorscale="RdYlGn",
            zmid=0,
            colorbar=dict(title="Return"),
        )
        fig.add_trace(heatmap, row=1, col=1)

    # 3. Returns distribution
    if not eq.empty:
        returns = eq.pct_change().dropna() * 100
        fig.add_trace(
            go.Histogram(
                x=returns.values,
                nbinsx=50,
                name="Distribution",
                marker_color="steelblue",
                opacity=0.7,
            ),
            row=3, col=1,
        )

    # 4. Trade summary (table)
    closed = [t for t in trades if t.is_closed and t.exit_price is not None]
    if closed:
        trade_text = "Trade Table"
        for t in closed[:10]:  # первые 10 сделок
            pnl_str = f"${t.pnl:,.2f}" if t.pnl else "N/A"
            trade_text += f"<br><b>{t.symbol}</b> {t.entry_time.strftime('%Y-%m-%d')}: "
            trade_text += f"entry=${t.entry_price:,.2f} → exit=${t.exit_price:,.2f} | "
            trade_text += f"PnL={pnl_str} ({t.pnl_pct:.1f}%) [{t.exit_reason}]"

        fig.add_trace(
            go.Table(
                header=dict(values=["Trade History"], align="left"),
                cells=dict(values=[[trade_text]], align="left"),
            ),
            row=4, col=1,
        )

    # --- Layout ---
    fig.update_layout(
        title="<b>Backtest Report — Range Trading with Trend Filter</b>",
        height=1200,
        showlegend=False,
        template="plotly_white",
    )

    # Update axes
    for i, row_idx in enumerate([1, 2, 3]):
        fig.update_xaxes(
            tickformat="%Y-%m",
            tickangle=30,
            row=row_idx, col=1,
        )

    fig.write_html(str(save_path), full_html=True, include_plotlyjs="cdn")
    logger.info("HTML report saved to %s", save_path)


def _get_monthly_returns(equity_curve: pd.DataFrame) -> pd.DataFrame:
    """Вычислить monthly returns."""
    daily = equity_curve["equity"].pct_change().dropna()
    daily.index.name = "date"
    monthly = daily.groupby(
        [daily.index.year, daily.index.month]
    ).apply(lambda x: (1 + x).prod() - 1)
    result = monthly.unstack(level=1)
    result.columns = [str(c) for c in result.columns]
    return result


# ---------------------------------------------------------------------------
# Пакетный экспорт
# ---------------------------------------------------------------------------


def export_all_visualizations(
    equity_curve: pd.DataFrame,
    trades: List[Trade],
    metrics: PerformanceMetrics,
    results_dir: str,
    generate_html: bool = True,
    save_png: bool = True,
) -> None:
    """
    Экспортировать все визуализации.

    Параметры
    ---------
    equity_curve : pd.DataFrame
    trades : List[Trade]
    metrics : PerformanceMetrics
    results_dir : str — директория для файлов
    generate_html : bool
    save_png : bool
    """
    out = pathlib.Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)

    if save_png:
        plot_equity_curve(equity_curve, metrics, save_path=out / "equity_curve.png")
        plot_trades_on_equity(equity_curve, trades, save_path=out / "trades_over_equity.png")
        logger.info("PNG charts saved to %s", out)

    if generate_html:
        try:
            create_plotly_report(
                equity_curve, trades, metrics,
                save_path=out / "backtest_report.html",
            )
        except ImportError:
            logger.warning("plotly not available, skipping HTML report")
        except Exception as e:
            logger.error("Failed to create HTML report: %s", e)
