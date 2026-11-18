#!/usr/bin/env python3
"""
Grid Search для оптимизации параметров стратегии Range Trading.

Ключевые особенности:
1. Данные загружаются ОДИН РАЗ в начале (экономия времени)
2. Для каждой комбинации создаётся ГЛУБОКАЯ КОПИЯ данных (предотвращает модификацию)
3. В логах видно, какие параметры применяются и что результат меняется
4. Защита от мусорных результатов (слишком мало сделок)
5. Красивый вывод с прогрессом и ETA
"""

import copy
import itertools
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.config import Config
from src.data_loader import DataLoader
from src.indicators import add_all_indicators
from src.strategy import generate_signals, simulate_trading
from src.metrics import compute_metrics

# Настраиваем логирование: подавляем лишние сообщения, но оставляем важные
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("src.data_loader").setLevel(logging.WARNING)
logging.getLogger("src.backtest_engine").setLevel(logging.WARNING)

@dataclass
class GridResult:
    """Результат одной комбинации параметров."""
    params: Dict
    total_trades: int
    total_return: float
    max_dd_pct: float
    calmar: float
    sharpe: float
    win_rate: float
    profit_factor: float
    avg_holding_bars: float
    commissions: float

def run_single_backtest(
    params: Dict,
    raw_data: Dict[str, pd.DataFrame],
    base_cfg: Config
) -> GridResult:
    """
    Запускает бэктест с заданными параметрами.
    
    ВАЖНО: Создаёт глубокую копию данных, чтобы не модифицировать оригинал.
    """
    # Создаём глубокую копию конфигурации
    cfg = copy.deepcopy(base_cfg)
    
    # Применяем параметры из сетки
    cfg.strategy.rsi_threshold = params["rsi_threshold"]
    cfg.strategy.stop_multiplier = params["stop_multiplier"]
    cfg.strategy.atr_pct_min = params["atr_pct_min"]
    cfg.strategy.atr_pct_max = params["atr_pct_max"]
    cfg.strategy.risk_per_trade = params["risk_per_trade"]
    
    # Создаём ГЛУБОКУЮ КОПИЮ данных для этой комбинации
    # Это критически важно, т.к. simulate_trading модифицирует DataFrame inplace
    symbols_data = {}
    for symbol, df in raw_data.items():
        df_copy = df.copy(deep=True)
        
        # Рассчитываем индикаторы с текущими параметрами
        df_with_indicators = add_all_indicators(df_copy, cfg.strategy)
        
        # Генерируем сигналы
        df_with_signals = generate_signals(
            df_with_indicators, 
            cfg.strategy, 
            fee_rate=cfg.strategy.maker_fee
        )
        
        symbols_data[symbol] = df_with_signals
    
    # Запускаем симуляцию
    funding_pct = 0.0
    if cfg.data.market_type == "futures" and not cfg.strategy.use_real_funding:
        funding_pct = cfg.strategy.fixed_funding_pct
    
    result = simulate_trading(
        symbols_data,
        cfg.strategy,
        cfg.backtest,
        fee_rate=cfg.strategy.maker_fee,
        slippage_pct=cfg.strategy.slippage_pct,
        funding_pct=funding_pct,
    )
    
    # Считаем метрики
    metrics = compute_metrics(
        result.trades,
        result.equity_curve,
        cfg.backtest.initial_cash,
        timeframe=cfg.data.timeframe,
        final_equity_override=result.final_equity
    )
    
    return GridResult(
        params=params,
        total_trades=metrics.total_trades,
        total_return=metrics.total_return,
        max_dd_pct=metrics.max_drawdown_pct,
        calmar=metrics.calmar_ratio,
        sharpe=metrics.sharpe_ratio,
        win_rate=metrics.win_rate,
        profit_factor=metrics.profit_factor,
        avg_holding_bars=metrics.avg_holding_bars,
        commissions=metrics.total_commissions,
    )

def main():
    print("=" * 80)
    print("🚀 GRID SEARCH ОПТИМИЗАЦИЯ ПАРАМЕТРОВ СТРАТЕГИИ")
    print("=" * 80)
    print()
    
    # 1. Загружаем конфигурацию
    print("📋 Загрузка конфигурации из config.yaml...")
    base_cfg = Config()
    print(f"   ✓ Таймфрейм: {base_cfg.data.timeframe}")
    print(f"   ✓ Монет: {base_cfg.data.top_coins}")
    print(f"   ✓ Период: {base_cfg.data.data_years} лет(года)")
    print(f"   ✓ Плечо: {base_cfg.strategy.leverage}x")
    print()
    
    # 2. Загружаем данные ОДИН РАЗ
    print("⏳ Загрузка исторических данных с Bybit (это может занять 1-2 минуты)...")
    data_loader = DataLoader(base_cfg)
    raw_data = data_loader.load_for_backtest()
    
    if not raw_data:
        print("❌ Ошибка: не удалось загрузить данные. Проверьте интернет и config.yaml.")
        return
    
    print(f"✅ Данные загружены: {len(raw_data)} символов")
    for symbol, df in list(raw_data.items())[:3]:
        print(f"   • {symbol}: {len(df)} свечей ({df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')})")
    if len(raw_data) > 3:
        print(f"   ... и ещё {len(raw_data) - 3} символов")
    print()
    
    # 3. Определяем сетку параметров
    param_grid = {
        "rsi_threshold": [30, 35, 40],           # Порог RSI для входа
        "stop_multiplier": [1.5, 1.8, 2.0],      # Множитель ATR для стопа
        "atr_pct_min": [0.5, 1.0, 1.5],          # Минимальная волатильность ATR%
        "atr_pct_max": [4.0, 6.0, 8.0],          # Максимальная волатильность ATR%
        "risk_per_trade": [0.01, 0.015, 0.02],   # Риск на сделку (1%, 1.5%, 2%)
    }
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    print("🔍 Сетка параметров для оптимизации:")
    for key, vals in param_grid.items():
        print(f"   • {key}: {vals}")
    print(f"\n📊 Всего комбинаций: {len(combinations)}")
    print()
    
    # 4. Запускаем оптимизацию
    print("=" * 80)
    print("🏃 НАЧАЛО ОПТИМИЗАЦИИ")
    print("=" * 80)
    print()
    
    results: List[GridResult] = []
    start_time = time.time()
    
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        
        # Выводим информацию о текущей комбинации
        print(f"[{i+1:3d}/{len(combinations)}]", end=" ")
        print(f"RSI<{params['rsi_threshold']:2d} ", end="")
        print(f"Stop={params['stop_multiplier']:.1f}x ", end="")
        print(f"ATR%[{params['atr_pct_min']:.1f}-{params['atr_pct_max']:.1f}] ", end="")
        print(f"Risk={params['risk_per_trade']*100:.1f}% ... ", end="", flush=True)
        
        try:
            # Запускаем бэктест
            result = run_single_backtest(params, raw_data, base_cfg)
            results.append(result)
            
            # Выводим результат
            print(f"✅ Сделок: {result.total_trades:3d} | ", end="")
            print(f"Return: {result.total_return:>6.1%} | ", end="")
            print(f"MaxDD: {result.max_dd_pct:>5.1%} | ", end="")
            print(f"Calmar: {result.calmar:.3f} | ", end="")
            print(f"PF: {result.profit_factor:.2f}")
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
        
        # Показываем ETA каждые 10 итераций
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            eta = avg_time * (len(combinations) - i - 1)
            print(f"       ⏱️  Прошло: {elapsed/60:.1f} мин | ETA: {eta/60:.1f} мин")
            print()
    
    total_time = time.time() - start_time
    print()
    print("=" * 80)
    print(f"✅ ОПТИМИЗАЦИЯ ЗАВЕРШЕНА за {total_time/60:.1f} минут")
    print("=" * 80)
    print()
    
    # 5. Анализ результатов
    if not results:
        print("❌ Не получено ни одного результата.")
        return
    
    df = pd.DataFrame([
        {
            **r.params,
            "total_trades": r.total_trades,
            "total_return": r.total_return,
            "max_dd_pct": r.max_dd_pct,
            "calmar": r.calmar,
            "sharpe": r.sharpe,
            "win_rate": r.win_rate,
            "profit_factor": r.profit_factor,
            "avg_holding_bars": r.avg_holding_bars,
            "commissions": r.commissions,
        }
        for r in results
    ])
    
    # Фильтруем заведомо плохие результаты
    filtered_df = df[
        (df["total_trades"] >= 30) &       # Минимум 30 сделок для статистики
        (df["max_dd_pct"] <= 0.40) &       # Максимальная просадка не более 40%
        (df["profit_factor"] >= 1.05)      # Стратегия должна быть прибыльной
    ].copy()
    
    if filtered_df.empty:
        print("⚠️  НИ ОДНА конфигурация не прошла базовые фильтры:")
        print("   • Минимум 30 сделок")
        print("   • Максимальная просадка <= 40%")
        print("   • Profit Factor >= 1.05")
        print()
        print("💡 Показываю ТОП-5 по Calmar Ratio без фильтров:")
        print()
        
        top5 = df.sort_values("calmar", ascending=False).head(5)
        for idx, row in top5.iterrows():
            print(f"   RSI<{int(row['rsi_threshold'])} ", end="")
            print(f"Stop={row['stop_multiplier']:.1f}x ", end="")
            print(f"ATR%[{row['atr_pct_min']:.1f}-{row['atr_pct_max']:.1f}] ", end="")
            print(f"Risk={row['risk_per_trade']*100:.1f}%")
            print(f"   → Сделок: {int(row['total_trades'])} | ", end="")
            print(f"Return: {row['total_return']:.1%} | ", end="")
            print(f"MaxDD: {row['max_dd_pct']:.1%} | ", end="")
            print(f"Calmar: {row['calmar']:.3f} | ", end="")
            print(f"PF: {row['profit_factor']:.2f}")
            print()
    else:
        # Сортируем по Calmar Ratio (доходность с поправкой на риск)
        filtered_df = filtered_df.sort_values("calmar", ascending=False)
        
        print("🏆 ТОП-10 НАИБОЛЕЕ УСТОЙЧИВЫХ КОНФИГУРАЦИЙ:")
        print()
        
        for rank, (idx, row) in enumerate(filtered_df.head(10).iterrows(), 1):
            print(f"#{rank:2d} | Calmar: {row['calmar']:.3f}")
            print(f"    Параметры: RSI<{int(row['rsi_threshold'])} | ", end="")
            print(f"Stop={row['stop_multiplier']:.1f}x | ", end="")
            print(f"ATR%[{row['atr_pct_min']:.1f}-{row['atr_pct_max']:.1f}] | ", end="")
            print(f"Risk={row['risk_per_trade']*100:.1f}%")
            print(f"    Метрики:   Return: {row['total_return']:.1%} | ", end="")
            print(f"MaxDD: {row['max_dd_pct']:.1%} | ", end="")
            print(f"Sharpe: {row['sharpe']:.2f} | ", end="")
            print(f"PF: {row['profit_factor']:.2f} | ", end="")
            print(f"WR: {row['win_rate']:.1%} | ", end="")
            print(f"Сделок: {int(row['total_trades'])}")
            print()
    
    # 6. Сохраняем полный отчёт в CSV
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "grid_search_results.csv"
    df_sorted = df.sort_values("calmar", ascending=False)
    df_sorted.to_csv(output_file, index=False)
    
    print("=" * 80)
    print(f"💾 Полный отчёт сохранён в: {output_file}")
    print("=" * 80)
    print()
    print("📊 Статистика по всем комбинациям:")
    print(f"   • Всего протестировано: {len(results)}")
    print(f"   • Прошли фильтры: {len(filtered_df)}")
    print(f"   • Средний Return: {df['total_return'].mean():.1%}")
    print(f"   • Средний MaxDD: {df['max_dd_pct'].mean():.1%}")
    print(f"   • Средний Calmar: {df['calmar'].mean():.3f}")
    print()

if __name__ == "__main__":
    main()
