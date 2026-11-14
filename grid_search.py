#!/usr/bin/env python3
"""
Grid Search для оптимизации параметров стратегии.
Перебирает комбинации параметров и выводит топ-10 результатов по Calmar Ratio.
"""
import itertools
import logging
import time
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from src.backtest_engine import BacktestEngine
from src.config import Config

# Подавляем информационные логи движка для ускорения Grid Search
logging.getLogger("src.backtest_engine").setLevel(logging.WARNING)
logging.getLogger("src.strategy").setLevel(logging.WARNING)

@dataclass
class GridResult:
    params: Dict
    total_return: float
    sharpe: float
    max_dd_pct: float
    calmar: float
    win_rate: float
    profit_factor: float
    total_trades: int

def run_single_backtest(params: Dict) -> GridResult:
    config = Config()
    config.strategy.rsi_threshold = params["rsi_threshold"]
    config.strategy.adx_threshold = params["adx_threshold"]
    config.strategy.stop_multiplier = params["stop_multiplier"]
    config.strategy.risk_per_trade = params["risk_per_trade"]
    
    engine = BacktestEngine(config)
    engine.run()
    
    return GridResult(
        params=params,
        total_return=engine.metrics.total_return,
        sharpe=engine.metrics.sharpe_ratio,
        max_dd_pct=engine.metrics.max_drawdown_pct,
        calmar=engine.metrics.calmar_ratio,
        win_rate=engine.metrics.win_rate,
        profit_factor=engine.metrics.profit_factor,
        total_trades=engine.metrics.total_trades,
    )

def main():
    param_grid = {
        "rsi_threshold": [35, 40, 45],       # Было 30, 35, 40. Разрешаем вход чуть выше.
        "adx_threshold": [15, 20, 25],       # Было 20, 25, 30. Разрешаем торговать при слабом тренде.
        "stop_multiplier": [1.5, 1.8, 2.0],  # Оставляем как есть, это дает хороший R/R.
        "risk_per_trade": [0.01, 0.015, 0.02]# Оставляем. 1-2% это золотой стандарт.
    }
    """
    param_grid = {
        "rsi_threshold": [30, 35, 40],
        "adx_threshold": [20, 25, 30],
        "stop_multiplier": [1.5, 1.8, 2.0],
        "risk_per_trade": [0.01, 0.015, 0.02],
    }
    """
    combinations = list(itertools.product(*param_grid.values()))
    print(f"🔍 Grid Search: {len(combinations)} комбинаций параметров")
    
    results: List[GridResult] = []
    start_time = time.time()
    
    for i, combo in enumerate(combinations):
        params = dict(zip(param_grid.keys(), combo))
        print(f"[{i+1}/{len(combinations)}] Testing: {params}", end=" ... ")
        
        try:
            result = run_single_backtest(params)
            results.append(result)
            print(f"✅ Return: {result.total_return:.2%}, DD: {result.max_dd_pct:.1%}, Calmar: {result.calmar:.3f}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n⏱️  Завершено за {time.time() - start_time:.1f} секунд")
    
    # Сортировка по Calmar Ratio (доходность с поправкой на риск)
    results.sort(key=lambda r: r.calmar, reverse=True)
    
    print("\n" + "="*80)
    print("🏆 ТОП-10 РЕЗУЛЬТАТОВ (по Calmar Ratio)")
    print("="*80)
    
    for i, r in enumerate(results[:10], 1):
        print(f"\n#{i} | Params: {r.params}")
        print(f"   Return: {r.total_return:.2%} | Max DD: {r.max_dd_pct:.1%} | Calmar: {r.calmar:.3f}")
        print(f"   Win Rate: {r.win_rate:.1%} | Profit Factor: {r.profit_factor:.3f} | Trades: {r.total_trades}")
    
    # Сохранение в CSV
    df = pd.DataFrame([{**r.params, **{k: v for k, v in r.__dict__.items() if k != 'params'}} for r in results])
    df.to_csv("results/grid_search_results.csv", index=False)
    print(f"\n💾 Результаты сохранены в results/grid_search_results.csv")

if __name__ == "__main__":
    main()
