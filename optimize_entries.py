#!/usr/bin/env python3
"""
Оптимизатор параметров условий входа (Entry Conditions Optimizer).

КЛЮЧЕВОЕ ОТЛИЧИЕ: данные загружаются ОДИН РАЗ, затем прогоняются все комбинации.
Это ускоряет оптимизацию в 100+ раз (минуты вместо часов).
"""

import copy
import itertools
import logging
import time
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

# Подавляем лишние логи во время оптимизации
logging.getLogger("src.backtest_engine").setLevel(logging.WARNING)
logging.getLogger("src.strategy").setLevel(logging.WARNING)
logging.getLogger("src.data_loader").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

from src.backtest_engine import BacktestEngine
from src.config import Config
from src.data_loader import DataLoader
from src.indicators import add_all_indicators
from src.strategy import generate_signals, simulate_trading
from src.metrics import compute_metrics

@dataclass
class OptResult:
    params: Dict
    total_return: float
    sharpe: float
    max_dd_pct: float
    calmar: float
    win_rate: float
    profit_factor: float
    total_trades: int
    robustness_score: float

def run_single_optimization(
    params: Dict, 
    symbols_data: Dict[str, pd.DataFrame],
    base_config: Config,
) -> OptResult:
    """
    Запускает симуляцию с заданными параметрами на УЖЕ ЗАГРУЖЕННЫХ данных.
    """
    cfg = copy.deepcopy(base_config)
    
    # Применяем параметры из сетки
    cfg.strategy.rsi_threshold = params["rsi_threshold"]
    cfg.strategy.adx_threshold = params["adx_threshold"]
    cfg.strategy.volume_multiplier = params.get("volume_multiplier", 1.0)
    cfg.strategy.atr_pct_min = params["atr_pct_min"]
    cfg.strategy.atr_pct_max = params["atr_pct_max"]
    
    # === ШАГ 1: Индикаторы (уже посчитаны, но пересчитываем для чистоты) ===
    # На самом деле индикаторы не зависят от параметров входа, 
    # поэтому их можно посчитать один раз. Но для безопасности пересчитаем.
    for symbol in symbols_data:
        symbols_data[symbol] = add_all_indicators(symbols_data[symbol], cfg.strategy)
    
    # === ШАГ 2: Сигналы ===
    for symbol in symbols_data:
        symbols_data[symbol] = generate_signals(
            symbols_data[symbol], cfg.strategy,
            fee_rate=cfg.strategy.maker_fee,
        )
    
    # === ШАГ 3: Симуляция ===
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
    
    # === ШАГ 4: Метрики ===
    m = compute_metrics(
        result.trades,
        result.equity_curve,
        cfg.backtest.initial_cash,
        timeframe=cfg.data.timeframe,
        final_equity_override=result.final_equity,
    )
    
    # Расчет оценки робастности
    trade_penalty = (m.total_trades / 100.0) ** 0.5 if m.total_trades > 0 else 0.1
    calmar = m.calmar_ratio if m.calmar_ratio > 0 else 0.0
    score = calmar * trade_penalty
    
    return OptResult(
        params=params,
        total_return=m.total_return,
        sharpe=m.sharpe_ratio,
        max_dd_pct=m.max_drawdown_pct,
        calmar=m.calmar_ratio,
        win_rate=m.win_rate,
        profit_factor=m.profit_factor,
        total_trades=m.total_trades,
        robustness_score=score
    )

def main():
    print("🚀 Запуск оптимизатора параметров входа...")
    print("📊 Данные будут загружены ОДИН раз, затем прогон всех комбинаций.\n")
    
    # 1. Базовая конфигурация
    base_config = Config()
    
    # 2. Загружаем данные ОДИН РАЗ
    print("[1/3] Загрузка данных...")
    data_loader = DataLoader(base_config)
    symbols_data = data_loader.load_for_backtest()
    
    if not symbols_data:
        print("❌ Не удалось загрузить данные. Проверьте config.yaml и подключение к Bybit.")
        return
    
    print(f"✅ Загружено {len(symbols_data)} символов\n")
    
    # 3. Сетка параметров
    param_grid = {
        "rsi_threshold": [30, 35, 40],
        "adx_threshold": [15, 20, 25],
        "volume_multiplier": [0.8, 1.0, 1.2],
        "atr_pct_min": [0.5, 1.0],
        "atr_pct_max": [4.0, 6.0, 8.0],
    }
    
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    print(f"[2/3] Всего комбинаций: {len(combinations)}")
    print("⏱️  Прогон начался...\n")
    
    results: List[OptResult] = []
    start_time = time.time()
    
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        
        # Индикатор прогресса
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (len(combinations) - i - 1)
            print(f"[{i+1}/{len(combinations)}] ETA: {eta:.0f}s ... ", end="", flush=True)
        
        try:
            # Передаём уже загруженные данные, а не создаём новый движок
            res = run_single_optimization(params, symbols_data, base_config)
            results.append(res)
            
            if (i + 1) % 10 == 0 or i == 0:
                print(f"✅ Сделок: {res.total_trades:3d} | Return: {res.total_return:>6.1%} | DD: {res.max_dd_pct:>5.1%} | Calmar: {res.calmar:.3f}")
        except Exception as e:
            if (i + 1) % 10 == 0 or i == 0:
                print(f"❌ Ошибка: {e}")
    
    elapsed = time.time() - start_time
    print(f"\n[3/3] Оптимизация завершена за {elapsed:.1f} секунд ({elapsed/60:.1f} минут)\n")
    
    # 4. Фильтрация и сортировка
    valid_results = [
        r for r in results 
        if r.total_trades >= 30 and r.profit_factor >= 1.0 and r.max_dd_pct <= 0.40
    ]
    
    valid_results.sort(key=lambda r: r.robustness_score, reverse=True)
    
    print("="*90)
    print("🏆 ТОП-10 НАИБОЛЕЕ УСТОЙЧИВЫХ КОНФИГУРАЦИЙ")
    print("="*90)
    
    if not valid_results:
        print("\n⚠️  Не найдено ни одной конфигурации, удовлетворяющей базовым фильтрам.")
        print("💡 Попробуйте расширить сетку параметров или смягчить фильтры.\n")
        # Показываем топ-5 по Calmar без фильтров
        all_sorted = sorted(results, key=lambda r: r.calmar, reverse=True)
        print("\n📊 ТОП-5 по Calmar Ratio (без фильтров):")
        for i, r in enumerate(all_sorted[:5], 1):
            print(f"#{i} | Params: {r.params}")
            print(f"   Return={r.total_return:.2%} | DD={r.max_dd_pct:.1%} | Calmar={r.calmar:.3f} | Trades={r.total_trades}")
    else:
        for i, r in enumerate(valid_results[:10], 1):
            print(f"\n#{i} | Score: {r.robustness_score:.3f}")
            print(f"   Параметры : RSI<{r.params['rsi_threshold']}, ADX>{r.params['adx_threshold']}, Vol>{r.params['volume_multiplier']}x, ATR% [{r.params['atr_pct_min']}-{r.params['atr_pct_max']}]")
            print(f"   Метрики   : Return={r.total_return:>6.1%} | MaxDD={r.max_dd_pct:>5.1%} | Calmar={r.calmar:.3f} | PF={r.profit_factor:.2f} | WR={r.win_rate:.1%} | Сделок={r.total_trades}")
    
    # 5. Сохранение в CSV
    if results:
        df = pd.DataFrame([{
            **r.params,
            "total_return": r.total_return,
            "sharpe": r.sharpe,
            "max_dd_pct": r.max_dd_pct,
            "calmar": r.calmar,
            "win_rate": r.win_rate,
            "profit_factor": r.profit_factor,
            "total_trades": r.total_trades,
            "robustness_score": r.robustness_score,
        } for r in results])
        
        out_path = "results/optimization_results.csv"
        df.to_csv(out_path, index=False)
        print(f"\n💾 Полный отчет сохранен в: {out_path}")

if __name__ == "__main__":
    main()
