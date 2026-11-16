#!/usr/bin/env python3
"""
Grid Search для оптимизации параметров стратегии Range Trading.
"""

import copy
import itertools
import logging
import time
from pathlib import Path

import pandas as pd

from src.config import Config
from src.backtest_engine import BacktestEngine

# Подавляем информационные логи во время оптимизации для чистоты вывода и скорости
logging.getLogger("src.backtest_engine").setLevel(logging.WARNING)
logging.getLogger("src.strategy").setLevel(logging.WARNING)
logging.getLogger("src.data_loader").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

def run_single_backtest(params: dict, base_cfg: Config) -> dict:
    """Запускает один прогон бэктеста с заданными параметрами."""
    # Глубокая копия конфигурации, чтобы не мутировать оригинал
    cfg = copy.deepcopy(base_cfg)
    
    # Применяем параметры из сетки
    cfg.strategy.rsi_threshold = params["rsi_threshold"]
    cfg.strategy.adx_threshold = params["adx_threshold"]
    cfg.strategy.stop_multiplier = params["stop_multiplier"]
    cfg.strategy.atr_pct_min = params["atr_pct_min"]
    cfg.strategy.atr_pct_max = params["atr_pct_max"]
    
    # Запускаем движок
    engine = BacktestEngine(cfg)
    engine.run()
    
    m = engine.metrics
    
    return {
        **params,
        "total_return": m.total_return,
        "max_dd_pct": m.max_drawdown_pct,
        "sharpe": m.sharpe_ratio,
        "calmar": m.calmar_ratio,
        "win_rate": m.win_rate,
        "profit_factor": m.profit_factor,
        "total_trades": m.total_trades,
    }

def main():
    print("🚀 Запуск Grid Search оптимизации...")
    print("⚠️  Убедитесь, что в config.yaml установлено top_coins=10 и data_years=1 для скорости!\n")
    
    # 1. Базовая конфигурация
    base_cfg = Config()
    
    # 2. Сетка параметров для перебора
    param_grid = {
        "rsi_threshold": [30, 35, 40],
        "adx_threshold": [15, 20, 25],
        "stop_multiplier": [1.5, 1.8, 2.0],
        "atr_pct_min": [0.5, 1.0],
        "atr_pct_max": [4.0, 6.0, 8.0],
    }
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    print(f"🔍 Всего комбинаций для проверки: {len(combinations)}")
    print("⏱️  Это может занять от 30 минут до нескольких часов в зависимости от мощности ПК.\n")
    
    results = []
    start_time = time.time()
    
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        
        # Индикатор прогресса
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1) if i > 0 else 10
        eta = avg_time * (len(combinations) - i - 1)
        
        print(f"[{i+1}/{len(combinations)}] ETA: {eta/60:.1f} мин | "
              f"RSI<{params['rsi_threshold']} ADX>{params['adx_threshold']} "
              f"Stop={params['stop_multiplier']}x ... ", end="", flush=True)
        
        try:
            res = run_single_backtest(params, base_cfg)
            results.append(res)
            
            # Краткий вывод результата
            print(f"✅ Return: {res['total_return']:.1%} | DD: {res['max_dd_pct']:.1%} | "
                  f"Calmar: {res['calmar']:.3f} | Trades: {res['total_trades']}")
        except Exception as e:
            print(f"❌ Ошибка: {e}")
    
    total_time = time.time() - start_time
    print(f"\n⏱️  Оптимизация завершена за {total_time/60:.1f} минут.\n")
    
    # 3. Обработка результатов
    if not results:
        print("⚠️  Не получено ни одного результата.")
        return
        
    df = pd.DataFrame(results)
    
    # Фильтрация заведомо плохих результатов
    # Минимум 50 сделок для статистической значимости
    # Максимальная просадка не более 30% (можно изменить под свои риски)
    filtered_df = df[
        (df["total_trades"] >= 50) & 
        (df["max_dd_pct"] <= 0.30) &
        (df["profit_factor"] >= 1.1)
    ].copy()
    
    if filtered_df.empty:
        print("⚠️  Не найдено ни одной конфигурации, удовлетворяющей базовым фильтрам:")
        print("   (Min 50 trades, Max DD <= 30%, Profit Factor >= 1.1)")
        print("💡 Совет: Попробуйте расширить сетку параметров или смягчить фильтры в коде.")
        
        print("\n📊 ТОП-5 по Calmar Ratio (без фильтров):")
        print(df.sort_values("calmar", ascending=False).head(5).to_string(index=False))
    else:
        # Расчет оценки робастности: Calmar * sqrt(кол-во сделок / 100)
        # Это поощряет как высокую доходность при низкой просадке, так и достаточное количество сделок
        filtered_df["robustness_score"] = filtered_df["calmar"] * ((filtered_df["total_trades"] / 100.0) ** 0.5)
        
        # Сортировка по оценке робастности (по убыванию)
        filtered_df = filtered_df.sort_values("robustness_score", ascending=False)
        
        print("="*100)
        print("🏆 ТОП-10 НАИБОЛЕЕ УСТОЙЧИВЫХ КОНФИГУРАЦИЙ")
        print("="*100)
        
        for idx, row in filtered_df.head(10).iterrows():
            print(f"\n#{idx+1} | Score: {row['robustness_score']:.3f}")
            print(f"   Параметры : RSI<{row['rsi_threshold']}, ADX>{row['adx_threshold']}, "
                  f"Stop={row['stop_multiplier']}x, ATR% [{row['atr_pct_min']}-{row['atr_pct_max']}]")
            print(f"   Метрики   : Return={row['total_return']:.1%} | MaxDD={row['max_dd_pct']:.1%} | "
                  f"Calmar={row['calmar']:.3f} | PF={row['profit_factor']:.2f} | WR={row['win_rate']:.1%} | "
                  f"Сделок={int(row['total_trades'])}")
            
    # 4. Сохранение полного отчета в CSV
    output_file = "results/grid_search_results.csv"
    Path("results").mkdir(parents=True, exist_ok=True)
    
    # Сортируем полный датафрейм по Calmar для удобства просмотра в Excel
    df = df.sort_values("calmar", ascending=False)
    df.to_csv(output_file, index=False)
    print(f"\n💾 Полный отчет сохранен в: {output_file}")

if __name__ == "__main__":
    main()
