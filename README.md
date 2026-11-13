# Backtest Framework — Range Trading with Trend Filter

## Описание

Полнофункциональный бэктест-фреймворк для криптовалютной стратегии
**Range Trading с трендовым фильтром**.

### Архитектура

```
config.yaml          — все параметры стратегии и бэктеста
config.py            — загрузка и валидация конфигурации
data_loader.py       — загрузка данных с Binance через ccxt, кэширование в Parquet
indicators.py        — технический анализ: ATR, RSI, свинг-пивоты, каналы
strategy.py          — логика сигналов входа/выхода (Range Trading + Trend Filter)
risk_manager.py      — позиционирование, корреляционный фильтр, лимиты
backtest_engine.py   — движок на основе backtrader
metrics.py           — расчёт всех ключевых метрик
visualization.py     — графики: equity curve, drawdown, сделки
main.py              — точка входа
```

### Установка

```bash
# Виртуальное окружение (рекомендуется)
python -m venv .venv
source .venv/bin/activate

# Зависимости
pip install -r requirements.txt
```

### Запуск

```bash
# Запуск по умолчанию (1h timeframe, последние 2 года, 30 монет)
python main.py

# С указанием конфига
python main.py --config config.yaml

# Изменение параметров
python main.py --timeframe 4h --top-coins 20 --years 1

# Загрузка данных без бэктеста
python main.py --download-only

# Генерация отчёта без скачивания (использует кэш)
python main.py --skip-download
```

### Выход

Результаты сохраняются в папке `results/`:
- `metrics_report.txt` — текстовый отчёт с метриками
- `trades.csv` — история всех сделок
- `equity_curve.csv` — кривая капитала
- `backtest_report.html` — интерактивный HTML-отчёт (plotly)

### Стратегия

1. **Тренд**: определение через свинг-пивоты на дневном и недельном ТФ
2. **Канал**: ATR-каналы (SMA ± k*ATR) или Donchian channel
3. **Вход LONG**: отскок от нижней границы канала при подтверждении тренда
4. **Выход**: тейк на верхней границе, стоп 1.8*ATR, трейлинг
5. **Риск-менеджмент**: 2% на сделку, макс 4 позиции, корреляция к BTC

### Тесты

```bash
pytest tests/ -v
```

### ⚠️ Дисклеймер

Этот инструмент предназначен исключительно для образовательных и исследовательских целей.
Прошлые результаты не гарантируют будущей доходности. Торговля криптовалютами связана
с высоким риском потери капитала.
