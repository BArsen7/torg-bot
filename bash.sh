#!/bin/bash
set -e

FILE="src/data_loader.py"

if [ ! -f "$FILE" ]; then
    echo "❌ Ошибка: Файл $FILE не найден. Запустите скрипт из корня проекта (~/python/torg-bot/)."
    exit 1
fi

echo "🛡️  Создание резервной копии $FILE.bak..."
cp "$FILE" "$FILE.bak"

echo "🔧 Применение исправления пагинации Bybit V5..."

python3 << 'PYTHON_SCRIPT'
import re

with open("src/data_loader.py", "r", encoding="utf-8") as f:
    content = f.read()

# Находим старую функцию _fetch_ohlcv_paginated и заменяем её на правильную
old_func_pattern = r'async def _fetch_ohlcv_paginated\([\s\S]*?return unique_ohlcv'

new_func = '''async def _fetch_ohlcv_paginated(
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

    return unique_ohlcv'''

if re.search(old_func_pattern, content):
    content = re.sub(old_func_pattern, new_func, content)
    with open("src/data_loader.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ Функция _fetch_ohlcv_paginated успешно заменена на корректную версию с использованием параметра 'end'.")
else:
    print("⚠️  Не удалось найти старую функцию _fetch_ohlcv_paginated для замены.")
    print("💡 Возможно, она уже была изменена или форматирование отличается.")

PYTHON_SCRIPT

echo "🚀 Готово! Теперь выполни:"
echo "  rm -rf data_cache/"
echo "  python src/main.py"
echo "💡 Теперь загрузка данных будет идти от настоящего к прошлому, что гарантирует корректную пагинацию Bybit V5 без зацикливания."
