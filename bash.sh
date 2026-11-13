#!/bin/bash
set -e # Остановить выполнение при любой ошибке

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🛡️  Проверка окружения...${NC}"
if [ ! -f "src/main.py" ]; then
    echo -e "${RED}❌ Ошибка: Запусти скрипт из корневой директор проекта (где лежит src/main.py)${NC}"
    exit 1
fi

echo -e "${YELLOW}🛡️  Создание резервных копий (.bak)...${NC}"
cp src/risk_manager.py src/risk_manager.py.bak
cp src/data_loader.py src/data_loader.py.bak
cp src/strategy.py src/strategy.py.bak
echo -e "${GREEN}✅ Бэкапы созданы.${NC}"

echo -e "${YELLOW}🔧 Применение исправлений через Python-патчер...${NC}"

python3 << 'PYTHON_SCRIPT'
import re
import os

def patch_file(filepath, old_str, new_str, description):
    if not os.path.exists(filepath):
        print(f"⚠️  Файл {filepath} не найден, пропускаем.")
        return
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if old_str in content:
        content = content.replace(old_str, new_str)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ {description}")
    else:
        print(f"⚠️  Не удалось найти целевой код в {filepath}. Возможно, он уже изменён или отличается.")

# ---------------------------------------------------------
# 1. Исправление risk_manager.py (Критический баг sizing)
# ---------------------------------------------------------
old_risk = "position_size = (free_equity * risk_per_trade * entry_price) / price_distance"
new_risk = """# ИСПРАВЛЕНО: Убран лишний entry_price, добавлена проверка маржи
    risk_amount = free_equity * risk_per_trade
    position_size = risk_amount / price_distance
    
    required_margin = (position_size * entry_price) / leverage
    if required_margin > free_equity:
        position_size = (free_equity * leverage) / entry_price
        if leverage > 1:
            logger.warning(f"Stop too wide for margin. Req: {required_margin:.2f} > Equity: {free_equity:.2f}")
            return 0.0"""

patch_file('src/risk_manager.py', old_risk, new_risk, "Fixed src/risk_manager.py (position sizing)")

# ---------------------------------------------------------
# 2. Исправление data_loader.py (Фьючерсы вместо спота)
# ---------------------------------------------------------
with open('src/data_loader.py', 'r', encoding='utf-8') as f:
    content = f.read()

if "'defaultType': 'future'" not in content:
    # Заменяем инициализацию exchange
    content = content.replace(
        "exchange = ccxt.binance({",
        "exchange = ccxt.binance({'options': {'defaultType': 'future'}} #"
    )
    content = content.replace(
        "exchange = ccxt.binance_async({",
        "exchange = ccxt.binance_async({'options': {'defaultType': 'future'}} #"
    )
    
    # Пытаемся добавить exchange.close() в конец функции загрузки, если это async
    if 'await exchange.close()' not in content and 'exchange.close()' not in content:
        # Добавляем в блок finally или в конец функции fetch (best effort)
        content = re.sub(
            r"(return df\n)",
            r"\1    finally:\n        await exchange.close()\n",
            content
        )
    
    with open('src/data_loader.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Fixed src/data_loader.py (added futures defaultType & close)")
else:
    print("ℹ️  src/data_loader.py уже содержит настройки фьючерсов.")

# ---------------------------------------------------------
# 3. Исправление strategy.py (Корректное закрытие при Halt)
# ---------------------------------------------------------
with open('src/strategy.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Ищем блок с halted и break, чтобы заменить его на корректное закрытие
# Используем regex, так как форматирование может отличаться
old_halt_pattern = r"(logger\.(?:error|warning)\(f?[\"']Halted due to max drawdown limit.*?\n)(\s*break)"

new_halt_logic = r"""\1
        # !!! ПРОВЕРЬ ИМЕНА ПЕРЕМЕННЫХ НИЖЕ (open_positions, current_time, current_close, closed_trades) !!!
        for sym, open_trade in list(open_positions.items()):
            open_trade.exit_time = current_time
            open_trade.exit_price = current_close
            open_trade.exit_reason = 'halt'
            open_trade.is_closed = True
            
            raw_pnl = open_trade.size * (current_close - open_trade.entry_price)
            open_trade.pnl = raw_pnl - open_trade.commission
            open_trade.pnl_pct = (open_trade.pnl / (open_trade.size * open_trade.entry_price)) * 100 if open_trade.entry_price > 0 else 0.0
            
            closed_trades.append(open_trade)
            current_equity += open_trade.pnl
            
        logger.warning(f"Emergency closed positions. Final equity: {current_equity:.2f}")
        \2"""

if re.search(old_halt_pattern, content, re.IGNORECASE):
    content = re.sub(old_halt_pattern, new_halt_logic, content, flags=re.IGNORECASE)
    with open('src/strategy.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Fixed src/strategy.py (halt position closing)")
else:
    print("⚠️  Не удалось автоматически исправить src/strategy.py.")
    print("   Открой src/strategy.py, найди 'Halted due to max drawdown' и замени 'break' на код из инструкции.")

print("\n🎉 Патчинг завершён!")
PYTHON_SCRIPT

echo -e "${GREEN}🚀 Готово! Теперь выполни:${NC}"
echo -e "  rm -rf data_cache/"
echo -e "  python src/main.py"
echo -e "${YELLOW}💡 Совет: Открой src/strategy.py и проверь блок с комментарием '!!! ПРОВЕРЬ ИМЕНА ПЕРЕМЕННЫХ'${NC}"
