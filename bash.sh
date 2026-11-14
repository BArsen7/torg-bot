#!/bin/bash
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}🛡️  Создание резервных копий...${NC}"
cp src/strategy.py src/strategy.py.ultimate_bak
cp src/data_loader.py src/data_loader.py.ultimate_bak

echo -e "${YELLOW}🔧 Применение исправлений бухгалтерии фьючерсов...${NC}"

python3 << 'PYTHON_SCRIPT'
import os

def replace_in_file(filepath, old_str, new_str, desc):
    if not os.path.exists(filepath):
        return
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    if old_str in content:
        content = content.replace(old_str, new_str)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ {desc}")
    else:
        print(f"⚠️  Не найдено: {desc}")

# 1. ИСПРАВЛЕНИЕ БУХГАЛТЕРИИ: Списание МАРЖИ при открытии, а не полной стоимости
old_entry_cash = """                # --- Вычитаем стоимость позиции из кэша (ВАЖНО!) ---
                cash -= position_cost"""

new_entry_cash = """                # --- ИСПРАВЛЕНО: Вычитаем МАРЖУ, а не полную стоимость (фьючерсы) ---
                margin_cost = position_cost / strategy_cfg.leverage
                cash -= margin_cost
                logger.info(f"MARGIN DEDUCTED: {margin_cost:.2f} (Lev: {strategy_cfg.leverage}x). Remaining cash: {cash:.2f}")"""

replace_in_file('src/strategy.py', old_entry_cash, new_entry_cash, "Fixed entry cash deduction (margin only)")

# 2. ИСПРАВЛЕНИЕ БУХГАЛТЕРИИ: Возврат МАРЖИ + PnL при закрытии
old_exit_cash = """            # --- Обновляем кэш: при закрытии_long возвращаем cost + PnL ---
            # Раньше: cash += exit_price * size (это cost + PnL)
            # Но cost уже был вычтен при входе → cash += exit_price * size верно
            cash += exit_price * trade.entry_size"""

new_exit_cash = """            # --- ИСПРАВЛЕНО: Возвращаем МАРЖУ + PnL при закрытии фьючерса ---
            margin_returned = (trade.entry_price * trade.entry_size) / strategy_cfg.leverage
            cash += margin_returned + gross_pnl - commission"""

replace_in_file('src/strategy.py', old_exit_cash, new_exit_cash, "Fixed exit cash return (margin + pnl)")

# Также исправим закрытие при HALT (там та же логика)
old_halt_cash = """                    trade.is_closed = True
                    cash += row["close"] * trade.entry_size
                    result.trades.append(trade)"""

new_halt_cash = """                    trade.is_closed = True
                    # Возвращаем маржу + PnL при принудительном закрытии
                    margin_returned = (trade.entry_price * trade.entry_size) / strategy_cfg.leverage
                    cash += margin_returned + pnl
                    result.trades.append(trade)"""

replace_in_file('src/strategy.py', old_halt_cash, new_halt_cash, "Fixed halt cash return (margin + pnl)")

# 3. ИСПРАВЛЕНИЕ NameError в data_loader.py
old_fallback_def = """_FALLBACK_SYMBOLS_SPOT = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "MATIC/USDT",
    "DOT/USDT", "UNI/USDT", "ATOM/USDT", "LTC/USDT", "ETC/USDT",
    "FIL/USDT", "APT/USDT", "ARBN/USDT", "NEAR/USDT", "ALGO/USDT",
    "FTM/USDT", "SAND/USDT", "MANA/USDT", "AXS/USDT", "THETA/USDT",
    "ICP/USDT", "VET/USDT", "HBAR/USDT", "EGLD/USDT", "XTZ/USDT",
]
_FALLBACK_SYMBOLS_FUTURES = [s + ":USDT" for s in _FALLBACK_SYMBOLS_SPOT]"""

# Упростим до одного списка с динамическим добавлением :USDT, чтобы избежать NameError
new_fallback_def = """_FALLBACK_SYMBOLS_BASE = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "MATIC/USDT",
    "DOT/USDT", "UNI/USDT", "ATOM/USDT", "LTC/USDT", "ETC/USDT",
    "FIL/USDT", "APT/USDT", "ARBN/USDT", "NEAR/USDT", "ALGO/USDT",
    "FTM/USDT", "SAND/USDT", "MANA/USDT", "AXS/USDT", "THETA/USDT",
    "ICP/USDT", "VET/USDT", "HBAR/USDT", "EGLD/USDT", "XTZ/USDT",
]"""

replace_in_file('src/data_loader.py', old_fallback_def, new_fallback_def, "Fixed fallback symbols definition")

old_fallback_ret = """        # Fallback: если API не доступен — используем хардкод
        logger.warning("All attempts failed, using fallback symbols")
        top = self.cfg.data.top_coins
        symbols = _FALLBACK_SYMBOLS_FUTURES[:top] if self.cfg.data.market_type == "futures" else _FALLBACK_SYMBOLS_SPOT[:top]
        logger.info("Using fallback: %d symbols", len(symbols))
        return symbols"""

new_fallback_ret = """        # Fallback: если API не доступен — используем хардкод
        logger.warning("All attempts failed, using fallback symbols")
        top = self.cfg.data.top_coins
        base_symbols = _FALLBACK_SYMBOLS_BASE[:top]
        # Для фьючерсов CCXT требует суффикс :USDT
        symbols = [s + ":USDT" if not s.endswith(":USDT") else s for s in base_symbols] if self.cfg.data.market_type == "futures" else base_symbols
        logger.info("Using fallback: %d symbols", len(symbols))
        return symbols"""

replace_in_file('src/data_loader.py', old_fallback_ret, new_fallback_ret, "Fixed fallback return logic")

print("\n🎉 Бухгалтерия фьючерсов исправлена!")
PYTHON_SCRIPT

echo -e "${GREEN}🚀 Готово! Теперь выполни:${NC}"
echo -e "  rm -rf data_cache/"
echo -e "  python src/main.py"
echo -e "${YELLOW}💡 Убедись, что в config.yaml указано leverage: 5 (или 10). При leverage=1 риск 2% со стопом 2% потребует 100% маржи, что корректно, но выглядит страшно.${NC}"
