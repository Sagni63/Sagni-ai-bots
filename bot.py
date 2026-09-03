import os
import json
import time
from datetime import datetime, timezone, timedelta
import requests
import pandas as pd
import numpy as np
import MetaTrader5 as mt5

# =========================================================
# CONFIGURATIONS
# =========================================================

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

JOURNAL_FILE = "trade_journal.json"
MEMORY_FILE = "signal_memory.json"
ERROR_LOG_FILE = "error_log.txt"

MAX_OPEN_TRADES = 2
COOLDOWN_HOURS = 4  

# የ MT5 Ticker Symbols Mapping (እንደ ብሮከርህ ስም ማስተካከል ትችላለህ)
SYMBOLS = {
    "XAU/USD": {"mt5_symbol": "XAUUSD", "type": "commodity"},
    "BTC/USD": {"mt5_symbol": "BTCUSD", "type": "crypto"},
    "GBP/USD": {"mt5_symbol": "GBPUSD", "type": "forex"},
    "USD/JPY": {"mt5_symbol": "USDJPY", "type": "forex"}
}


# =========================================================
# UTILITIES & FILE MANAGEMENT
# =========================================================

def send_telegram(message):
    if not BOT_TOKEN or not CHAT_ID:
        print("❌ Telegram BOT_TOKEN or CHAT_ID missing")
        return False
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        res = requests.post(url, json={"chat_id": CHAT_ID, "text": message}, timeout=15)
        return res.ok
    except Exception as e:
        print(f"❌ Telegram Error: {e}")
        return False

def load_json_file(filename, default_value):
    if not os.path.exists(filename):
        return default_value
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default_value

def save_json_file(filename, data):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"❌ File Save Error ({filename}): {e}")

def log_error(msg):
    try:
        with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat()}] {msg}\n")
    except Exception:
        pass


# =========================================================
# METATRADER 5 DATA & INDICATORS
# =========================================================

def get_mt5_data(mt5_symbol, tf_str, count=100):
    """ከ MT5 በቀጥታ ያለ ምንም መዘገየት መረጃ መውሰጃ"""
    tf_map = {
        "1m": mt5.TIMEFRAME_M1,
        "15m": mt5.TIMEFRAME_M15,
        "1h": mt5.TIMEFRAME_H1,
        "1d": mt5.TIMEFRAME_D1
    }
    timeframe = tf_map.get(tf_str, mt5.TIMEFRAME_M15)
    
    rates = mt5.copy_rates_from_pos(mt5_symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
    
    return df[['Open', 'High', 'Low', 'Close']].dropna()

def EMA(series, period):
    return series.ewm(span=period, adjust=False).mean()

def ATR(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def get_trend(df):
    if len(df) < 20: 
        return "NEUTRAL"
    fast = EMA(df["Close"], 10).iloc[-1]
    slow = EMA(df["Close"], 30).iloc[-1]
    return "BULLISH" if fast > slow else "BEARISH"


# =========================================================
# TECHNICAL ANALYSIS
# =========================================================

def analyze_symbol(symbol_name, symbol_config):
    now = datetime.now(timezone.utc)
    memory = load_json_file(MEMORY_FILE, {})
    
    # 🔴 1. COOLDOWN CHECK
    if symbol_name in memory:
        try:
            last_sent_time = datetime.fromisoformat(memory[symbol_name])
            if now - last_sent_time < timedelta(hours=COOLDOWN_HOURS):
                print(f"⏳ {symbol_name} is in cooldown lock.")
                return None
        except Exception:
            pass

    mt5_symbol = symbol_config["mt5_symbol"]
    live_df = get_mt5_data(mt5_symbol, "1m", 10)
    m15 = get_mt5_data(mt5_symbol, "15m", 100)
    h1 = get_mt5_data(mt5_symbol, "1h", 100)

    if live_df.empty or m15.empty or len(m15) < 20:
        return None

    current_price = float(live_df["Close"].iloc[-1])

    # 🔴 2. PRICE SAFETY FILTER
    if symbol_name == "XAU/USD" and (current_price > 4000 or current_price < 1500):
        print(f"⚠️ Bad Price Data Skipped for {symbol_name}: {current_price}")
        return None

    atr_series = ATR(m15)
    if atr_series.empty or pd.isna(atr_series.iloc[-1]):
        return None
    atr_val = float(atr_series.iloc[-1])

    m15_trend = get_trend(m15)
    h1_trend = get_trend(h1) if not h1.empty else "NEUTRAL"

    if m15_trend == "NEUTRAL" or m15_trend != h1_trend:
        return None  

    direction = "BUY" if m15_trend == "BULLISH" else "SELL"
    entry = current_price

    if direction == "BUY":
        sl = entry - (atr_val * 1.5)
        tp1 = entry + (atr_val * 1.5)
        tp2 = entry + (atr_val * 3.0)
    else:
        sl = entry + (atr_val * 1.5)
        tp1 = entry - (atr_val * 1.5)
        tp2 = entry - (atr_val * 3.0)

    return {
        "symbol": symbol_name,
        "mt5_symbol": mt5_symbol,
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "reasons": [
            f"✅ Real-Time MT5 Price: {entry:.4f}",
            f"✅ Trend Alignment ({direction})"
        ],
        "time": now.isoformat()
    }


# =========================================================
# MONITORING & COOLDOWN LOCK
# =========================================================

def check_open_trades():
    journal = load_json_file(JOURNAL_FILE, {"open": [], "closed": []})
    memory = load_json_file(MEMORY_FILE, {})
    still_open = []
    
    for trade in journal.get("open", []):
        symbol_name = trade["symbol"]
        df = get_mt5_data(trade["mt5_symbol"], "1m", 5)
        
        if df.empty:
            still_open.append(trade)
            continue

        last_close = float(df["Close"].iloc[-1])
        last_low = float(df["Low"].iloc[-1])
        last_high = float(df["High"].iloc[-1])
        
        direction = trade["direction"]
        result = None

        if direction == "BUY":
            if last_high >= trade["tp2"]:
                result = "WIN (TP2)"
            elif last_low <= trade["sl"]:
                result = "LOSS (SL)"
        else:
            if last_low <= trade["tp2"]:
                result = "WIN (TP2)"
            elif last_high >= trade["sl"]:
                result = "LOSS (SL)"

        if result:
            trade["exit"] = last_close
            trade["result"] = result
            journal["closed"].append(trade)
            
            # 🔴 ትሬዱ ሲዘጋ ድጋሚ እንዳይልከው COOLDOWN መቆለፍ
            memory[symbol_name] = datetime.now(timezone.utc).isoformat()
            save_json_file(MEMORY_FILE, memory)

            send_telegram(f"📕 CLOSED: {trade['symbol']} {direction} = {result} at {last_close:.4f}")
            print(f"📕 Closed Trade for {symbol_name}: {result}")
        else:
            still_open.append(trade)

    journal["open"] = still_open
    save_json_file(JOURNAL_FILE, journal)

def make_message(signal):
    reasons_str = "\n".join(signal["reasons"])
    return f"""
🔊 NEW TRADE SIGNAL 🔊

Symbol: {signal['symbol']}
Direction: {signal['direction']}

📍 Entry: {signal['entry']:.4f}
🛑 SL: {signal['sl']:.4f}
🎯 TP1: {signal['tp1']:.4f}
🎯 TP2: {signal['tp2']:.4f}

Confirmations:
{reasons_str}
""".strip()


# =========================================================
# MAIN BOT LOOP
# =========================================================

def run_bot():
    print("🤖 Trading Bot Starting with MT5 Real-Time Feed...")
    
    # MT5 ን ማስጀመር
    if not mt5.initialize():
        print("❌ MT5 Initialization Failed! MT5 App ክፍት መሆኑን ያረጋግጡ። Error:", mt5.last_error())
        return
    else:
        print("✅ MT5 Connected Successfully!")

    last_scan_time = 0
    scan_interval = 900  # 15 ደቂቃ (በሰከንድ)

    try:
        while True:
            current_timestamp = time.time()

            # 1. ክፍት ትሬዶች ካሉ በየ 15 ሰከንዱ ፈጣን ፍተሻ ያደርጋል
            check_open_trades()

            # 2. አዳዲስ ሲግናሎችን በየ 15 ደቂቃው ብቻ ይቃኛል
            if current_timestamp - last_scan_time >= scan_interval:
                print(f"\n🔎 Scanning Markets Real-Time at {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC...")
                
                journal = load_json_file(JOURNAL_FILE, {"open": [], "closed": []})
                
                if len(journal.get("open", [])) >= MAX_OPEN_TRADES:
                    print("⏳ Max open trades limit reached.")
                else:
                    for sym_name, sym_config in SYMBOLS.items():
                        open_symbols = [t["symbol"] for t in journal.get("open", [])]
                        if sym_name in open_symbols:
                            continue

                        sig = analyze_symbol(sym_name, sym_config)
                        if sig:
                            msg = make_message(sig)
                            if send_telegram(msg):
                                journal["open"].append(sig)
                                save_json_file(JOURNAL_FILE, journal)
                                
                                memory = load_json_file(MEMORY_FILE, {})
                                memory[sym_name] = datetime.now(timezone.utc).isoformat()
                                save_json_file(MEMORY_FILE, memory)
                                
                                print(f"✅ Signal SENT and LOCKED for {sym_name}")

                last_scan_time = current_timestamp

            # በየ 15 ሰከንዱ የ SL/TP ሁኔታዎችን ፈጥኖ እንዲፈትሽ አጭር ዕረፍት ያደርጋል
            time.sleep(15)

    except KeyboardInterrupt:
        print("\n🛑 Bot Stopped Manually.")
    except Exception as e:
        print(f"❌ Main Loop Error: {e}")
        log_error(str(e))
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    run_bot()
