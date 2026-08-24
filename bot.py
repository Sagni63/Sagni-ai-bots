import os
import json
import time
from datetime import datetime, timezone, timedelta
import requests
import yfinance as yf
import pandas as pd
import numpy as np


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

SYMBOLS = {
    "XAU/USD": {"ticker": "XAUUSD=X", "type": "commodity"},
    "BTC/USD": {"ticker": "BTC-USD", "type": "crypto"},
    "GBP/USD": {"ticker": "GBPUSD=X", "type": "forex"},
    "USD/JPY": {"ticker": "USDJPY=X", "type": "forex"}
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
# MARKET DATA & INDICATORS
# =========================================================

def get_data(symbol, period, interval):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            df = yf.download(symbol, period=period, interval=interval, progress=False)

        if df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [str(c).capitalize() for c in df.columns]
        required = ["Open", "High", "Low", "Close"]
        
        if not all(col in df.columns for col in required):
            return pd.DataFrame()

        return df[required].dropna()
    except Exception as e:
        print(f"❌ Fetch Error ({symbol}): {e}")
        return pd.DataFrame()

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

    ticker = symbol_config["ticker"]
    live_df = get_data(ticker, "1d", "1m")
    m15 = get_data(ticker, "3d", "15m")
    h1 = get_data(ticker, "14d", "1h")

    if live_df.empty or m15.empty or len(m15) < 20:
        return None

    current_price = float(live_df["Close"].iloc[-1])

    # 🔴 2. PRICE SAFETY FILTER
    if symbol_name == "XAU/USD" and (current_price > 3500 or current_price < 1500):
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
        "ticker": ticker,
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "reasons": [
            f"✅ Validated Live Price: {entry:.4f}",
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
        df = get_data(trade["ticker"], "1d", "1m")
        
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
    print("🤖 Trading Bot Started Safely (Strict 15-Minute Loop)...")
    
    while True:
        try:
            print(f"\n🔎 Scanning Markets at {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC...")

            # 1. ክፍት ትሬዶች ካሉ ሁኔታቸውን ፈትሽ
            check_open_trades()

            journal = load_json_file(JOURNAL_FILE, {"open": [], "closed": []})
            
            # 2. ክፍት ትሬድ ከሞላ አትፈልግ
            if len(journal.get("open", [])) >= MAX_OPEN_TRADES:
                print("⏳ Max open trades limit reached.")
            else:
                # 3. አዲስ ሲግናል ፈልግ
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

            # 🔴 15 ደቂቃ ይተኛል
            print("💤 Sleeping for 15 minutes...")
            time.sleep(900)

        except Exception as e:
            print(f"❌ Main Loop Error: {e}")
            log_error(str(e))
            time.sleep(60)

if __name__ == "__main__":
    run_bot()
