import os
import json
import time
from datetime import datetime, timezone, timedelta
import traceback

import requests
import yfinance as yf
import pandas as pd
import numpy as np


# =========================================================
# SETTINGS & CONFIGURATIONS
# =========================================================

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

JOURNAL_FILE = "trade_journal.json"
ERROR_LOG_FILE = "error_log.txt"

MAX_OPEN_TRADES = 2
RISK_PER_TRADE = 1.0

# 🔴 ሲግናል ቶሎ ቶሎ እንዳይልከ የማገጃ ሰዓት (በሰዓት)
COOLDOWN_HOURS = 4  
SIGNAL_MEMORY = {}

# 🔴 የ XAUUSD Ticker ወደ XAUUSD=X ተቀይሯል (ከ GC=F ይልቅ)
SYMBOLS = {
    "XAU/USD": {"ticker": "XAUUSD=X", "type": "commodity"},
    "BTC/USD": {"ticker": "BTC-USD", "type": "crypto"},
    "GBP/USD": {"ticker": "GBPUSD=X", "type": "forex"},
    "USD/JPY": {"ticker": "USDJPY=X", "type": "forex"}
}


# =========================================================
# TELEGRAM & LOGGING
# =========================================================

def send_telegram(message):
    if not BOT_TOKEN or not CHAT_ID:
        print("❌ Telegram BOT_TOKEN or CHAT_ID missing in Environment Variables")
        return False
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        res = requests.post(url, json={"chat_id": CHAT_ID, "text": message}, timeout=15)
        return res.ok
    except Exception as e:
        print(f"❌ Telegram Error: {e}")
        return False

def log_error(msg):
    try:
        with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat()}] {msg}\n")
    except:
        pass


# =========================================================
# DATA FETCHING (SAFE YFINANCE)
# =========================================================

def get_data(symbol, period, interval):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            df = yf.download(symbol, period=period, interval=interval, progress=False)

        if df.empty:
            return pd.DataFrame()

        # MultiIndex Column ማስተካከያ
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


# =========================================================
# INDICATORS
# =========================================================

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
# TECHNICAL ANALYSIS & PRICE SANITY CHECK
# =========================================================

def analyze_symbol(symbol_name, symbol_config):
    now = datetime.now(timezone.utc)
    
    # 1. Cooldown Check
    if symbol_name in SIGNAL_MEMORY:
        last_sent = SIGNAL_MEMORY[symbol_name]
        if now - last_sent < timedelta(hours=COOLDOWN_HOURS):
            return None

    ticker = symbol_config["ticker"]
    
    # 🔴 አሁን ያለውን የቀጥታ ዋጋ ከ 1m ዳታ መውሰድ
    live_df = get_data(ticker, "1d", "1m")
    m15 = get_data(ticker, "3d", "15m")
    h1 = get_data(ticker, "14d", "1h")

    if live_df.empty or m15.empty or len(m15) < 20:
        return None

    current_price = float(live_df["Close"].iloc[-1]) # ትክክለኛው LIVE PRICE

    # 🔴 Price Safety Filter: የተሳሳተ የወርቅ ዋጋ ከተመለሰ ውድቅ ያደርጋል
    if symbol_name == "XAU/USD" and (current_price > 3500 or current_price < 1500):
        print(f"⚠️ Bad Gold Data Detected: {current_price}. Signal Skipped.")
        return None

    atr_series = ATR(m15)
    if atr_series.empty or pd.isna(atr_series.iloc[-1]):
        return None
    atr_val = float(atr_series.iloc[-1])

    m15_trend = get_trend(m15)
    h1_trend = get_trend(h1) if not h1.empty else "NEUTRAL"

    # ሁለቱም Timeframe ካልተስማሙ አይነግድም
    if m15_trend == "NEUTRAL" or m15_trend != h1_trend:
        return None  

    direction = "BUY" if m15_trend == "BULLISH" else "SELL"
    entry = current_price # Entry ሁልጊዜ አሁን ያለው ገበያ ዋጋ ነው

    if direction == "BUY":
        sl = entry - (atr_val * 1.5)
        tp1 = entry + (atr_val * 1.5)
        tp2 = entry + (atr_val * 3.0)
    else:
        sl = entry + (atr_val * 1.5)
        tp1 = entry - (atr_val * 1.5)
        tp2 = entry - (atr_val * 3.0)

    # ሲግናል መላኩን መዝግቦ መያዝ
    SIGNAL_MEMORY[symbol_name] = now

    return {
        "symbol": symbol_name,
        "ticker": ticker,
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "score": 8,
        "reasons": [
            f"✅ Validated Live Price: {entry:.2f}",
            f"✅ Trend Alignment ({direction})"
        ],
        "time": now.isoformat()
    }


# =========================================================
# JOURNAL & LIVE MONITORING
# =========================================================

def load_journal():
    if not os.path.exists(JOURNAL_FILE):
        return {"open": [], "closed": []}
    try:
        with open(JOURNAL_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {"open": [], "closed": []}

def save_journal(j):
    try:
        with open(JOURNAL_FILE, "w", encoding="utf-8") as f:
            json.dump(j, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"❌ Error Saving Journal: {e}")

def check_open_trades(journal):
    still_open = []
    for trade in journal["open"]:
        df = get_data(trade["ticker"], "1d", "1m")
        if df.empty:
            still_open.append(trade)
            continue

        current_price = float(df["Close"].iloc[-1])
        direction = trade["direction"]
        result = None

        if direction == "BUY":
            if current_price >= trade["tp2"]:
                result = "WIN (TP2)"
            elif current_price <= trade["sl"]:
                result = "LOSS (SL)"
        else:
            if current_price <= trade["tp2"]:
                result = "WIN (TP2)"
            elif current_price >= trade["sl"]:
                result = "LOSS (SL)"

        if result:
            trade["exit"] = current_price
            trade["result"] = result
            journal["closed"].append(trade)
            send_telegram(f"📕 TRADE CLOSED: {trade['symbol']}\nDirection: {direction}\nResult: {result}\nExit Price: {current_price:.4f}")
        else:
            still_open.append(trade)

    journal["open"] = still_open


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
    print("🤖 Trading Bot Started Safely...")
    while True:
        try:
            journal = load_journal()
            
            # 1. ክፍት ትሬዶችን በ Live Price ፈትሽ
            if journal.get("open"):
                check_open_trades(journal)
                save_journal(journal)

            # 2. ከ 2 በላይ ክፍት ትሬድ ካለ አዲስ አትፈልግ
            if len(journal.get("open", [])) >= MAX_OPEN_TRADES:
                time.sleep(60)
                continue

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
                        save_journal(journal)
                        print(f"✅ Real-time Signal Sent for {sym_name}")

            time.sleep(60) # በየ 1 ደቂቃው ገበያውን ይፈትሻል

        except Exception as e:
            print(f"❌ Main Loop Error: {e}")
            log_error(str(e))
            time.sleep(10)

if __name__ == "__main__":
    run_bot()
