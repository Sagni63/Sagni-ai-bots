import os
import json
import time
import shutil
from datetime import datetime, timezone
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

MIN_SCORE = 8

SYMBOLS = {
    "XAU/USD": {"ticker": "XAUUSD=X", "type": "commodity", "leverage": 20},
    "BTC/USD": {"ticker": "BTC-USD", "type": "crypto", "leverage": 10},
    "GBP/USD": {"ticker": "GBPUSD=X", "type": "forex", "leverage": 30},
    "USD/JPY": {"ticker": "USDJPY=X", "type": "forex", "leverage": 30}
}

TIMEFRAME_CONFIG = {
    "crypto": {
        "W1": ("6mo", "1wk"),
        "D1": ("3mo", "1d"),
        "H4": ("30d", "4h"),
        "H1": ("14d", "1h"),
        "M15": ("3d", "15m")
    },
    "forex": {
        "W1": ("2y", "1wk"),
        "D1": ("1y", "1d"),
        "H4": ("60d", "4h"),
        "H1": ("30d", "1h"),
        "M15": ("7d", "15m")
    },
    "commodity": {
        "W1": ("2y", "1wk"),
        "D1": ("1y", "1d"),
        "H4": ("60d", "4h"),
        "H1": ("30d", "1h"),
        "M15": ("7d", "15m")
    }
}

SIGNAL_MEMORY = {}
COOLDOWN_HOURS = 6
LAST_CLOSED_ENTRY = {}
LAST_SIGNAL_PRICE = {}
MIN_PRICE_CHANGE_FOR_SIGNAL = 0.5
MAX_PRICE_CHANGE_PERCENT = 1.0
RISK_PER_TRADE = 1.0
MAX_OPEN_TRADES = 2

BTC_COOLDOWN_HOURS = 4
BTC_MIN_SCORE = 6
BTC_MAX_PRICE_CHANGE = 2.0

XAU_COOLDOWN_HOURS = 6
XAU_MIN_SCORE = 7
XAU_MAX_PRICE_CHANGE = 1.5

TP_PADDING = 0.001
SL_PADDING = 0.002


# =========================================================
# TELEGRAM & LOGGING
# =========================================================

def send_telegram(message):
    if not BOT_TOKEN or not CHAT_ID:
        print("❌ BOT_TOKEN or CHAT_ID is missing")
        return False

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    try:
        response = requests.post(
            url,
            json={"chat_id": CHAT_ID, "text": message},
            timeout=30
        )
        response.raise_for_status()
        print("✅ Telegram message sent")
        return True
    except Exception as e:
        print(f"❌ Telegram error: {repr(e)}")
        return False


def log_error(error_message):
    try:
        with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.now().isoformat()}] {error_message}\n")
            f.write(traceback.format_exc())
            f.write("\n" + "="*50 + "\n")
    except:
        pass


# =========================================================
# MARKET DATA (FIXED YFINANCE DATA STRUCTURE)
# =========================================================

def get_data(symbol, period, interval):
    """
    yfinance የውሂብ መዋቅር (MultiIndex Columns) ችግር እንዳይፈጥር
    በጥንቃቄ የተቀረጸ የዳታ መሳቢያ ፋንክሽን።
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty and (symbol == "GC=F" or symbol == "XAUUSD=X"):
            alt_symbol = "XAUUSD=X" if symbol == "GC=F" else "GC=F"
            ticker = yf.Ticker(alt_symbol)
            df = ticker.history(period=period, interval=interval)

        if df.empty:
            df = yf.download(symbol, period=period, interval=interval, progress=False)

        if df.empty:
            print(f"❌ No data found for {symbol} ({interval})")
            return pd.DataFrame()

        # MultiIndex Column ማስተካከያ (ዋናው ማስተካከያ)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Columns Standardize ማድረግ (የመጀመሪያ አቢይ ፊደል)
        df.columns = [str(c).capitalize() for c in df.columns]

        # የሚፈለጉት ዋና የካንድል ኮልሞች መኖራቸውን ማረጋገጥ
        required = ["Open", "High", "Low", "Close"]
        missing = [col for col in required if col not in df.columns]

        if missing:
            print(f"❌ Missing required columns {missing} for {symbol}")
            return pd.DataFrame()

        # የጎደሉ (NaN) መረጃዎችን ማጽዳት
        df = df[required].dropna()
        return df

    except Exception as e:
        print(f"❌ Data fetch error ({symbol}): {repr(e)}")
        log_error(f"get_data error for {symbol}: {repr(e)}")
        return pd.DataFrame()


# =========================================================
# INDICATORS & ANALYSIS
# =========================================================

def EMA(series, period):
    return series.ewm(span=period, adjust=False).mean()


def ATR(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    return tr.rolling(period).mean()


def find_pivots(df, lookback=5):
    highs = df["High"]
    lows = df["Low"]
    pivot_highs, pivot_lows = [], []

    for i in range(lookback, len(df) - lookback):
        if highs.iloc[i] == highs.iloc[i-lookback:i+lookback+1].max():
            pivot_highs.append((df.index[i], highs.iloc[i]))
        if lows.iloc[i] == lows.iloc[i-lookback:i+lookback+1].min():
            pivot_lows.append((df.index[i], lows.iloc[i]))

    return pivot_highs, pivot_lows


def find_trendlines(df, lookback=30):
    if len(df) < lookback:
        return [], []
    pivot_highs, pivot_lows = find_pivots(df, lookback=3)
    upper_trendline = [pivot_highs[-2], pivot_highs[-1]] if len(pivot_highs) >= 2 else []
    lower_trendline = [pivot_lows[-2], pivot_lows[-1]] if len(pivot_lows) >= 2 else []
    return upper_trendline, lower_trendline


def check_trendline_break(df, trendline, direction="UP"):
    if not trendline or len(trendline) < 2:
        return False, None

    p1_time, p1_price = trendline[0]
    p2_time, p2_price = trendline[1]

    if isinstance(p1_time, pd.Timestamp) and isinstance(p2_time, pd.Timestamp):
        time_diff = (p2_time - p1_time).total_seconds()
        if time_diff == 0:
            return False, None
        slope = (p2_price - p1_price) / time_diff
    else:
        return False, None

    last_price = float(df["Close"].iloc[-1])
    last_time = df.index[-1]
    time_diff_last = (last_time - p1_time).total_seconds()
    expected_price = p1_price + slope * time_diff_last

    return (last_price > expected_price, expected_price) if direction == "UP" else (last_price < expected_price, expected_price)


def find_supply_demand_improved(df, lookback=60):
    if len(df) < lookback:
        return [], []

    zones = []
    recent = df.tail(lookback)
    ranges = recent["High"] - recent["Low"]
    average_range = ranges.median()

    if not np.isfinite(average_range) or average_range <= 0:
        return [], []

    for i in range(3, len(recent) - 3):
        base = recent.iloc[i-1:i+2]
        base_range = (base["High"] - base["Low"]).mean()
        previous = recent.iloc[i-2]
        future = recent.iloc[i+2]
        movement = float(future["Close"] - previous["Close"])

        if base_range < average_range * 1.3 and movement < -average_range * 1.2:
            zones.append({
                "type": "SUPPLY",
                "low": float(base["Low"].min()),
                "high": float(base["High"].max()),
                "strength": min(10, int(abs(movement) / average_range * 1.5))
            })
        elif base_range < average_range * 1.3 and movement > average_range * 1.2:
            zones.append({
                "type": "DEMAND",
                "low": float(base["Low"].min()),
                "high": float(base["High"].max()),
                "strength": min(10, int(abs(movement) / average_range * 1.5))
            })

    return zones[-10:], zones[-5:] if len(zones) > 5 else []


def is_near_zone(price, zones, atr_value, direction="BUY"):
    if not zones:
        return None, None

    for zone in reversed(zones):
        low, high = zone["low"], zone["high"]
        zone_type = zone["type"]
        strength = zone.get("strength", 3)

        if low <= price <= high:
            return zone_type, strength

        distance = min(abs(price - low), abs(price - high))
        if distance <= atr_value * 0.5:
            return zone_type, strength

    return None, None


def detect_candlestick_patterns(df, direction="BUY"):
    if len(df) < 2:
        return 0, []

    candle, previous = df.iloc[-1], df.iloc[-2]
    c_close, c_open = float(candle["Close"]), float(candle["Open"])
    c_high, c_low = float(candle["High"]), float(candle["Low"])
    p_close, p_open = float(previous["Close"]), float(previous["Open"])

    body = abs(c_close - c_open)
    body_test = max(body, 0.000001)
    upper_wick = c_high - max(c_open, c_close)
    lower_wick = min(c_open, c_close) - c_low

    reasons, score = [], 0

    if direction == "BUY":
        if c_close > c_open and c_high > max(c_open, c_close):
            score += 1
            reasons.append("📈 Bullish Candle")
        if c_close > c_open and c_open <= p_close and c_close >= p_open:
            score += 2
            reasons.append("✅ Bullish Engulfing")
        if lower_wick >= body_test * 1.2 and upper_wick <= body_test * 0.8:
            score += 1
            reasons.append("✅ Hammer-like")
    else:
        if c_close < c_open and c_low < min(c_open, c_close):
            score += 1
            reasons.append("📉 Bearish Candle")
        if c_close < c_open and c_open >= p_close and c_close <= p_open:
            score += 2
            reasons.append("✅ Bearish Engulfing")
        if upper_wick >= body_test * 1.2 and lower_wick <= body_test * 0.8:
            score += 1
            reasons.append("✅ Shooting Star-like")

    return min(score, 4), reasons


def get_trend_improved(df, min_candles=15):
    if len(df) < min_candles:
        return "NEUTRAL", 0

    p1 = 10 if len(df) >= 30 else 5
    p2 = 30 if len(df) >= 30 else 10

    ema_fast = EMA(df["Close"], p1)
    ema_slow = EMA(df["Close"], p2)

    current_fast = float(ema_fast.iloc[-1])
    current_slow = float(ema_slow.iloc[-1])

    last_3 = df["Close"].tail(3)
    trend_strength = 0

    if len(last_3) >= 3:
        if all(last_3.iloc[i] > last_3.iloc[i-1] for i in range(1, len(last_3))):
            trend_strength += 1
        elif all(last_3.iloc[i] < last_3.iloc[i-1] for i in range(1, len(last_3))):
            trend_strength += 1

    if current_fast > current_slow:
        trend_strength += 1
        return "BULLISH", min(trend_strength, 5)
    elif current_fast < current_slow:
        trend_strength += 1
        return "BEARISH", min(trend_strength, 5)

    return "NEUTRAL", trend_strength


def multi_timeframe_analysis(symbol_name, ticker, market_type="forex"):
    config = TIMEFRAME_CONFIG.get(market_type, TIMEFRAME_CONFIG["forex"])
    data, trends, strengths = {}, {}, {}

    for name, (period, interval) in config.items():
        try:
            df = get_data(ticker, period, interval)
            if df.empty:
                continue
            data[name] = df
            trend, strength = get_trend_improved(df)
            trends[name] = trend
            strengths[name] = strength
        except Exception:
            continue

    if not data:
        return None

    higher = [t for t in ["W1", "D1", "H4"] if t in trends]
    bullish_count = sum(1 for t in higher if trends.get(t) == "BULLISH")
    bearish_count = sum(1 for t in higher if trends.get(t) == "BEARISH")
    entry_trend = trends.get("M15", "NEUTRAL")

    return {
        "data": data,
        "trends": trends,
        "strengths": strengths,
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "entry_trend": entry_trend
    }


def fibonacci_levels(df, lookback=40):
    if len(df) < lookback:
        return 0, None

    recent = df.tail(lookback)
    low, high = float(recent["Low"].min()), float(recent["High"].max())
    price = float(df["Close"].iloc[-1])
    distance = high - low

    if distance <= 0:
        return 0, None

    levels = {
        "23.6%": low + distance * 0.236,
        "38.2%": low + distance * 0.382,
        "50.0%": low + distance * 0.500,
        "61.8%": low + distance * 0.618
    }

    for name, level in levels.items():
        if abs(price - level) <= distance * 0.03:
            return 1, f"Near Fibonacci {name}"

    return 0, None


# =========================================================
# MAIN SIGNAL ANALYSIS
# =========================================================

def analyze_improved(symbol_name, symbol_config):
    try:
        ticker = symbol_config["ticker"]
        market_type = symbol_config["type"]

        print(f"\n🔎 Analyzing {symbol_name} ({market_type})...")

        is_xau = (symbol_name == "XAU/USD")
        is_btc = (symbol_name == "BTC/USD")
        min_score = XAU_MIN_SCORE if is_xau else (BTC_MIN_SCORE if is_btc else MIN_SCORE)

        mta = multi_timeframe_analysis(symbol_name, ticker, market_type)
        if not mta:
            print("❌ Multi-timeframe analysis failed")
            return None

        data = mta["data"]
        trends = mta["trends"]
        bullish_count = mta["bullish_count"]
        bearish_count = mta["bearish_count"]
        entry_trend = mta["entry_trend"]

        if "M15" not in data:
            return None

        m15 = data["M15"]
        h1 = data.get("H1", m15)

        if len(m15) < 15:
            return None

        score = 0
        reasons = []

        if entry_trend == "BULLISH":
            direction = "BUY"
        elif entry_trend == "BEARISH":
            direction = "SELL"
        else:
            last_close = float(m15["Close"].iloc[-1])
            prev_close = float(m15["Close"].iloc[-2])
            direction = "BUY" if last_close > prev_close else "SELL"

        # 1. HTF Trend
        if direction == "BUY" and bullish_count >= 1:
            score += 2
            reasons.append(f"✅ HTF Bullish ({bullish_count}/3)")
        elif direction == "SELL" and bearish_count >= 1:
            score += 2
            reasons.append(f"✅ HTF Bearish ({bearish_count}/3)")

        # 2. H1 Trend
        h1_trend = trends.get("H1", "NEUTRAL")
        if direction == "BUY" and h1_trend == "BULLISH":
            score += 1
            reasons.append("✅ H1 Bullish")
        elif direction == "SELL" and h1_trend == "BEARISH":
            score += 1
            reasons.append("✅ H1 Bearish")

        # 3. S&D Zones
        supply_zones, demand_zones = find_supply_demand_improved(h1, lookback=60)
        price = float(m15["Close"].iloc[-1])
        atr_series = ATR(m15, period=10)
        atr_value = float(atr_series.iloc[-1]) if not atr_series.empty else 1.0

        if direction == "BUY" and demand_zones:
            z_type, z_str = is_near_zone(price, demand_zones, atr_value, "BUY")
            if z_type == "DEMAND":
                score += min(z_str, 3)
                reasons.append(f"✅ Near Demand (Strength: {z_str}/10)")
        elif direction == "SELL" and supply_zones:
            z_type, z_str = is_near_zone(price, supply_zones, atr_value, "SELL")
            if z_type == "SUPPLY":
                score += min(z_str, 3)
                reasons.append(f"✅ Near Supply (Strength: {z_str}/10)")

        # 4. Trendline
        upper_tl, lower_tl = find_trendlines(m15, lookback=15)
        if direction == "BUY" and lower_tl:
            is_broken, _ = check_trendline_break(m15, lower_tl, "UP")
            if is_broken:
                score += 1
                reasons.append("✅ Trendline Breakout UP")
        elif direction == "SELL" and upper_tl:
            is_broken, _ = check_trendline_break(m15, upper_tl, "DOWN")
            if is_broken:
                score += 1
                reasons.append("✅ Trendline Breakout DOWN")

        # 5. Candlestick & Fibonacci
        c_score, c_reasons = detect_candlestick_patterns(m15, direction)
        score += c_score
        reasons.extend(c_reasons)

        f_score, f_reason = fibonacci_levels(h1)
        score += f_score
        if f_reason:
            reasons.append(f_reason)

        if score < min_score:
            print(f"❌ NO TRADE | Score {score} < {min_score}")
            return None

        # Entry, SL, TP Calculation
        if direction == "BUY":
            entry = demand_zones[-1]["high"] * 1.001 if demand_zones else price
            sl = entry - (atr_value * 1.2)
            risk = max(entry - sl, atr_value * 0.5)
            tp1 = entry + (risk * 1.5)
            tp2 = entry + (risk * 2.5)
        else:
            entry = supply_zones[-1]["low"] * 0.999 if supply_zones else price
            sl = entry + (atr_value * 1.2)
            risk = max(sl - entry, atr_value * 0.5)
            tp1 = entry - (risk * 1.5)
            tp2 = entry - (risk * 2.5)

        return {
            "symbol": symbol_name,
            "ticker": ticker,
            "direction": direction,
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "score": min(score, 14),
            "reasons": reasons,
            "timeframe": "M15",
            "time": datetime.now(timezone.utc).isoformat(),
            "atr": atr_value,
            "trends": trends
        }

    except Exception as e:
        print(f"❌ Error in analyze_improved: {repr(e)}")
        log_error(f"analyze_improved: {repr(e)}")
        return None


# =========================================================
# JOURNAL MANAGEMENT & TRADE MONITORING
# =========================================================

def load_journal():
    if not os.path.exists(JOURNAL_FILE):
        return {"open": [], "closed": []}
    try:
        with open(JOURNAL_FILE, "r", encoding="utf-8") as file:
            journal = json.load(file)
            journal.setdefault("open", [])
            journal.setdefault("closed", [])
            return journal
    except Exception:
        return {"open": [], "closed": []}


def save_journal(journal):
    try:
        with open(JOURNAL_FILE, "w", encoding="utf-8") as file:
            json.dump(journal, file, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"❌ Error saving journal: {repr(e)}")


def check_open_trades(journal):
    still_open = []

    for trade in journal["open"]:
        try:
            df = get_data(trade["ticker"], "1d", "1m")
            if df.empty:
                df = get_data(trade["ticker"], "1d", "15m")

            if df.empty:
                still_open.append(trade)
                continue

            try:
                entry_time = pd.to_datetime(trade["time"], utc=True)
            except Exception:
                still_open.append(trade)
                continue

            result = None
            exit_price = None
            hit_tp1 = trade.get("hit_tp1", False)
            hit_tp2 = False

            for timestamp, candle in df.iterrows():
                try:
                    candle_time = pd.to_datetime(timestamp, utc=True)
                    if candle_time <= entry_time:
                        continue
                except Exception:
                    continue

                c_high = float(candle["High"])
                c_low = float(candle["Low"])

                if trade["direction"] == "BUY":
                    if c_high >= trade["tp2"]:
                        result = "WIN"
                        exit_price = trade["tp2"]
                        hit_tp2 = True
                        break
                    elif c_high >= trade["tp1"] and not hit_tp1:
                        hit_tp1 = True
                        trade["hit_tp1"] = True
                        send_telegram(f"🎯 TP1 HIT for {trade['symbol']} BUY at {trade['tp1']:.4f}!")
                    elif c_low <= trade["sl"]:
                        result = "LOSS" if not hit_tp1 else "PARTIAL WIN"
                        exit_price = trade["sl"]
                        break

                else:  # SELL
                    if c_low <= trade["tp2"]:
                        result = "WIN"
                        exit_price = trade["tp2"]
                        hit_tp2 = True
                        break
                    elif c_low <= trade["tp1"] and not hit_tp1:
                        hit_tp1 = True
                        trade["hit_tp1"] = True
                        send_telegram(f"🎯 TP1 HIT for {trade['symbol']} SELL at {trade['tp1']:.4f}!")
                    elif c_high >= trade["sl"]:
                        result = "LOSS" if not hit_tp1 else "PARTIAL WIN"
                        exit_price = trade["sl"]
                        break

            if result:
                trade["result"] = result
                trade["exit"] = exit_price
                trade["closed_at"] = datetime.now(timezone.utc).isoformat()
                trade["hit_tp1"] = hit_tp1
                trade["hit_tp2"] = hit_tp2
                journal["closed"].append(trade)

                send_telegram(f"📕 CLOSED: {trade['symbol']} {trade['direction']} = {result} at {exit_price:.4f}")
            else:
                still_open.append(trade)

        except Exception as e:
            print(f"❌ Error checking open trade: {repr(e)}")
            still_open.append(trade)

    journal["open"] = still_open


def make_message(signal):
    confirmation = "\n".join(f"• {reason}" for reason in signal["reasons"])
    return f"""
🔊🔊🔊 {signal['symbol']} SIGNAL 🔊🔊🔊

📌 Direction: {signal["direction"]}

📍 Entry: {signal["entry"]:.4f}
🛑 SL: {signal["sl"]:.4f}
🎯 TP1: {signal["tp1"]:.4f}
🎯 TP2: {signal["tp2"]:.4f}

📊 Score: {signal["score"]}/14

✅ Confirmation:
{confirmation}

⚠️ Risk: {RISK_PER_TRADE}% per trade
""".strip()


# =========================================================
# MAIN LOOP
# =========================================================

def run_continuous_bot():
    print("==========================================")
    print("🔊 TRADING BOT IS RUNNING SUCCESSFULLY...")
    print("==========================================")

    while True:
        try:
            journal = load_journal()

            if journal.get("open"):
                check_open_trades(journal)
                save_journal(journal)

            open_count = len(journal.get("open", []))
            if open_count >= MAX_OPEN_TRADES:
                print(f"⏳ Max open trades reached ({open_count}/{MAX_OPEN_TRADES})")
                time.sleep(60)
                continue

            for symbol_name, symbol_config in SYMBOLS.items():
                already_open = {trade["symbol"] for trade in journal.get("open", [])}
                if symbol_name in already_open:
                    continue

                signal = analyze_improved(symbol_name, symbol_config)
                if signal:
                    msg = make_message(signal)
                    if send_telegram(msg):
                        journal["open"].append(signal)
                        save_journal(journal)
                        print(f"✅ Signal executed for {symbol_name}")

            time.sleep(60)

        except Exception as e:
            print(f"❌ Main Loop Error: {repr(e)}")
            log_error(f"Main Loop: {repr(e)}")
            time.sleep(10)


if __name__ == "__main__":
    run_continuous_bot()
