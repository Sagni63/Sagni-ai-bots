import os
import json
import time
import shutil
from datetime import datetime, timezone

import requests
import yfinance as yf
import pandas as pd
import numpy as np


# =========================================================
# SETTINGS - ለሲግናል ፈጣን መላክ የተበጀ
# =========================================================

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

JOURNAL_FILE = "trade_journal.json"

# =========================================================
# 🟢 ዝቅተኛ ውጤት ቀንሷል - ብዙ ሲግናሎች እንዲላኩ
# =========================================================
MIN_SCORE = 3  # ከ6 ወደ 3 ቀንሷል

# =========================================================
# የገበያ ጥንዶች - ትክክለኛ ቲከሮች
# =========================================================
SYMBOLS = {
    "XAU/USD": {"ticker": "GC=F", "type": "commodity", "leverage": 20},
    "BTC/USD": {"ticker": "BTC-USD", "type": "crypto", "leverage": 10},
    "GBP/USD": {"ticker": "GBPUSD=X", "type": "forex", "leverage": 30},
    "USD/JPY": {"ticker": "USDJPY=X", "type": "forex", "leverage": 30}
}

# =========================================================
# ለእያንዳንዱ ጥንድ የተለያዩ የጊዜ ማዕቀፎች
# =========================================================
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

# =========================================================
# ሌሎች ቅንጅቶች
# =========================================================
SIGNAL_MEMORY = {}
COOLDOWN_HOURS = 2  # ከ4 ወደ 2 ሰዓት ቀንሷል
LAST_CLOSED_ENTRY = {}
MAX_PRICE_CHANGE_PERCENT = 1.0  # ከ0.5% ወደ 1% ጨምሯል
RISK_PER_TRADE = 1.0
MAX_OPEN_TRADES = 4

# ለBTC ልዩ ቅንጅቶች
BTC_COOLDOWN_HOURS = 1
BTC_MIN_SCORE = 2
BTC_MAX_PRICE_CHANGE = 2.0

TP_PADDING = 0.001
SL_PADDING = 0.002


# =========================================================
# TELEGRAM
# =========================================================

def send_telegram(message):
    if not BOT_TOKEN or not CHAT_ID:
        print("❌ BOT_TOKEN or CHAT_ID is missing")
        return False

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    try:
        response = requests.post(
            url,
            json={
                "chat_id": CHAT_ID,
                "text": message
            },
            timeout=30
        )
        response.raise_for_status()
        print("✅ Telegram message sent")
        return True
    except Exception as e:
        print(f"❌ Telegram error: {repr(e)}")
        return False


# =========================================================
# MARKET DATA - ካሽ አጽድቶ አዲስ ውሂብ ያመጣል
# =========================================================

def get_data(symbol, period, interval):
    """ካሽ አጽድቶ አዲስ ውሂብ ያመጣል"""
    try:
        # 🆕 ካሽን አጽዳ (እያንዳንዱን ጊዜ አዲስ ውሂብ ለማግኘት)
        try:
            yf.clear_cache()
        except:
            pass
        
        # ካሽ አቃፊን ሰርዝ
        cache_dir = os.path.expanduser("~/.cache/py-yfinance")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if df.empty:
            print(f"❌ No data: {symbol} {interval}")
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [str(c).title() for c in df.columns]

        required = ["Open", "High", "Low", "Close"]
        for col in required:
            if col not in df.columns:
                print(f"❌ Missing column: {col}")
                return pd.DataFrame()

        return df.dropna(subset=required)

    except Exception as e:
        print(f"❌ Data fetch error ({symbol}): {repr(e)}")
        return pd.DataFrame()


def make_45m(df):
    if df.empty:
        return df
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df.resample("45min").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last"
    }).dropna()


def make_4h(df):
    if df.empty:
        return df
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df.resample("4h").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last"
    }).dropna()


# =========================================================
# INDICATORS
# =========================================================

def EMA(series, period):
    return series.ewm(span=period, adjust=False).mean()


def SMA(series, period):
    return series.rolling(window=period).mean()


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
    
    pivot_highs = []
    pivot_lows = []
    
    for i in range(lookback, len(df) - lookback):
        if highs.iloc[i] == highs.iloc[i-lookback:i+lookback+1].max():
            pivot_highs.append((df.index[i], highs.iloc[i]))
        if lows.iloc[i] == lows.iloc[i-lookback:i+lookback+1].min():
            pivot_lows.append((df.index[i], lows.iloc[i]))
    
    return pivot_highs, pivot_lows


# =========================================================
# TRENDLINE DETECTION
# =========================================================

def find_trendlines(df, lookback=30):
    if len(df) < lookback:
        return [], []
    
    pivot_highs, pivot_lows = find_pivots(df, lookback=3)
    
    upper_trendline = []
    lower_trendline = []
    
    if len(pivot_highs) >= 2:
        p1 = pivot_highs[-2]
        p2 = pivot_highs[-1]
        upper_trendline = [p1, p2]
    
    if len(pivot_lows) >= 2:
        p1 = pivot_lows[-2]
        p2 = pivot_lows[-1]
        lower_trendline = [p1, p2]
    
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
    
    if direction == "UP":
        return last_price > expected_price, expected_price
    else:
        return last_price < expected_price, expected_price


# =========================================================
# SUPPLY & DEMAND ZONES - የተሻሻለ
# =========================================================

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
        low = zone["low"]
        high = zone["high"]
        zone_type = zone["type"]
        strength = zone.get("strength", 3)
        
        if low <= price <= high:
            return zone_type, strength
        
        distance = min(abs(price - low), abs(price - high))
        if distance <= atr_value * 0.5:
            return zone_type, strength
    
    return None, None


# =========================================================
# CANDLESTICK PATTERNS - የተሻሻለ
# =========================================================

def detect_candlestick_patterns(df, direction="BUY"):
    if len(df) < 2:
        return 0, []
    
    candle = df.iloc[-1]
    previous = df.iloc[-2]
    
    c_close = float(candle["Close"])
    c_open = float(candle["Open"])
    c_high = float(candle["High"])
    c_low = float(candle["Low"])
    
    p_close = float(previous["Close"])
    p_open = float(previous["Open"])
    
    body = abs(c_close - c_open)
    body_test = max(body, 0.000001)
    
    upper_wick = c_high - max(c_open, c_close)
    lower_wick = min(c_open, c_close) - c_low
    
    reasons = []
    score = 0
    
    if direction == "BUY":
        if c_close > c_open and c_high > max(c_open, c_close):
            score += 1
            reasons.append("📈 Bullish Candle")
        if (c_close > c_open and c_open <= p_close and c_close >= p_open):
            score += 2
            reasons.append("✅ Bullish Engulfing")
        if (lower_wick >= body_test * 1.2 and upper_wick <= body_test * 0.8):
            score += 1
            reasons.append("✅ Hammer-like")
    else:
        if c_close < c_open and c_low < min(c_open, c_close):
            score += 1
            reasons.append("📉 Bearish Candle")
        if (c_close < c_open and c_open >= p_close and c_close <= p_open):
            score += 2
            reasons.append("✅ Bearish Engulfing")
        if (upper_wick >= body_test * 1.2 and lower_wick <= body_test * 0.8):
            score += 1
            reasons.append("✅ Shooting Star-like")
    
    return min(score, 4), reasons


# =========================================================
# TREND ANALYSIS - የተሻሻለ
# =========================================================

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


# =========================================================
# MULTI-TIMEFRAME ANALYSIS - የተሻሻለ
# =========================================================

def multi_timeframe_analysis(symbol_name, ticker, market_type="forex"):
    config = TIMEFRAME_CONFIG.get(market_type, TIMEFRAME_CONFIG["forex"])
    timeframes = config
    
    data = {}
    trends = {}
    strengths = {}
    
    for name, (period, interval) in timeframes.items():
        df = get_data(ticker, period, interval)
        if df.empty:
            print(f"⚠️ Empty data for {name} ({ticker}) - skipping")
            continue
        data[name] = df
        trend, strength = get_trend_improved(df)
        trends[name] = trend
        strengths[name] = strength
    
    if not data:
        return None
    
    available = list(trends.keys())
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


# =========================================================
# FIBONACCI
# =========================================================

def fibonacci_levels(df, lookback=40):
    if len(df) < lookback:
        return None
    
    recent = df.tail(lookback)
    low = float(recent["Low"].min())
    high = float(recent["High"].max())
    price = float(df["Close"].iloc[-1])
    
    distance = high - low
    if distance <= 0:
        return None
    
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
# MAIN ANALYSIS - ለሲግናል ፈጣን መላክ የተበጀ
# =========================================================

def analyze_improved(symbol_name, symbol_config):
    ticker = symbol_config["ticker"]
    market_type = symbol_config["type"]
    
    print(f"\n🔎 Analyzing {symbol_name} ({market_type})...")
    
    is_btc = (symbol_name == "BTC/USD")
    min_score = BTC_MIN_SCORE if is_btc else MIN_SCORE
    
    mta = multi_timeframe_analysis(symbol_name, ticker, market_type)
    if not mta:
        print("❌ Multi-timeframe analysis failed - no data")
        return None
    
    data = mta["data"]
    trends = mta["trends"]
    bullish_count = mta["bullish_count"]
    bearish_count = mta["bearish_count"]
    entry_trend = mta["entry_trend"]
    
    if "M15" not in data:
        print("❌ No M15 data available")
        return None
    
    m15 = data["M15"]
    h1 = data.get("H1", m15)
    d1 = data.get("D1", h1)
    w1 = data.get("W1", d1)
    
    if len(m15) < 15:
        print("❌ Not enough M15 candles")
        return None
    
    score = 0
    reasons = []
    
    if entry_trend == "BULLISH":
        direction = "BUY"
    elif entry_trend == "BEARISH":
        direction = "SELL"
    else:
        if len(m15) >= 5:
            last_close = float(m15["Close"].iloc[-1])
            prev_close = float(m15["Close"].iloc[-2])
            if last_close > prev_close:
                direction = "BUY"
                entry_trend = "BULLISH"
            else:
                direction = "SELL"
                entry_trend = "BEARISH"
        else:
            print("❌ Cannot determine direction")
            return None
    
    print(f"📊 Entry direction: {direction}")
    print(f"   Available trends: {trends}")
    
    # 1. HTF ማረጋገጫ
    if direction == "BUY" and bullish_count >= 1:
        score += 2
        reasons.append(f"✅ HTF Bullish ({bullish_count}/3)")
    elif direction == "SELL" and bearish_count >= 1:
        score += 2
        reasons.append(f"✅ HTF Bearish ({bearish_count}/3)")
    else:
        reasons.append("⚠️ HTF neutral")
        score += 1
    
    # 2. H1 ማረጋገጫ
    h1_trend = trends.get("H1", "NEUTRAL")
    if direction == "BUY" and h1_trend == "BULLISH":
        score += 1
        reasons.append("✅ H1 Bullish")
    elif direction == "SELL" and h1_trend == "BEARISH":
        score += 1
        reasons.append("✅ H1 Bearish")
    else:
        reasons.append("⚠️ H1 neutral")
        score += 0
    
    # 3. Supply & Demand
    supply_zones, demand_zones = find_supply_demand_improved(h1, lookback=60)
    price = float(m15["Close"].iloc[-1])
    
    atr_series = ATR(m15, period=10)
    if atr_series.empty:
        atr_value = 1.0
    else:
        atr_value = float(atr_series.iloc[-1])
        if not np.isfinite(atr_value) or atr_value <= 0:
            atr_value = 1.0
    
    zone_found = False
    if direction == "BUY" and demand_zones:
        zone_type, zone_strength = is_near_zone(price, demand_zones, atr_value, "BUY")
        if zone_type == "DEMAND":
            score += min(zone_strength, 3)
            reasons.append(f"✅ Near Demand (Strength: {zone_strength}/10)")
            zone_found = True
    elif direction == "SELL" and supply_zones:
        zone_type, zone_strength = is_near_zone(price, supply_zones, atr_value, "SELL")
        if zone_type == "SUPPLY":
            score += min(zone_strength, 3)
            reasons.append(f"✅ Near Supply (Strength: {zone_strength}/10)")
            zone_found = True
    
    if not zone_found and (demand_zones or supply_zones):
        reasons.append("⚠️ Not near a zone")
        score += 0
    
    # 4. Trendline Breakout
    upper_tl, lower_tl = find_trendlines(m15, lookback=15)
    tl_broken = False
    
    if direction == "BUY" and lower_tl:
        is_broken, expected = check_trendline_break(m15, lower_tl, direction="UP")
        if is_broken:
            score += 1
            reasons.append("✅ Trendline Breakout UP")
            tl_broken = True
    elif direction == "SELL" and upper_tl:
        is_broken, expected = check_trendline_break(m15, upper_tl, direction="DOWN")
        if is_broken:
            score += 1
            reasons.append("✅ Trendline Breakout DOWN")
            tl_broken = True
    
    if not tl_broken:
        reasons.append("⚠️ No trendline breakout")
        score += 0
    
    # 5. ሻማ ቅርጾች
    candle_score, candle_reasons = detect_candlestick_patterns(m15, direction)
    score += candle_score
    reasons.extend(candle_reasons)
    
    # 6. ፊቦናቺ
    fib_score, fib_reason = fibonacci_levels(h1)
    score += fib_score
    if fib_reason:
        reasons.append(fib_reason)
    
    print(f"📊 Total Score: {score}/14 (Min: {min_score})")
    
    if score < min_score:
        print(f"❌ NO TRADE | Score {score} < {min_score}")
        print(f"   Reasons: {reasons}")
        return None
    
    # የመግቢያ ነጥቦች ማስላት
    if direction == "BUY":
        if demand_zones:
            entry = demand_zones[-1]["high"] * 1.001
            sl = demand_zones[-1]["low"] - atr_value * SL_PADDING
        else:
            entry = price
            sl = price - atr_value * 1.2
        
        risk = entry - sl
        if risk <= 0:
            risk = atr_value * 0.5
        tp1 = entry + risk * 1.5
        tp2 = entry + risk * 2.5
        
    else:
        if supply_zones:
            entry = supply_zones[-1]["low"] * 0.999
            sl = supply_zones[-1]["high"] + atr_value * SL_PADDING
        else:
            entry = price
            sl = price + atr_value * 1.2
        
        risk = sl - entry
        if risk <= 0:
            risk = atr_value * 0.5
        tp1 = entry - risk * 1.5
        tp2 = entry - risk * 2.5
    
    zone_info = None
    if direction == "BUY" and demand_zones:
        zone_info = {
            "type": "DEMAND",
            "low": demand_zones[-1]["low"],
            "high": demand_zones[-1]["high"],
            "strength": demand_zones[-1].get("strength", 0)
        }
    elif direction == "SELL" and supply_zones:
        zone_info = {
            "type": "SUPPLY",
            "low": supply_zones[-1]["low"],
            "high": supply_zones[-1]["high"],
            "strength": supply_zones[-1].get("strength", 0)
        }
    
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
        "zone": zone_info,
        "atr": atr_value,
        "trends": trends
    }


# =========================================================
# JOURNAL
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
    with open(JOURNAL_FILE, "w", encoding="utf-8") as file:
        json.dump(journal, file, indent=2, ensure_ascii=False)


# =========================================================
# CHECK OPEN TRADES
# =========================================================

def check_open_trades(journal):
    still_open = []

    for trade in journal["open"]:
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
        hit_tp1 = False
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
                elif c_high >= trade["tp1"]:
                    result = "WIN"
                    exit_price = trade["tp1"]
                    hit_tp1 = True
                    break
                elif c_low <= trade["sl"]:
                    result = "LOSS"
                    exit_price = trade["sl"]
                    break
            else:
                if c_low <= trade["tp2"]:
                    result = "WIN"
                    exit_price = trade["tp2"]
                    hit_tp2 = True
                    break
                elif c_low <= trade["tp1"]:
                    result = "WIN"
                    exit_price = trade["tp1"]
                    hit_tp1 = True
                    break
                elif c_high >= trade["sl"]:
                    result = "LOSS"
                    exit_price = trade["sl"]
                    break

        if result:
            trade["result"] = result
            trade["exit"] = exit_price
            trade["closed_at"] = datetime.now(timezone.utc).isoformat()
            trade["hit_tp1"] = hit_tp1
            trade["hit_tp2"] = hit_tp2
            
            journal["closed"].append(trade)
            LAST_CLOSED_ENTRY[trade["symbol"]] = trade["entry"]
            
            if hit_tp1 or hit_tp2:
                tp_msg = f"📢 {trade['symbol']} {trade['direction']} UPDATE\n"
                if hit_tp2:
                    tp_msg += f"✅ TP2 HIT at {trade['tp2']:.4f} 🎉\n"
                elif hit_tp1:
                    tp_msg += f"✅ TP1 HIT at {trade['tp1']:.4f} 🎯\n"
                tp_msg += f"📌 Result: {result}"
                send_telegram(tp_msg)
            
            print(f"📕 CLOSED: {trade['symbol']} = {result}")
        else:
            still_open.append(trade)

    journal["open"] = still_open


# =========================================================
# STATISTICS
# =========================================================

def get_statistics(journal):
    wins = sum(1 for trade in journal["closed"] if trade.get("result") == "WIN")
    losses = sum(1 for trade in journal["closed"] if trade.get("result") == "LOSS")
    total = wins + losses
    win_rate = (wins / total * 100) if total > 0 else 0
    
    total_pips = 0
    for trade in journal["closed"]:
        if trade.get("exit") and trade.get("entry"):
            pips = abs(trade["exit"] - trade["entry"])
            total_pips += pips
    
    avg_pips = total_pips / total if total > 0 else 0

    return wins, losses, total, win_rate, avg_pips


# =========================================================
# TELEGRAM MESSAGE
# =========================================================

def make_message(signal, journal):
    wins, losses, total, win_rate, avg_pips = get_statistics(journal)
    confirmation = "\n".join(f"• {reason}" for reason in signal["reasons"])
    
    zone_info = ""
    if signal.get("zone"):
        z = signal["zone"]
        zone_info = f"\n📍 {z['type']} Zone: {z['low']:.4f} - {z['high']:.4f} (Strength: {z['strength']}/10)"
    
    trend_info = ""
    if signal.get("trends"):
        t = signal["trends"]
        trend_info = f"\n📊 Trends: {', '.join([f'{k}={v}' for k, v in t.items()])}"
    
    return f"""
🚨 {signal['symbol']} SIGNAL

📌 {signal["direction"]} — {signal["symbol"]}

📍 Entry: {signal["entry"]:.4f}
🛑 SL: {signal["sl"]:.4f}
🎯 TP1: {signal["tp1"]:.4f}
🎯 TP2: {signal["tp2"]:.4f}
{zone_info}
{trend_info}

📊 Score: {signal["score"]}/14

✅ Confirmation:
{confirmation}

🏆 PERFORMANCE
📈 Win Rate: {win_rate:.1f}%
📊 Total: {total} | ✅ Wins: {wins} | ❌ Losses: {losses}
📏 Avg: {avg_pips:.1f} pips/trade

⚠️ Risk: {RISK_PER_TRADE}% per trade
""".strip()


# =========================================================
# CONTINUOUS BOT LOOP
# =========================================================

def run_continuous_bot():
    print("==========================================")
    print("🚀 MULTI-MARKET BOT v3.0 (FAST MODE)")
    print("✅ Markets: XAU/USD, BTC/USD, GBP/USD, USD/JPY")
    print(f"✅ Min Score: {MIN_SCORE}/14 (BTC: {BTC_MIN_SCORE})")
    print(f"✅ Cooldown: {COOLDOWN_HOURS}h (BTC: {BTC_COOLDOWN_HOURS}h)")
    print("✅ Cache cleared on every request")
    print("==========================================")

    while True:
        try:
            journal = load_journal()

            if journal.get("open"):
                check_open_trades(journal)
                save_journal(journal)

            open_count = len(journal.get("open", []))
            if open_count >= MAX_OPEN_TRADES:
                print(f"⏳ Max open trades: {open_count}/{MAX_OPEN_TRADES}")
                time.sleep(30)
                continue

            for symbol_name, symbol_config in SYMBOLS.items():
                already_open = {trade["symbol"] for trade in journal.get("open", [])}

                if symbol_name not in already_open:
                    is_btc = (symbol_name == "BTC/USD")
                    cooldown = BTC_COOLDOWN_HOURS if is_btc else COOLDOWN_HOURS
                    max_price_change = BTC_MAX_PRICE_CHANGE if is_btc else MAX_PRICE_CHANGE_PERCENT
                    
                    # Cooldown
                    last_signal_time = SIGNAL_MEMORY.get(symbol_name)
                    if last_signal_time:
                        try:
                            last_dt = datetime.fromisoformat(last_signal_time)
                            hours_passed = (datetime.now() - last_dt).total_seconds() / 3600
                            if hours_passed < cooldown:
                                print(f"⏳ Cooldown: {symbol_name} ({hours_passed:.1f}h)")
                                continue
                        except Exception:
                            pass
                    
                    # Price Gap
                    last_entry = LAST_CLOSED_ENTRY.get(symbol_name)
                    if last_entry:
                        test_df = get_data(symbol_config["ticker"], "1d", "5m")
                        if not test_df.empty:
                            current_price = float(test_df["Close"].iloc[-1])
                            percent_change = abs((current_price - last_entry) / last_entry * 100)
                            if percent_change > max_price_change:
                                print(f"⏳ Price gap: {symbol_name} ({percent_change:.2f}%)")
                                continue

                    signal = analyze_improved(symbol_name, symbol_config)
                    if signal:
                        message = make_message(signal, journal)
                        if send_telegram(message):
                            journal["open"].append(signal)
                            save_journal(journal)
                            SIGNAL_MEMORY[symbol_name] = datetime.now().isoformat()
                            print(f"✅ SIGNAL SENT: {symbol_name}")

        except Exception as error:
            print(f"❌ Error: {repr(error)}")

        time.sleep(30)


# =========================================================
# START
# =========================================================

if __name__ == "__main__":
    run_continuous_bot()
