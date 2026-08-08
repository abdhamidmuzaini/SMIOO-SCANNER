"""
SMIOO-SCANNER - scanner_qml.py
BOT ASTA LOGIC - Break Line + Volume Confirmation
"""

import os
import sys
import logging
import pytz
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from sklearn.ensemble import RandomForestClassifier
import pickle

# ==================== KONFIGURASI ====================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

WITA = pytz.timezone('Asia/Makassar')
NOW = datetime.now(WITA).strftime('%Y-%m-%d %H:%M:%S WITA')

TICKER_FILE = "tickers.txt"
MODEL_FILE = "model.pkl"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== TELEGRAM ====================
def send_telegram(message):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logger.error("Telegram token or chat ID missing!")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code == 200:
            logger.info("Telegram message sent")
        else:
            logger.error(f"Telegram error: {r.text}")
    except Exception as e:
        logger.error(f"Telegram error: {e}")

# ==================== LOAD TICKERS ====================
def load_tickers(file_path):
    try:
        if not os.path.exists(file_path):
            logger.error(f"File {file_path} not found!")
            return []
        with open(file_path, 'r') as f:
            tickers = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        logger.info(f"Loaded {len(tickers)} tickers from {file_path}")
        return tickers
    except Exception as e:
        logger.error(f"Error loading tickers: {e}")
        return []

# ==================== DOWNLOAD DATA ====================
def get_stock_data(ticker, period='3mo'):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            logger.warning(f"No data for {ticker}")
            return None
        return df
    except Exception as e:
        logger.error(f"Error downloading {ticker}: {e}")
        return None

# ==================== INDIKATOR ====================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_psar(df, acceleration=0.02, maximum=0.2):
    high = df['High']; low = df['Low']; close = df['Close']
    psar = high.copy()
    psar_bull = True
    af = acceleration
    ep = low.iloc[0] if psar_bull else high.iloc[0]
    psar.iloc[0] = close.iloc[0]
    for i in range(1, len(df)):
        if psar_bull:
            psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
            if low.iloc[i] <= psar.iloc[i]:
                psar_bull = False
                psar.iloc[i] = ep
                af = acceleration
                ep = low.iloc[i]
        else:
            psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
            if high.iloc[i] >= psar.iloc[i]:
                psar_bull = True
                psar.iloc[i] = ep
                af = acceleration
                ep = high.iloc[i]
    return psar

def get_indicators(df):
    df = df.copy()
    df['RSI'] = calculate_rsi(df['Close'])
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['Volume_MA20'] = df['Volume'].rolling(20).mean()
    df['PSAR'] = calculate_psar(df, acceleration=0.02, maximum=0.2)
    return df

# ==================== BOT ASTA LOGIC ====================

def check_break_psar_bot_asta(df):
    """
    Logika Bot Asta:
    1. PSAR flip (kemarin di atas harga, hari ini di bawah harga)
    2. Break Line = harga > PSAR kemarin (titik flip)
    3. Volume > Volume_MA20 * 1.5 (konfirmasi volume)
    4. Filter harga, ARA, RSI, dll
    """
    if df is None or df.empty or len(df) < 3:
        return None
    
    df = get_indicators(df)
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 1. PSAR flip (kemarin di atas harga, hari ini di bawah harga)
    if not (prev['Close'] <= prev['PSAR'] and last['Close'] > last['PSAR']):
        return None
    
    # 2. Break Line = harga > PSAR kemarin (titik flip)
    break_line = prev['PSAR']
    if last['Close'] <= break_line:
        logger.info("SKIP: Harga tidak menembus Break Line")
        return None
    
    # 3. Volume > Volume_MA20 * 1.5 (konfirmasi volume)
    volume_ratio = last['Volume'] / last['Volume_MA20'] if last['Volume_MA20'] > 0 else 1
    if volume_ratio < 1.5:
        logger.info(f"SKIP: Volume rendah ({volume_ratio:.1f}x)")
        return None
    
    # 4. Price < 1000
    if last['Close'] >= 1000:
        return None
    
    # 5. Avg Value 20H > 2 Miliar
    avg_value = last['Volume_MA20'] * last['Close']
    if avg_value < 2_000_000_000:
        return None
    
    # 6. Close < ARA
    ara = prev['Close'] * 1.20
    if last['Close'] >= ara:
        return None
    
    # 7. RSI > RSI H-1
    if last['RSI'] <= prev['RSI']:
        return None
    
    change = (last['Close'] - prev['Close']) / prev['Close'] * 100
    
    return {
        'price': last['Close'],
        'rsi': last['RSI'],
        'volume': last['Volume'],
        'volume_ma20': last['Volume_MA20'],
        'avg_value': avg_value,
        'change': change,
        'break_line': break_line,
        'volume_ratio': volume_ratio,
        'date': last.name.strftime('%Y-%m-%d')
    }

# ==================== MAIN ====================
def main():
    logger.info(f"Starting SMIOO-SCANNER at {NOW}")
    
    tickers = load_tickers(TICKER_FILE)
    if not tickers:
        send_telegram(f"❌ {NOW} - tickers.txt tidak ditemukan!")
        return
    
    results = []
    for ticker in tickers:
        logger.info(f"Scanning {ticker}...")
        df = get_stock_data(ticker, period='3mo')
        if df is None or df.empty:
            continue
        
        signal = check_break_psar_bot_asta(df)
        if signal is None:
            continue
        
        # PF dummy (nanti diganti pake model asli)
        pf_score = 50
        if signal['rsi'] > 60 and signal['change'] > 3:
            pf_score = 85
        elif signal['rsi'] > 50:
            pf_score = 70
        
        results.append({
            'ticker': ticker.replace('.JK', ''),
            'price': signal['price'],
            'rsi': signal['rsi'],
            'change': signal['change'],
            'pf': pf_score,
            'volume_ratio': signal['volume_ratio'],
            'avg_value_b': signal['avg_value'] / 1_000_000_000,
            'break_line': signal['break_line'],
            'date': signal['date']
        })
    
    results.sort(key=lambda x: x['pf'], reverse=True)
    
    if not results:
        msg = f"📭 *{NOW}*\nTidak ada sinyal Break PSAR hari ini."
        send_telegram(msg)
        logger.info("No signals found")
        return
    
    msg = f"📈 *SMIOO-SCANNER (BOT ASTA LOGIC) - {NOW}*\n"
    msg += f"Total Sinyal: {len(results)}\n"
    msg += "==============================\n\n"
    
    for r in results:
        msg += f"*{r['ticker']}* ({r['price']:.0f}) | PF: {r['pf']} | RSI: {r['rsi']:.0f}\n"
        msg += f"  Δ: {r['change']:.1f}% | Vol: {r['volume_ratio']:.1f}x | Avg: {r['avg_value_b']:.1f}B\n"
        msg += f"  Break Line: {r['break_line']:.0f}\n"
        msg += f"  Target: {r['price'] * 1.06:.0f} (+6%) | Hold: 3 hari\n\n"
    
    msg += "==============================\n"
    msg += "⚠️ *Tetap DYOR!* Hasil hanya referensi awal."
    
    send_telegram(msg)
    logger.info(f"Done. {len(results)} signals found.")

if __name__ == "__main__":
    main()
