"""
SMIOO-SCANNER - scanner_qml.py
FULL VERSION - Break Falling PSAR + PSAR Age + Predictive Score (PF)
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
    """Kirim pesan ke Telegram pake requests (sync)"""
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logger.error("Telegram token or chat ID missing!")
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
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
    """Baca daftar saham dari tickers.txt"""
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
    """Download data saham dari Yahoo Finance"""
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

# ==================== INDIKATOR TEKNIKAL ====================

def calculate_rsi(series, period=14):
    """Hitung RSI"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_psar(df, acceleration=0.02, maximum=0.2):
    """Hitung Parabolic SAR"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
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

def calculate_atr(df, period=14):
    """Hitung ATR"""
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(period).mean()
    return atr

def get_indicators(df):
    """Hitung semua indikator"""
    df = df.copy()
    df['RSI'] = calculate_rsi(df['Close'])
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['Volume_MA20'] = df['Volume'].rolling(20).mean()
    df['ATR'] = calculate_atr(df)
    df['PSAR'] = calculate_psar(df)
    return df

# ==================== SCREENER ====================

def get_psar_age(df):
    """Hitung berapa hari PSAR sudah di atas harga (falling)"""
    if df is None or df.empty or 'PSAR' not in df.columns:
        return 0
    
    age = 0
    for i in range(len(df)-1, 0, -1):
        if df.iloc[i]['Close'] < df.iloc[i]['PSAR']:
            age += 1
        else:
            break
    return age

def check_break_psar(df, max_psar_age=3):
    """
    Cek Break Falling PSAR:
    - PSAR kemarin di atas harga (trend turun)
    - PSAR hari ini di bawah harga (trend naik)
    - PSAR Age <= max_psar_age (biar gak kasih sinyal bekas)
    + semua filter tambahan
    """
    if df is None or df.empty or len(df) < 3:
        return None
    
    df = get_indicators(df)
    
    # Ambil 2 hari terakhir
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 1. PSAR Age (biar gak kasih sinyal bekas)
    psar_age = get_psar_age(df)
    if psar_age > max_psar_age:
        logger.debug(f"PSAR Age too old: {psar_age} days")
        return None
    
    # 2. Break Falling PSAR (PSAR kemarin di atas harga, hari ini di bawah)
    if not (prev['Close'] <= prev['PSAR'] and last['Close'] > last['PSAR']):
        return None
    
    # 3. Price < 1000
    if last['Close'] >= 1000:
        return None
    
    # 4. Avg Value 20H > 2 Miliar
    avg_value = last['Volume_MA20'] * last['Close']
    if avg_value < 2_000_000_000:
        return None
    
    # 5. Close < ARA (20% dari harga sebelumnya)
    ara = prev['Close'] * 1.20
    if last['Close'] >= ara:
        return None
    
    # 6. Volume > Volume H-1
    if last['Volume'] <= prev['Volume']:
        return None
    
    # 7. RSI > RSI H-1
    if last['RSI'] <= prev['RSI']:
        return None
    
    # Semua lolos
    return {
        'ticker': None,  # diisi di main
        'price': last['Close'],
        'rsi': last['RSI'],
        'volume': last['Volume'],
        'volume_ma20': last['Volume_MA20'],
        'avg_value': avg_value,
        'psar': last['PSAR'],
        'atr': last['ATR'],
        'change': (last['Close'] - prev['Close']) / prev['Close'] * 100,
        'psar_age': psar_age,
        'date': last.name.strftime('%Y-%m-%d')
    }

# ==================== PREDICTIVE SCORE (PF) ====================

def load_model():
    """Load model dari file .pkl, kalo gak ada pake dummy"""
    try:
        if os.path.exists(MODEL_FILE):
            with open(MODEL_FILE, 'rb') as f:
                model = pickle.load(f)
            logger.info("Model loaded from model.pkl")
            return model
        else:
            logger.warning("model.pkl not found, using dummy model")
            return train_dummy_model()
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return train_dummy_model()

def train_dummy_model():
    """Dummy model kalo gak ada model.pkl"""
    np.random.seed(42)
    n = 700
    X = np.random.rand(n, 4)
    y = (X[:, 0] * 0.3 + X[:, 1] * 0.4 + X[:, 2] * 0.2 > 0.5).astype(int)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    logger.info("Dummy model trained")
    return model

def predict_pf(model, features):
    """Hitung Predictive Score (0-100)"""
    if model is None:
        return 50
    try:
        features = np.array(features).reshape(1, -1)
        proba = model.predict_proba(features)[0][1]
        return int(min(max(proba * 100, 0), 100))
    except Exception as e:
        logger.error(f"PF prediction error: {e}")
        return 50

# ==================== MAIN ====================

def main():
    logger.info(f"Starting SMIOO-SCANNER at {NOW}")
    
    # 1. Load tickers
    tickers = load_tickers(TICKER_FILE)
    if not tickers:
        send_telegram(f"❌ {NOW} - tickers.txt tidak ditemukan!")
        return
    
    # 2. Load model PF
    model = load_model()
    
    # 3. Screening
    results = []
    for ticker in tickers:
        logger.info(f"Scanning {ticker}...")
        df = get_stock_data(ticker, period='3mo')
        if df is None or df.empty:
            continue
        
        signal = check_break_psar(df, max_psar_age=3)  # max 3 hari
        if signal is None:
            continue
        
        # Hitung PF
        volume_ratio = signal['volume'] / signal['volume_ma20'] if signal['volume_ma20'] > 0 else 1
        features = [
            signal['rsi'] / 100,
            volume_ratio / 5,
            signal['change'] / 10,
            signal['atr'] / 100 if signal['atr'] > 0 else 0.01
        ]
        pf_score = predict_pf(model, features)
        
        # Klasifikasi trade (dengan PSAR Age bonus)
        psar_bonus = 5 if signal['psar_age'] <= 1 else 0  # sinyal fresh dapet bonus
        pf_adjusted = min(pf_score + psar_bonus, 100)
        
        if pf_adjusted >= 85 and signal['rsi'] > 60 and volume_ratio > 2:
            trade_type = "🔥 FAST TRADE (P1-P3)"
            hold_time = "3 hari"
            tp = 6
        elif pf_adjusted >= 70 and 40 <= signal['rsi'] <= 60:
            trade_type = "🟡 MAX PROFIT (P4-P7)"
            hold_time = "7 hari"
            tp = 10
        else:
            continue  # skip sinyal jelek
        
        results.append({
            'ticker': ticker.replace('.JK', ''),
            'price': signal['price'],
            'rsi': signal['rsi'],
            'change': signal['change'],
            'pf': pf_adjusted,
            'volume_ratio': volume_ratio,
            'avg_value_b': signal['avg_value'] / 1_000_000_000,
            'psar_age': signal['psar_age'],
            'trade_type': trade_type,
            'hold_time': hold_time,
            'tp': tp,
            'date': signal['date']
        })
    
    # 4. Sort by PF
    results.sort(key=lambda x: x['pf'], reverse=True)
    
    # 5. Kirim ke Telegram
    if not results:
        msg = f"📭 *{NOW}*\nTidak ada sinyal Break Falling PSAR hari ini."
        send_telegram(msg)
        logger.info("No signals found")
        return
    
    msg = f"📈 *SMIOO-SCANNER - {NOW}*\n"
    msg += f"Total Sinyal: {len(results)}\n"
    msg += "==============================\n\n"
    
    for r in results:
        msg += f"*{r['ticker']}* ({r['price']:.0f}) | PF: {r['pf']} | RSI: {r['rsi']:.0f}\n"
        msg += f"  Δ: {r['change']:.1f}% | Vol Ratio: {r['volume_ratio']:.1f}x | Avg: {r['avg_value_b']:.1f}B\n"
        msg += f"  PSAR Age: {r['psar_age']} hari | {r['trade_type']}\n"
        msg += f"  Target: {r['price'] * (1 + r['tp']/100):.0f} (+{r['tp']}%) | Hold: {r['hold_time']}\n"
        msg += "\n"
    
    msg += "==============================\n"
    msg += "⚠️ *Tetap DYOR!* Hasil hanya referensi awal."
    
    send_telegram(msg)
    logger.info(f"Done. {len(results)} signals found.")

if __name__ == "__main__":
    main()
