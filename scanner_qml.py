"""
SMIOO-SCANNER - scanner_qml.py
Strategi: Break Falling PSAR + Filter Teknikal + Predictive Score (PF)
Output: Rekomendasi Fast Trade / Max Profit ke Telegram
"""

import os
import sys
import logging
import pytz
from datetime import datetime, timedelta
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from telegram import Bot
from telegram.error import TelegramError
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ==================== KONFIGURASI ====================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

WITA = pytz.timezone('Asia/Makassar')
NOW = datetime.now(WITA).strftime('%Y-%m-%d %H:%M:%S WITA')

# File daftar saham
TICKER_FILE = "tickers.txt"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== FUNGSI BANTUAN ====================

def load_tickers(file_path):
    """Baca daftar saham dari tickers.txt"""
    try:
        with open(file_path, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(tickers)} tickers from {file_path}")
        return tickers
    except FileNotFoundError:
        logger.error(f"File {file_path} not found!")
        return []

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

def calculate_psar(df, acceleration=0.02, maximum=0.2):
    """Hitung Parabolic SAR"""
    try:
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
    except Exception as e:
        logger.error(f"PSAR calculation error: {e}")
        return None

def calculate_indicators(df):
    """Hitung indikator teknikal: RSI, MA, Volume, ATR"""
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Moving Average
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # Volume MA20 (buat rata-rata transaksi 20 hari)
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    return df

def get_break_psar_signal(df):
    """
    Deteksi Break Falling PSAR:
    - PSAR sebelumnya di atas harga (trend turun)
    - PSAR hari ini di bawah harga (trend naik)
    - Ini sinyal 'break' dari falling ke rising
    """
    if df is None or df.empty or len(df) < 3:
        return None
    
    df = calculate_indicators(df)
    df['PSAR'] = calculate_psar(df)
    
    if df['PSAR'] is None:
        return None
    
    # Ambil 2 hari terakhir
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # CEK BREAK FALLING PSAR:
    # Kondisi 1: PSAR kemarin di ATAS harga (trend turun)
    # Kondisi 2: PSAR hari ini di BAWAH harga (trend naik)
    psar_break = (prev['Close'] <= prev['PSAR']) and (last['Close'] > last['PSAR'])
    
    if not psar_break:
        return None
    
    # Filter tambahan (dari screener lo)
    # 1. Price < 1000
    if last['Close'] >= 1000:
        return None
    
    # 2. Avg Value 20H > 2 Miliar (dari Volume_MA20)
    avg_value = last['Volume_MA20'] * last['Close']  # dalam Rupiah
    if avg_value < 2_000_000_000:  # 2 Miliar
        return None
    
    # 3. Close < ARA (harga belum mentok di batas atas)
    # ARA = harga penutupan kemarin * 1.20 (untuk saham biasa)
    ara = prev['Close'] * 1.20
    if last['Close'] >= ara:
        return None
    
    # 4. Volume > Volume H-1
    if last['Volume'] <= prev['Volume']:
        return None
    
    # 5. RSI > RSI H-1
    if last['RSI'] <= prev['RSI']:
        return None
    
    # Semua filter lolos!
    return {
        'signal': 'BUY',
        'price': last['Close'],
        'psar': last['PSAR'],
        'rsi': last['RSI'],
        'volume': last['Volume'],
        'volume_ma20': last['Volume_MA20'],
        'avg_value': avg_value,
        'close_to_ma50': (last['Close'] - last['MA50']) / last['MA50'] * 100 if not pd.isna(last['MA50']) else 0,
        'atr': last['ATR'],
        'change': (last['Close'] - prev['Close']) / prev['Close'] * 100
    }

def train_rf_model():
    """Training model Random Forest untuk PF (Predictive Score)"""
    # Data dummy (nanti ganti dengan data backtest lo)
    np.random.seed(42)
    n_samples = 700
    
    # Fitur: RSI, Volume_Ratio, Close_to_MA50, ATR
    X = np.random.rand(n_samples, 4)
    y = (X[:, 0] * 0.3 + X[:, 1] * 0.4 + X[:, 2] * 0.2 > 0.5).astype(int)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model

def predict_pf(model, features):
    """Prediksi Predictive Score (PF)"""
    if model is None:
        return 50
    
    features = np.array(features).reshape(1, -1)
    proba = model.predict_proba(features)[0][1]
    pf_score = int(proba * 100)
    return min(max(pf_score, 0), 100)

def classify_trade(pf_score, rsi, volume_ratio, avg_value):
    """Klasifikasi Fast Trade vs Max Profit"""
    # Fast Trade: PF > 85, RSI > 60, Volume > 2x MA, Avg Value > 5M
    if pf_score > 85 and rsi > 60 and volume_ratio > 2 and avg_value > 5_000_000_000:
        return "🔥 FAST TRADE (P1-P3)", "3 hari", 6
    # Max Profit: PF 70-85, RSI 40-60, volume normal
    elif pf_score >= 70 and 40 <= rsi <= 60:
        return "🟡 MAX PROFIT (P4-P7)", "7 hari", 10
    # Skip: PF < 70
    else:
        return "🔴 SKIP", "-", 0

def send_telegram(message):
    """Kirim pesan ke Telegram"""
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logger.error("TELEGRAM_TOKEN or CHAT_ID not set!")
        return
    
    try:
        bot = Bot(token=TELEGRAM_TOKEN)
        bot.send_message(chat_id=CHAT_ID, text=message)
        logger.info("Message sent to Telegram")
    except TelegramError as e:
        logger.error(f"Telegram error: {e}")

def send_results_to_telegram(results):
    """Format dan kirim hasil screening ke Telegram"""
    if not results:
        send_telegram(f"📭 *{NOW}* - Tidak ada sinyal PSAR hari ini.")
        return
    
    message = f"📈 *SMIOO-SCANNER - {NOW}*\n"
    message += f"Total Sinyal: {len(results)}\n"
    message += "="*30 + "\n\n"
    
    for r in results:
        message += f"*{r['ticker']}* ({r['price']:.0f}) | PF: {r['pf_score']} | RSI: {r['rsi']:.0f} | Δ: {r['change']:.1f}%\n"
        message += f"  {r['trade_type']}\n"
        if r['tp'] > 0:
            message += f"  Target: {r['price'] * (1 + r['tp']/100):.0f} (+{r['tp']}%) | Hold: {r['hold_time']}\n"
        message += "\n"
    
    message += "="*30 + "\n"
    message += "⚠️ *Tetap DYOR!* Hasil hanya referensi awal.\n"
    
    send_telegram(message)

# ==================== MAIN ====================

def main():
    logger.info(f"Starting SMIOO-SCANNER at {NOW}")
    
    tickers = load_tickers(TICKER_FILE)
    if not tickers:
        logger.error("No tickers loaded. Exiting.")
        return
    
    # Load/Train RF model (nanti ganti pake model .pkl)
    model = train_rf_model()
    
    results = []
    for ticker in tickers:
        logger.info(f"Scanning {ticker}...")
        df = get_stock_data(ticker, period='3mo')
        if df is None or df.empty:
            continue
        
        signal = get_break_psar_signal(df)
        if signal is None:
            continue
        
        # Hitung PF (Predictive Score)
        volume_ratio = signal['volume'] / signal['volume_ma20'] if signal['volume_ma20'] > 0 else 0
        features = [
            signal['rsi'] / 100,
            volume_ratio / 5,
            signal['close_to_ma50'] / 20,
            signal['atr'] / 100 if signal['atr'] > 0 else 0
        ]
        pf_score = predict_pf(model, features)
        
        # Klasifikasi trade
        trade_type, hold_time, tp_percent = classify_trade(
            pf_score,
            signal['rsi'],
            volume_ratio,
            signal['avg_value']
        )
        
        # Skip kalo trade_type = SKIP
        if trade_type == "🔴 SKIP":
            continue
        
        results.append({
            'ticker': ticker.replace('.JK', ''),
            'price': signal['price'],
            'pf_score': pf_score,
            'rsi': signal['rsi'],
            'change': signal['change'],
            'trade_type': trade_type,
            'hold_time': hold_time,
            'tp': tp_percent,
            'avg_value': signal['avg_value'] / 1_000_000_000  # dalam Miliar
        })
    
    # Sort by PF (descending)
    results.sort(key=lambda x: x['pf_score'], reverse=True)
    
    # Kirim ke Telegram
    send_results_to_telegram(results)
    
    logger.info(f"Scanning completed. {len(results)} signals found.")

if __name__ == "__main__":
    main()
