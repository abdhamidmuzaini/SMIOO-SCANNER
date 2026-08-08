"""
SMIOO-SCANNER - scanner_qml.py
FINAL: Bot Asta Logic + PSAR Age + ML + Q-Learning
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
import json

# ==================== KONFIGURASI ====================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

WITA = pytz.timezone('Asia/Makassar')
NOW = datetime.now(WITA).strftime('%Y-%m-%d %H:%M:%S WITA')

TICKER_FILE = "tickers.txt"
MODEL_FILE = "model.pkl"
Q_TABLE_FILE = "q_table.json"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== TELEGRAM ====================
def send_telegram(message):
    if not TELEGRAM_TOKEN or not CHAT_ID:
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

# ==================== PSAR AGE ====================
def get_psar_age(df):
    """Cari hari pertama PSAR di bawah harga"""
    if df is None or df.empty or 'PSAR' not in df.columns:
        return 999
    for i in range(len(df)):
        if df.iloc[i]['Close'] > df.iloc[i]['PSAR']:
            return len(df) - 1 - i
    return 999

# ==================== BOT ASTA LOGIC + PSAR AGE ====================
def check_break_psar_bot_asta(df, max_psar_age=3):
    if df is None or df.empty or len(df) < 3:
        return None
    
    df = get_indicators(df)
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 1. PSAR flip
    if not (prev['Close'] <= prev['PSAR'] and last['Close'] > last['PSAR']):
        return None
    
    # 2. PSAR Age (skip kalo >3 hari)
    psar_age = get_psar_age(df)
    if psar_age > max_psar_age:
        logger.info(f"SKIP: PSAR Age {psar_age} > 3")
        return None
    
    # 3. Break Line
    break_line = prev['PSAR']
    if last['Close'] <= break_line:
        return None
    
    # 4. Volume > 1.5x MA20
    volume_ratio = last['Volume'] / last['Volume_MA20'] if last['Volume_MA20'] > 0 else 1
    if volume_ratio < 1.5:
        return None
    
    # 5. Price < 1000
    if last['Close'] >= 1000:
        return None
    
    # 6. Avg Value > 2M
    avg_value = last['Volume_MA20'] * last['Close']
    if avg_value < 2_000_000_000:
        return None
    
    # 7. Close < ARA
    ara = prev['Close'] * 1.20
    if last['Close'] >= ara:
        return None
    
    # 8. RSI > RSI H-1
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
        'psar_age': psar_age,
        'date': last.name.strftime('%Y-%m-%d')
    }

# ==================== MACHINE LEARNING ====================
def load_or_train_model():
    try:
        if os.path.exists(MODEL_FILE):
            with open(MODEL_FILE, 'rb') as f:
                model = pickle.load(f)
            logger.info("Model loaded from model.pkl")
            return model
    except:
        pass
    return train_dummy_model()

def train_dummy_model():
    np.random.seed(42)
    n = 1000
    X = np.random.rand(n, 5)
    y = (X[:, 0] * 0.3 + X[:, 1] * 0.4 + X[:, 2] * 0.2 > 0.5).astype(int)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    logger.info("Dummy model trained")
    return model

def predict_pf(model, features):
    if model is None:
        return 50
    try:
        features = np.array(features).reshape(1, -1)
        proba = model.predict_proba(features)[0][1]
        return int(min(max(proba * 100, 0), 100))
    except:
        return 50

# ==================== Q-LEARNING ====================
class QLearning:
    def __init__(self):
        self.q_table = {}
        self.load_q_table()
    
    def get_state(self, signal):
        rsi = 'high' if signal['rsi'] > 70 else 'mid' if signal['rsi'] > 50 else 'low'
        vol = 'high' if signal['volume_ratio'] > 2 else 'mid' if signal['volume_ratio'] > 1 else 'low'
        pf = 'high' if signal['pf'] > 80 else 'mid' if signal['pf'] > 60 else 'low'
        return f"{rsi}_{vol}_{pf}"
    
    def get_action(self, state):
        actions = ['HOLD', 'SELL']
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in actions}
        # Pilih aksi dengan nilai tertinggi
        return max(self.q_table[state], key=self.q_table[state].get)
    
    def load_q_table(self):
        if os.path.exists(Q_TABLE_FILE):
            try:
                with open(Q_TABLE_FILE, 'r') as f:
                    self.q_table = json.load(f)
            except:
                self.q_table = {}
    
    def save_q_table(self):
        try:
            with open(Q_TABLE_FILE, 'w') as f:
                json.dump(self.q_table, f)
        except:
            pass

# ==================== MAIN ====================
def main():
    logger.info(f"Starting SMIOO-SCANNER at {NOW}")
    
    tickers = load_tickers(TICKER_FILE)
    if not tickers:
        send_telegram(f"❌ {NOW} - tickers.txt tidak ditemukan!")
        return
    
    model = load_or_train_model()
    ql = QLearning()
    
    results = []
    for ticker in tickers:
        logger.info(f"Scanning {ticker}...")
        df = get_stock_data(ticker, period='3mo')
        if df is None or df.empty:
            continue
        
        signal = check_break_psar_bot_asta(df, max_psar_age=3)
        if signal is None:
            continue
        
        features = [
            signal['rsi'] / 100,
            signal['volume_ratio'] / 5,
            signal['change'] / 10,
            signal['price'] / 1000,
            signal['avg_value'] / 10_000_000_000
        ]
        pf = predict_pf(model, features)
        signal['pf'] = pf
        
        state = ql.get_state(signal)
        ql_action = ql.get_action(state)
        
        results.append({
            'ticker': ticker.replace('.JK', ''),
            'price': signal['price'],
            'rsi': signal['rsi'],
            'change': signal['change'],
            'pf': pf,
            'volume_ratio': signal['volume_ratio'],
            'avg_value_b': signal['avg_value'] / 1_000_000_000,
            'break_line': signal['break_line'],
            'psar_age': signal['psar_age'],
            'ql_action': ql_action,
            'date': signal['date']
        })
    
    results.sort(key=lambda x: x['pf'], reverse=True)
    
    if not results:
        send_telegram(f"📭 *{NOW}*\nTidak ada sinyal Break PSAR hari ini.")
        return
    
    msg = f"📈 *SMIOO-SCANNER (FINAL) - {NOW}*\n"
    msg += f"Total Sinyal: {len(results)}\n"
    msg += "==============================\n\n"
    
    for r in results:
        msg += f"*{r['ticker']}* ({r['price']:.0f}) | PF: {r['pf']} | RSI: {r['rsi']:.0f}\n"
        msg += f"  Δ: {r['change']:.1f}% | Vol: {r['volume_ratio']:.1f}x | Avg: {r['avg_value_b']:.1f}B\n"
        msg += f"  PSAR Age: {r['psar_age']} | QL: *{r['ql_action']}*\n"
        msg += f"  Target: {r['price'] * 1.06:.0f} (+6%) | Hold: 3 hari\n\n"
    
    msg += "==============================\n"
    msg += "⚠️ *Tetap DYOR!*"
    
    send_telegram(msg)
    logger.info(f"Done. {len(results)} signals found.")

if __name__ == "__main__":
    main()
