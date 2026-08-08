"""
SMIOO-SCANNER - scanner_qml.py
BOT ASTA LOGIC + ML (Random Forest) + Q-Learning (Reinforcement Learning)
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
from sklearn.model_selection import train_test_split
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
    3. Volume > Volume_MA20 * 1.5
    """
    if df is None or df.empty or len(df) < 3:
        return None
    
    df = get_indicators(df)
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 1. PSAR flip
    if not (prev['Close'] <= prev['PSAR'] and last['Close'] > last['PSAR']):
        return None
    
    # 2. Break Line
    break_line = prev['PSAR']
    if last['Close'] <= break_line:
        return None
    
    # 3. Volume > 1.5x MA20
    volume_ratio = last['Volume'] / last['Volume_MA20'] if last['Volume_MA20'] > 0 else 1
    if volume_ratio < 1.5:
        return None
    
    # 4. Price < 1000
    if last['Close'] >= 1000:
        return None
    
    # 5. Avg Value > 2M
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

# ==================== MACHINE LEARNING (Random Forest) ====================
def load_or_train_model():
    """Load model dari file .pkl, kalo gak ada train dari dummy data"""
    try:
        if os.path.exists(MODEL_FILE):
            with open(MODEL_FILE, 'rb') as f:
                model = pickle.load(f)
            logger.info("Model loaded from model.pkl")
            return model
        else:
            logger.warning("model.pkl not found, training dummy model")
            return train_dummy_model()
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return train_dummy_model()

def train_dummy_model():
    """Training dummy model dengan data acak"""
    np.random.seed(42)
    n = 1000
    X = np.random.rand(n, 5)
    # Target: 1 = profit > 5%, 0 = loss
    y = (X[:, 0] * 0.3 + X[:, 1] * 0.4 + X[:, 2] * 0.2 > 0.5).astype(int)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    logger.info("Dummy model trained")
    return model

def predict_pf(model, features):
    """Predictive Score (0-100) dari Random Forest"""
    if model is None:
        return 50
    try:
        features = np.array(features).reshape(1, -1)
        proba = model.predict_proba(features)[0][1]
        return int(min(max(proba * 100, 0), 100))
    except Exception as e:
        logger.error(f"PF prediction error: {e}")
        return 50

# ==================== Q-LEARNING (Reinforcement Learning) ====================
class QLearning:
    def __init__(self, actions=['HOLD', 'SELL'], alpha=0.1, gamma=0.9, epsilon=0.2):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}
        self.load_q_table()
    
    def get_state(self, signal):
        """Buat state dari sinyal"""
        # State: (rsi_level, volume_level, pf_level)
        rsi_level = 'high' if signal['rsi'] > 70 else 'mid' if signal['rsi'] > 50 else 'low'
        vol_level = 'high' if signal['volume_ratio'] > 2 else 'mid' if signal['volume_ratio'] > 1 else 'low'
        pf_level = 'high' if signal['pf'] > 80 else 'mid' if signal['pf'] > 60 else 'low'
        return f"{rsi_level}_{vol_level}_{pf_level}"
    
    def get_action(self, state):
        """Pilih aksi berdasarkan Q-Table"""
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in self.actions}
        
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)  # Eksplorasi
        else:
            # Eksploitasi: pilih aksi dengan nilai tertinggi
            return max(self.q_table[state], key=self.q_table[state].get)
    
    def update(self, state, action, reward, next_state):
        """Update Q-Table dengan Q-Learning"""
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0 for a in self.actions}
        
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
        
        self.save_q_table()
    
    def get_recommendation(self, signal):
        """Kasih rekomendasi HOLD atau SELL"""
        state = self.get_state(signal)
        action = self.get_action(state)
        return action
    
    def load_q_table(self):
        """Load Q-Table dari file JSON"""
        if os.path.exists(Q_TABLE_FILE):
            try:
                with open(Q_TABLE_FILE, 'r') as f:
                    self.q_table = json.load(f)
                logger.info("Q-Table loaded")
            except:
                self.q_table = {}
    
    def save_q_table(self):
        """Save Q-Table ke file JSON"""
        try:
            with open(Q_TABLE_FILE, 'w') as f:
                json.dump(self.q_table, f)
        except Exception as e:
            logger.error(f"Error saving Q-Table: {e}")

# ==================== MAIN ====================
def main():
    logger.info(f"Starting SMIOO-SCANNER at {NOW}")
    
    # 1. Load tickers
    tickers = load_tickers(TICKER_FILE)
    if not tickers:
        send_telegram(f"❌ {NOW} - tickers.txt tidak ditemukan!")
        return
    
    # 2. Load ML model
    model = load_or_train_model()
    
    # 3. Init Q-Learning
    ql = QLearning()
    
    # 4. Screening
    results = []
    for ticker in tickers:
        logger.info(f"Scanning {ticker}...")
        df = get_stock_data(ticker, period='3mo')
        if df is None or df.empty:
            continue
        
        signal = check_break_psar_bot_asta(df)
        if signal is None:
            continue
        
        # Hitung PF (ML)
        features = [
            signal['rsi'] / 100,
            signal['volume_ratio'] / 5,
            signal['change'] / 10,
            signal['price'] / 1000,
            signal['avg_value'] / 10_000_000_000
        ]
        pf_score = predict_pf(model, features)
        signal['pf'] = pf_score
        
        # Q-Learning: rekomendasi HOLD atau SELL
        recommendation = ql.get_recommendation(signal)
        
        results.append({
            'ticker': ticker.replace('.JK', ''),
            'price': signal['price'],
            'rsi': signal['rsi'],
            'change': signal['change'],
            'pf': pf_score,
            'volume_ratio': signal['volume_ratio'],
            'avg_value_b': signal['avg_value'] / 1_000_000_000,
            'break_line': signal['break_line'],
            'recommendation': recommendation,
            'date': signal['date']
        })
    
    results.sort(key=lambda x: x['pf'], reverse=True)
    
    # 5. Kirim ke Telegram
    if not results:
        msg = f"📭 *{NOW}*\nTidak ada sinyal Break PSAR hari ini."
        send_telegram(msg)
        logger.info("No signals found")
        return
    
    msg = f"📈 *SMIOO-SCANNER (BOT ASTA + ML + QL) - {NOW}*\n"
    msg += f"Total Sinyal: {len(results)}\n"
    msg += "==============================\n\n"
    
    for r in results:
        msg += f"*{r['ticker']}* ({r['price']:.0f}) | PF: {r['pf']} | RSI: {r['rsi']:.0f}\n"
        msg += f"  Δ: {r['change']:.1f}% | Vol: {r['volume_ratio']:.1f}x | Avg: {r['avg_value_b']:.1f}B\n"
        msg += f"  Break Line: {r['break_line']:.0f}\n"
        msg += f"  QL: *{r['recommendation']}*\n"
        msg += f"  Target: {r['price'] * 1.06:.0f} (+6%) | Hold: 3 hari\n\n"
    
    msg += "==============================\n"
    msg += "⚠️ *Tetap DYOR!* Hasil hanya referensi awal."
    
    send_telegram(msg)
    logger.info(f"Done. {len(results)} signals found.")

if __name__ == "__main__":
    main()
