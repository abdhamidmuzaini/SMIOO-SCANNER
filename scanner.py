import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np

def send_telegram(message):
    token = os.environ.get('TELEGRAM_TOKEN')
    chat_id = os.environ.get('CHAT_ID')
    if not token or not chat_id:
        print("Telegram Token atau Chat ID belum diset.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {'chat_id': chat_id, 'text': message, 'parse_mode': 'Markdown'}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Gagal kirim telegram: {e}")

def calculate_dynamic_tp_sl(df, current_price):
    try:
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean().iloc[-1]
        
        if pd.isna(atr) or atr <= 0:
            sl = current_price * 0.95
            tp = current_price * 1.10
        else:
            sl = current_price - (1.5 * atr)
            tp = current_price + (3.0 * atr)
            
        return round(sl, 2), round(tp, 2)
    except Exception:
        return round(current_price * 0.95, 2), round(current_price * 1.10, 2)

def load_tickers():
    """Membaca daftar ratusan saham dari tickers.txt secara fleksibel (Newline, Koma, atau Spasi)"""
    tickers_list = []
    try:
        if os.path.exists('tickers.txt'):
            with open('tickers.txt', 'r') as f:
                content = f.read()
            
            # Bersihkan karakter aneh lalu pisahkan berdasarkan baris, koma, atau spasi
            cleaned = content.replace(',', ' ').replace('\n', ' ')
            tokens = cleaned.split()
            
            for t in tokens:
                clean_t = t.strip().upper()
                if clean_t and clean_t not in tickers_list:
                    tickers_list.append(clean_t)
    except Exception as e:
        print(f"Gagal baca tickers.txt: {e}")
        
    return tickers_list

def run_scanner():
    raw_tickers = load_tickers()
    if not raw_tickers:
        print("File tickers.txt kosong atau tidak terbaca.")
        return
        
    tickers = [t + ".JK" for t in raw_tickers]
    signals = []
    
    for symbol in tickers:
        try:
            ticker_obj = yf.Ticker(symbol)
            df = ticker_obj.history(period="3mo")
            if df is None or len(df) < 20:
                continue
                
            df['MA20'] = df['Close'].rolling(window=20).mean()
            current_price = float(df['Close'].iloc[-1])
            prev_price = float(df['Close'].iloc[-2])
            ma20 = float(df['MA20'].iloc[-1])
            prev_ma20 = float(df['MA20'].iloc[-2])
            
            # Filter Range Harga Opsional (sesuaikan jika perlu, misal 70 sampai 10000)
            if 70 <= current_price <= 10000:
                if prev_price <= prev_ma20 and current_price > ma20:
                    ticker_name = symbol.replace(".JK", "")
                    sl, tp = calculate_dynamic_tp_sl(df, current_price)
                    signals.append(f"🔹 *{ticker_name}*\n  • Harga Masuk: {current_price:.0f}\n  • 🔴 Stop Loss: {sl}\n  • 🟢 Take Profit: {tp}")
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    if signals:
        message = "🚨 **SWING DYNAMIC SIGNAL** 🚨\n📅 *Tanggal:* Hari Ini\n\n" + "\n\n".join(signals) + "\n\n_Cek manual chart sebelum eksekusi!_"
        send_telegram(message)
    else:
        print("Tidak ada sinyal yang memenuhi kriteria hari ini.")

if __name__ == "__main__":
    run_scanner()
