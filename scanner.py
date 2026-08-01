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
    """Menghitung Stop Loss dan Take Profit dinamis berdasarkan ATR (Volatility)"""
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
            # Fallback jika data kurang
            sl = current_price * 0.95
            tp = current_price * 1.10
        else:
            # Stop Loss 1.5 x ATR di bawah harga, Take Profit 3 x ATR di atas harga
            sl = current_price - (1.5 * atr)
            tp = current_price + (3.0 * atr)
            
        return round(sl, 2), round(tp, 2)
    except Exception:
        return round(current_price * 0.95, 2), round(current_price * 1.10, 2)

def run_scanner():
    # Daftar contoh saham LQ45 / pilihan untuk di-scan (bisa disesuaikan list lu)
    tickers = ["BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "ADRO.JK", "PTBA.JK", "ANTM.JK", "GOTO.JK", "ASII.JK", "UNVR.JK"]
    
    signals = []
    
    for symbol in tickers:
        try:
            df = yf.Ticker(symbol).history(period="3mo")
            if len(df) < 20:
                continue
                
            # Contoh logika sederhana Sinyal Swing (MA Cross / Momentum)
            df['MA20'] = df['Close'].rolling(window=20).mean()
            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2]
            ma20 = df['MA20'].iloc[-1]
            
            # Syarat Sinyal: Harga menembus ke atas MA20 atau kondisi momentum tertentu
            if prev_price < df['MA20'].iloc[-2] and current_price > ma20:
                ticker_name = symbol.replace(".JK", "")
                sl, tp = calculate_dynamic_tp_sl(df, current_price)
                
                signals.append(f"🔹 *{ticker_name}*\n  • Harga Masuk: {current_price:.0f}\n  • 🔴 Stop Loss: {sl}\n  • 🟢 Take Profit: {tp}")
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    # Kirim hasil ke Telegram
    if signals:
        message = "🚨 **SWING DYNAMIC SIGNAL** 🚨\n📅 *Tanggal:* Hari Ini\n\n" + "\n\n".join(signals) + "\n\n_Cek manual chart sebelum eksekusi!_"
        send_telegram(message)
    else:
        print("Tidak ada sinyal yang memenuhi kriteria hari ini.")

if __name__ == "__main__":
    run_scanner()
