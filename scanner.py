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
    tickers_list = []
    try:
        if os.path.exists('tickers.txt'):
            with open('tickers.txt', 'r') as f:
                content = f.read()
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
            if df is None or len(df) < 30:
                continue
                
            # Indikator Teknikal & Likuiditas
            df['SMIOO_Blue'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['SMIOO_Orange'] = df['SMIOO_Blue'].ewm(span=9, adjust=False).mean()
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
            df['Value_MA20'] = (df['Close'] * df['Volume']).rolling(window=20).mean()
            
            current_price = float(df['Close'].iloc[-1])
            blue_today = float(df['SMIOO_Blue'].iloc[-1])
            blue_prev = float(df['SMIOO_Blue'].iloc[-2])
            orange_today = float(df['SMIOO_Orange'].iloc[-1])
            orange_prev = float(df['SMIOO_Orange'].iloc[-2])
            
            ma5 = float(df['MA5'].iloc[-1])
            ma20 = float(df['MA20'].iloc[-1])
            vol_today = float(df['Volume'].iloc[-1])
            vol_ma20 = float(df['Vol_MA20'].iloc[-1])
            val_ma20 = float(df['Value_MA20'].iloc[-1])
            
            # 1. Filter Wajib Utama (Harga & Likuiditas)
            cond_price = 70 <= current_price < 1000
            cond_value = val_ma20 > 2_000_000_000
            
            if not (cond_price and cond_value):
                continue
                
            # Hitung TP, SL, dan Risk/Reward (RR)
            sl, tp = calculate_dynamic_tp_sl(df, current_price)
            risk = current_price - sl
            reward = tp - current_price
            rr = reward / risk if risk > 0 else 0
            
            # --- SISTEM SKORING TEKNIKAL (Maksimal 100 Poin) ---
            score = 0
            
            # A. Sinyal SMIOO (Golden Cross & Arah Menanjak / Slope Up) - Bobot: 30 Poin
            is_crossover = (blue_prev <= orange_prev) and (blue_today > orange_today)
            is_slope_up = blue_today > blue_prev
            if is_crossover and is_slope_up:
                score += 30
            elif is_slope_up:
                score += 15 # Masih ada momentum walau belum pas GC hari ini
                
            # B. Konfirmasi Tren Harga (Di atas MA5 & MA20) - Bobot: 25 Poin
            if current_price > ma5 and current_price > ma20:
                score += 25
            elif current_price > ma20:
                score += 15
                
            # C. Lonjakan Volume (Volume > Rata-rata 20 Hari) - Bobot: 25 Poin
            if vol_today > vol_ma20:
                score += 25
                
            # D. Kualitas Risk / Reward (RR Bagus >= 2.0) - Bobot: 20 Poin
            if rr >= 2.0:
                score += 20
            elif rr >= 1.5:
                score += 10
                
            # --- FILTER FINAL: HANYA LOLOS JIKA SKOR > 70 ---
            if score > 70:
                ticker_name = symbol.replace(".JK", "")
                signals.append(f"🔥 *{ticker_name}* (Score: {score} | RR: {rr:.1f})\n  • Harga: {current_price:.0f}\n  • 🔴 SL: {sl}\n  • 🟢 TP: {tp}")
                
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    if signals:
        message = "⭐ **HIGH SCORE SCREENER SIGNAL (Score > 70)** ⭐\n📅 *Tanggal:* Hari Ini\n\n" + "\n\n".join(signals) + "\n\n_Pastikan cek chart manual sebelum eksekusi!_"
        send_telegram(message)
    else:
        print("Tidak ada saham yang mencapai Score > 70 hari ini.")

if __name__ == "__main__":
    run_scanner()
