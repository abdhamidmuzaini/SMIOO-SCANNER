import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf

# ==========================================
# KONFIGURASI TELEGRAM
# ==========================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("CHAT_ID")

def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Token/Chat ID Telegram belum diset di Secrets!")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            print(f"Gagal kirim Telegram: {response.text}")
    except Exception as e:
        print(f"Error koneksi Telegram: {e}")

# ==========================================
# RUMUS INDIKATOR: SMIEO + ADAPTIVE TP & SL (ATR)
# ==========================================
def analyze_stock(ticker):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df.empty or len(df) < 50:
            return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        # 1. FILTER LIKUIDITAS & HARGA
        value_txn = close * volume
        avg_value_20 = value_txn.rolling(window=20).mean().iloc[-1]
        if avg_value_20 < 1_000_000_000: return None
        
        last_close = float(close.iloc[-1])
        if last_close > 2000: return None

        # 2. MENGHITUNG ATR (Average True Range) 14 HARI
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=14).mean() 

        # 3. RUMUS SMI ERGODIC OSCILLATOR
        long_len = 20
        short_len = 5
        sig_len = 5

        price_diff = close.diff()
        abs_price_diff = price_diff.abs()

        ema1_diff = price_diff.ewm(span=long_len, adjust=False).mean()
        ema2_diff = ema1_diff.ewm(span=short_len, adjust=False).mean()
        ema1_abs = abs_price_diff.ewm(span=long_len, adjust=False).mean()
        ema2_abs = ema1_abs.ewm(span=short_len, adjust=False).mean()

        smi_line = np.where(ema2_abs == 0, 0, (ema2_diff / ema2_abs) * 100)
        smi_line = pd.Series(smi_line, index=close.index)
        signal_line = smi_line.ewm(span=sig_len, adjust=False).mean()

        is_smi_cross = (smi_line.iloc[-1] > signal_line.iloc[-1]) and (smi_line.iloc[-2] <= signal_line.iloc[-2])
        is_smi_active = (smi_line.iloc[-1] > 0) and (smi_line.iloc[-1] > signal_line.iloc[-1])

        if not (is_smi_cross or is_smi_active):
            return None

        # 4. BACKTEST STATISTIK: Cek Realistis TP vs SL
        success_count = 0
        total_signals = 0
        
        for i in range(50, len(df) - 5):
            if smi_line.iloc[i] > signal_line.iloc[i]:
                total_signals += 1
                
                historical_target = close.iloc[i] + (1.5 * atr.iloc[i])
                historical_sl = close.iloc[i] - (1.0 * atr.iloc[i])
                
                win = False
                # Cek pergerakan harga maksimal 5 hari ke depan
                for day in range(1, 6):
                    future_idx = i + day
                    if future_idx < len(df):
                        # Jika harga tertinggi tembus Target Profit duluan
                        if high.iloc[future_idx] >= historical_target:
                            win = True
                            break
                        # Jika harga terendah tembus Stop Loss duluan
                        elif low.iloc[future_idx] <= historical_sl:
                            break # Terkena SL, Win dibatalkan
                            
                if win:
                    success_count += 1

        win_rate = int((success_count / total_signals) * 100) if total_signals > 0 else 50
        
        # Filter ketat: Win Rate historis minimal 70% setelah ada SL
        if win_rate < 70:
            return None

        # 5. MENENTUKAN TARGET & SL HARI INI
        current_atr = float(atr.iloc[-1])
        
        # Target Profit (1.5x ATR)
        target_tp = round(last_close + (1.5 * current_atr))
        target_pct = round(((target_tp - last_close) / last_close) * 100, 1)
        
        # Stop Loss (1.0x ATR)
        stop_ls = round(last_close - (1.0 * current_atr))
        stop_pct = round(((last_close - stop_ls) / last_close) * 100, 1)

        return {
            "ticker": ticker.replace(".JK", ""),
            "price": last_close,
            "target": target_tp,
            "target_pct": target_pct,
            "stop": stop_ls,
            "stop_pct": stop_pct,
            "win_rate": win_rate
        }

    except Exception as e:
        return None

# ==========================================
# EKSEKUSI UTAMA
# ==========================================
def main():
    if not os.path.exists("tickers.txt"):
        print("File tickers.txt tidak ditemukan!")
        return

    with open("tickers.txt", "r") as f:
        tickers = [line.strip() + ".JK" for line in f if line.strip()]

    print(f"Memindai {len(tickers)} emiten dengan SMI Ergodic & Adaptive TP/SL...")
    
    results = []
    for t in tickers:
        res = analyze_stock(t)
        if res:
            results.append(res)

    # Ambil 2 terbaik
    results = sorted(results, key=lambda x: x['win_rate'], reverse=True)[:2]

    if not results:
        print("Tidak ada saham yang lolos kualifikasi hari ini.")
        return

    message = "🚨 *SMI ERGODIC SWING SIGNAL* 🚨\n"
    message += "⚖️ *Prinsip: Keselamatan Modal Nomor 1*\n\n"

    for r in results:
        message += f"📌 **{r['ticker']}** (Harga: Rp {r['price']})\n"
        message += f"• Screener: SMI Ergodic Crossover + Likuiditas Sehat\n"
        message += f"• Strategi: Sell On Strength / Trend Following\n"
        message += f"• Target Cuan (ATR): +{r['target_pct']}% (Rp {r['target']})\n" 
        message += f"• Stop Loss (ATR): -{r['stop_pct']}% (Rp {r['stop']})\n"
        message += f"• Max Hold Time: P5 (5 Hari)\n"
        message += f"• Historical Win Rate: {r['win_rate']}% (Setelah Uji TP vs SL)\n\n"

    message += "💡 *Catatan Pagi:* Selalu disiplin pasang antrean jual otomatis (Target) sekaligus batas rugi (Stop Loss). Jangan baper sama saham!"
    
    send_telegram_message(message)
    print("Laporan berhasil dikirim ke Telegram!")

if __name__ == "__main__":
    main()
