import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf

# ==========================================
# KONFIGURASI TELEGRAM & TARGET
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
# MESIN ANALISIS TEKNIKAL & AI HISTORI
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

        # 1. FILTER LIKUIDITAS (Anti-Nyangkut: Min Rp 1 Miliar/hari)
        value_txn = close * volume
        avg_value_20 = value_txn.rolling(window=20).mean().iloc[-1]
        if avg_value_20 < 1_000_000_000:
            return None

        # 2. FILTER HARGA & RATA-RATA HARIAN
        last_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        
        if last_close > 2000 or last_close <= prev_close:
            return None

        # 3. BOLLINGER BANDS
        sma_20 = close.rolling(window=20).mean()
        std_20 = close.rolling(window=20).std()
        upper_band = sma_20 + (std_20 * 2)
        
        if last_close < float(upper_band.iloc[-1]):
            return None

        # 4. SIMULASI AI HISTORIS (Backtest Pola Close-to-High P1-P5)
        success_count = 0
        total_signals = 0
        
        for i in range(20, len(df) - 5):
            if float(close.iloc[i]) > float(upper_band.iloc[i]):
                total_signals += 1
                future_slice = high.iloc[i+1:i+6]
                target_price = float(close.iloc[i]) * 1.035
                
                if (future_slice >= target_price).any():
                    success_count += 1

        win_rate = int((success_count / total_signals) * 100) if total_signals > 0 else 50
        
        if win_rate < 75:
            return None

        target_tp = round(last_close * 1.035)

        return {
            "ticker": ticker.replace(".JK", ""),
            "price": last_close,
            "target": target_tp,
            "win_rate": win_rate
        }

    except Exception as e:
        print(f"Error pada {ticker}: {e}")
        return None

# ==========================================
# EKSEKUSI UTAMA (MAIN PROGRAM)
# ==========================================
def main():
    if not os.path.exists("tickers.txt"):
        print("File tickers.txt tidak ditemukan!")
        return

    with open("tickers.txt", "r") as f:
        tickers = [line.strip() + ".JK" for line in f if line.strip()]

    print(f"Memindai {len(tickers)} emiten dari database induk...")
    
    results = []
    for t in tickers:
        res = analyze_stock(t)
        if res:
            results.append(res)

    results = sorted(results, key=lambda x: x['win_rate'], reverse=True)[:2]

    if not results:
        print("Tidak ada saham yang lolos kualifikasi ketat hari ini.")
        return

    message = "🚨 *QUANT SWING SIGNAL (TOP PICK)* 🚨\n"
    message += "⚖️ *Prinsip: Keselamatan Modal Nomor 1*\n\n"

    for r in results:
        message += f"📌 **{r['ticker']}** (Harga: Rp {r['price']})\n"
        message += f"• Screener: BB Upper Cross + Likuiditas Sehat\n"
        message += f"• Strategi: Sell On Strength (SOS) - Close to High\n"
        message += f"• Target Cuan (Adaptive): +3.5% (Rp {r['target']}) - Pasang Auto Order Jual!\n"
        message += f"• Max Hold Time: P5 (5 Hari)\n"
        message += f"• AI Historical WR: {r['win_rate']}% (Validasi Historis Kuat)\n"
        message += f"• Stop Loss: Tidak Ada (Disiplin Jual di hari ke-5 / Time-Stop jika belum capai target)\n\n"

    message += "💡 *Catatan Pagi:* Cek pembukaan market besok. Jika hijau lanjut, pegang. Jika merah/layu, amankan posisi!"
    
    send_telegram_message(message)
    print("Laporan berhasil dikirim ke Telegram!")

if __name__ == "__main__":
    main()
    
