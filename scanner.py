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
# RUMUS INDIKATOR: SMI ERGODIC OSCILLATOR & AI HISTORIS
# ==========================================
def analyze_stock(ticker):
    try:
        # Unduh data historis
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df.empty or len(df) < 50:
            return None
        
        # Rapihkan kolom jika MultiIndex (dari yfinance terbaru)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = df['Close']
        high = df['High']
        volume = df['Volume']

        # 1. FILTER LIKUIDITAS (Anti-Nyangkut: Min Rp 1 Miliar/hari)
        value_txn = close * volume
        avg_value_20 = value_txn.rolling(window=20).mean().iloc[-1]
        if avg_value_20 < 1_000_000_000:
            return None

        # 2. FILTER HARGA DASAR (Di bawah Rp 2000)
        last_close = float(close.iloc[-1])
        if last_close > 2000:
            return None

        # 3. RUMUS UTAMA: SMI ERGODIC OSCILLATOR (SMIEO)
        # Parameter standar Blau Ergodic: Long EMA = 20, Short EMA = 5, Signal EMA = 5
        long_len = 20
        short_len = 5
        sig_len = 5

        # Hitung selisih harga (Price Change) & nilai absolutnya
        price_diff = close.diff()
        abs_price_diff = price_diff.abs()

        # Double Smoothing (Penghalusan Ganda)
        ema1_diff = price_diff.ewm(span=long_len, adjust=False).mean()
        ema2_diff = ema1_diff.ewm(span=short_len, adjust=False).mean()

        ema1_abs = abs_price_diff.ewm(span=long_len, adjust=False).mean()
        ema2_abs = ema1_abs.ewm(span=short_len, adjust=False).mean()

        # Hitung SMI Ergodic Line dan Signal Line
        # Menghindari pembagian dengan 0
        smi_line = np.where(ema2_abs == 0, 0, (ema2_diff / ema2_abs) * 100)
        smi_line = pd.Series(smi_line, index=close.index)
        signal_line = smi_line.ewm(span=sig_len, adjust=False).mean()

        # Kondisi: Hari ini SMI Ergodic crossing ke atas signal line / atau posisi momentum positif
        is_smi_cross = (smi_line.iloc[-1] > signal_line.iloc[-1]) and (smi_line.iloc[-2] <= signal_line.iloc[-2])
        is_smi_active = (smi_line.iloc[-1] > 0) and (smi_line.iloc[-1] > signal_line.iloc[-1])

        if not (is_smi_cross or is_smi_active):
            return None

        # 4. SIMULASI AI MACHINE LEARNING (Backtest Historis Pola SMI Ergodic)
        # Dioptimalkan: Menggunakan series yang sudah dihitung agar 100x lebih cepat
        success_count = 0
        total_signals = 0
        
        # Mulai dari index 50 agar indikator EMA sudah stabil (tidak ada efek data awal)
        for i in range(50, len(df) - 5):
            # Cek apakah di masa lalu garis Ergodic berada di atas Signal
            if smi_line.iloc[i] > signal_line.iloc[i]:
                total_signals += 1
                future_slice = high.iloc[i+1:i+6] # Cek 5 hari ke depan
                target_price = float(close.iloc[i]) * 1.035 # Target +3.5%
                
                if (future_slice >= target_price).any():
                    success_count += 1

        win_rate = int((success_count / total_signals) * 100) if total_signals > 0 else 50
        
        # Filter ketat: Win Rate historis minimal 75%
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
        # print(f"Error pada {ticker}: {e}") # Matikan print error agar terminal lebih bersih
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

    print(f"Memindai {len(tickers)} emiten dengan SMI Ergodic Oscillator & AI...")
    
    results = []
    for t in tickers:
        res = analyze_stock(t)
        if res:
            results.append(res)

    # Urutkan berdasarkan Win Rate tertinggi, ambil maksimal 2 saham terbaik
    results = sorted(results, key=lambda x: x['win_rate'], reverse=True)[:2]

    if not results:
        print("Tidak ada saham yang lolos kualifikasi SMIEO & AI hari ini.")
        return

    message = "🚨 *SMI ERGODIC SWING SIGNAL + AI* 🚨\n"
    message += "⚖️ *Prinsip: Keselamatan Modal Nomor 1*\n\n"

    for r in results:
        message += f"📌 **{r['ticker']}** (Harga: Rp {r['price']})\n"
        message += f"• Screener: SMI Ergodic Crossover + Likuiditas Sehat\n"
        message += f"• Strategi: Sell On Strength (SOS) / Trend Following\n"
        message += f"• Target Cuan (Adaptive): +3.5% (Rp {r['target']}) - Pasang Auto Order Jual!\n"
        message += f"• Max Hold Time: P5 (5 Hari)\n"
        message += f"• AI Machine Learning WR: {r['win_rate']}% (Validasi Sejarah Kuat)\n"
        message += f"• Stop Loss: Tidak Ada (Disiplin Jual di hari ke-5 / Patah Tren)\n\n"

    message += "💡 *Catatan Pagi:* Cek pembukaan market besok. Jika hijau lanjut, pegang. Jika merah/layu, amankan posisi!"
    
    send_telegram_message(message)
    print("Laporan SMI Ergodic berhasil dikirim ke Telegram!")

if __name__ == "__main__":
    main()
