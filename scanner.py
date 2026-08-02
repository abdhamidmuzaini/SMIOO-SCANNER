import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

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
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Error koneksi Telegram: {e}")

# ==========================================
# FUNGSI INDIKATOR TEKNIKAL & AI ML
# ==========================================
def analyze_stock_with_ai(ticker, df_saham):
    try:
        df = df_saham.dropna().copy()
        if df.empty or len(df) < 100:
            return None

        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        # FILTER AWAL (Likuiditas > 1M & Harga < 2000)
        value_txn = close * volume
        if value_txn.rolling(window=20).mean().iloc[-1] < 1_000_000_000: return None
        if float(close.iloc[-1]) > 2000 or float(close.iloc[-1]) < 50: return None

        # 1. FEATURE ENGINEERING (Bahan Belajar AI)
        
        # A. ATR (Volatilitas)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
        df['Volatilitas'] = df['ATR'] / close

        # B. SMIIO (SMI Ergodic Oscillator 5, 20, 5)
        price_diff = close.diff()
        ema1_diff = price_diff.ewm(span=20, adjust=False).mean()
        ema2_diff = ema1_diff.ewm(span=5, adjust=False).mean()
        ema1_abs = price_diff.abs().ewm(span=20, adjust=False).mean()
        ema2_abs = ema1_abs.ewm(span=5, adjust=False).mean()
        
        df['SMI'] = np.where(ema2_abs == 0, 0, (ema2_diff / ema2_abs) * 100)
        df['SMI_Signal'] = df['SMI'].ewm(span=5, adjust=False).mean()
        df['SMI_Hist'] = df['SMI'] - df['SMI_Signal']
        
        # C. Volume Surge
        df['Vol_Surge'] = volume / volume.rolling(20).mean()

        # D. Trend EMA 20 & 50
        df['EMA20'] = close.ewm(span=20, adjust=False).mean()
        df['EMA50'] = close.ewm(span=50, adjust=False).mean()
        df['Dist_EMA20'] = close / df['EMA20']
        df['Dist_EMA50'] = close / df['EMA50']

        # E. RSI 14
        gain = (price_diff.where(price_diff > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-price_diff.where(price_diff < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # 2. LABELING TARGET UNTUK AI
        df['Target_TP'] = close + (1.5 * df['ATR'])
        df['Target_SL'] = close - (1.0 * df['ATR'])
        
        labels = []
        for i in range(len(df)):
            if i >= len(df) - 5: 
                labels.append(np.nan)
                continue
            
            win = 0
            for day in range(1, 6):
                if high.iloc[i+day] >= df['Target_TP'].iloc[i]:
                    win = 1
                    break
                elif low.iloc[i+day] <= df['Target_SL'].iloc[i]:
                    break
            labels.append(win)
        
        df['Label'] = labels

        # ==========================================
        # FILTER KONDISI SMIIO (DIPERLUAS: 3 HARI & ZONA -0.2 s.d +0.4)
        # ==========================================
        # Cek apakah terjadi Golden Cross dalam 3 hari terakhir
        gc_h0 = (df['SMI'].iloc[-1] > df['SMI_Signal'].iloc[-1]) and (df['SMI'].iloc[-2] <= df['SMI_Signal'].iloc[-2])
        gc_h1 = (df['SMI'].iloc[-2] > df['SMI_Signal'].iloc[-2]) and (df['SMI'].iloc[-3] <= df['SMI_Signal'].iloc[-3])
        gc_h2 = (df['SMI'].iloc[-3] > df['SMI_Signal'].iloc[-3]) and (df['SMI'].iloc[-4] <= df['SMI_Signal'].iloc[-4])
        is_new_golden_cross = gc_h0 or gc_h1 or gc_h2

        # Zona SMIIO yang diperluas ke -0.2 s.d +0.4
        current_smi = df['SMI'].iloc[-1]
        is_in_sweet_spot = (-0.2 <= current_smi <= 0.4)

        if not (is_new_golden_cross and is_in_sweet_spot):
            return None

        # 3. TRAINING MACHINE LEARNING (Random Forest)
        df_clean = df.dropna().copy()
        if len(df_clean) < 100: return None
            
        features = ['SMI_Hist', 'Vol_Surge', 'RSI', 'Dist_EMA20', 'Dist_EMA50', 'Volatilitas']
        X = df_clean[features]
        y = df_clean['Label']

        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X, y)

        # 4. PREDIKSI (Confidence diturunkan ke 65%)
        today_features = df.iloc[-1:][features]
        ai_confidence = model.predict_proba(today_features)[0][1] * 100 
        
        if ai_confidence < 65:
            return None

        last_close = float(close.iloc[-1])
        current_atr = float(df['ATR'].iloc[-1])
        
        target_tp = round(last_close + (1.5 * current_atr))
        target_pct = round(((target_tp - last_close) / last_close) * 100, 1)
        
        stop_ls = round(last_close - (1.0 * current_atr))
        stop_pct = round(((last_close - stop_ls) / last_close) * 100, 1)

        return {
            "ticker": ticker.replace(".JK", ""),
            "price": last_close,
            "target": target_tp,
            "target_pct": target_pct,
            "stop": stop_ls,
            "stop_pct": stop_pct,
            "confidence": int(ai_confidence),
            "smi_val": round(current_smi, 2)
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

    print(f"📥 Mengunduh massal {len(tickers)} saham...")
    data_all = yf.download(tickers, period="18mo", interval="1d", progress=False)
    
    print("🤖 Memindai dengan Parameter Fleksibel SMIIO + AI...")
    
    results = []
    for t in tickers:
        try:
            df_saham = pd.DataFrame({
                'Close': data_all['Close'][t],
                'High': data_all['High'][t],
                'Low': data_all['Low'][t],
                'Volume': data_all['Volume'][t]
            })
            
            res = analyze_stock_with_ai(t, df_saham)
            if res:
                results.append(res)
        except Exception:
            continue

    results = sorted(results, key=lambda x: x['confidence'], reverse=True)[:3]

    if not results:
        print("Tidak ada saham yang lolos kualifikasi fleksibel hari ini.")
        return

    message = "🤖 *AI STOCK PREDICTOR (FLEKSIBEL MODE)* 🤖\n"
    message += "⚖️ *Filter: GC 3 Hari Terakhir | Zona -0.2 s.d +0.4 | Min Conf 65%*\n\n"

    for r in results:
        message += f"📌 **{r['ticker']}** (Harga: Rp {r['price']})\n"
        message += f"• SMIIO Value: {r['smi_val']} (Zona Aman Fleksibel)\n"
        message += f"• Prediksi AI (Confidence): 🔥 **{r['confidence']}%** 🔥\n"
        message += f"• Target Cuan (ATR): +{r['target_pct']}% (Rp {r['target']})\n" 
        message += f"• Stop Loss (ATR): -{r['stop_pct']}% (Rp {r['stop']})\n"
        message += f"• Max Hold Time: P5 (5 Hari)\n\n"

    message += "💡 *Catatan:* Filter diperlonggar agar bot lebih responsif menangkap momentum."
    
    send_telegram_message(message)
    print("Laporan berhasil dikirim!")

if __name__ == "__main__":
    main()
