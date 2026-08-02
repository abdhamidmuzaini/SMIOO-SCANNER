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
# FUNGSI ANALISA SAHAM (FILTER DILONGGARIN)
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

        # ==========================================
        # FILTER LIKUIDITAS & HARGA (DIPERLONGGAR)
        # ==========================================
        value_txn = close * volume
        avg_value = value_txn.rolling(window=20).mean().iloc[-1]
        
        if avg_value < 2_000_000_000:  # Tetap 2M
            return None
        
        last_price = float(close.iloc[-1])
        if not (50 <= last_price <= 1500):  # 50-1500
            return None

        # ==========================================
        # FEATURE ENGINEERING
        # ==========================================
        
        # ATR
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
        df['ATR_MA20'] = df['ATR'].rolling(20).mean()
        df['Volatilitas'] = df['ATR'] / close

        # SMI Ergodic (20, 5, 5)
        price_diff = close.diff()
        ema1_diff = price_diff.ewm(span=20, adjust=False).mean()
        ema2_diff = ema1_diff.ewm(span=5, adjust=False).mean()
        ema1_abs = price_diff.abs().ewm(span=20, adjust=False).mean()
        ema2_abs = ema1_abs.ewm(span=5, adjust=False).mean()
        
        df['SMI'] = np.where(ema2_abs == 0, 0, (ema2_diff / ema2_abs) * 100)
        df['SMI_Signal'] = df['SMI'].ewm(span=5, adjust=False).mean()
        df['SMI_Hist'] = df['SMI'] - df['SMI_Signal']
        
        # Volume & Trend
        df['Vol_Surge'] = volume / volume.rolling(20).mean()
        df['EMA20'] = close.ewm(span=20, adjust=False).mean()
        df['EMA50'] = close.ewm(span=50, adjust=False).mean()
        df['Dist_EMA20'] = close / df['EMA20']
        df['Dist_EMA50'] = close / df['EMA50']

        # RSI
        gain = (price_diff.where(price_diff > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-price_diff.where(price_diff < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # ==========================================
        # GOLDEN CROSS + SWEET SPOT (DIPERLEBAR)
        # ==========================================
        gc_h0 = (df['SMI'].iloc[-1] > df['SMI_Signal'].iloc[-1]) and (df['SMI'].iloc[-2] <= df['SMI_Signal'].iloc[-2])
        gc_h1 = (df['SMI'].iloc[-2] > df['SMI_Signal'].iloc[-2]) and (df['SMI'].iloc[-3] <= df['SMI_Signal'].iloc[-3])
        gc_h2 = (df['SMI'].iloc[-3] > df['SMI_Signal'].iloc[-3]) and (df['SMI'].iloc[-4] <= df['SMI_Signal'].iloc[-4])
        is_golden_cross = gc_h0 or gc_h1 or gc_h2

        current_smi = df['SMI'].iloc[-1]
        is_sweet_spot = (-0.5 <= current_smi <= 0.8)  # DIPERLEBAR

        if not (is_golden_cross and is_sweet_spot):
            return None

        # ==========================================
        # BACKTEST VERIFICATION (WR MIN 35%)
        # ==========================================
        wins, total = 0, 0
        
        for i in range(100, len(df)-5):
            past_gc = (df['SMI'].iloc[i] > df['SMI_Signal'].iloc[i]) and \
                      (df['SMI'].iloc[i-1] <= df['SMI_Signal'].iloc[i-1])
            past_sweet = -0.5 <= df['SMI'].iloc[i] <= 0.8
            
            if past_gc and past_sweet:
                entry = df['Close'].iloc[i]
                exit_p = df['Close'].iloc[i+5]
                ret = (exit_p/entry - 1) * 100
                
                if ret > 0:
                    wins += 1
                total += 1
        
        if total < 3:
            return None
        
        wr = wins / total * 100
        
        if wr < 35:  # Diturunin ke 35%
            return None
        
        # Profit Factor
        total_win = 0
        total_loss = 0
        for i in range(100, len(df)-5):
            past_gc = (df['SMI'].iloc[i] > df['SMI_Signal'].iloc[i]) and \
                      (df['SMI'].iloc[i-1] <= df['SMI_Signal'].iloc[i-1])
            past_sweet = -0.5 <= df['SMI'].iloc[i] <= 0.8
            
            if past_gc and past_sweet:
                entry = df['Close'].iloc[i]
                exit_p = df['Close'].iloc[i+5]
                ret = (exit_p/entry - 1) * 100
                
                if ret > 0:
                    total_win += ret
                else:
                    total_loss += abs(ret)
        
        profit_factor = round(total_win / total_loss, 2) if total_loss > 0 else 999

        # ==========================================
        # AI/ML PREDICTION (CONF MIN 60%)
        # ==========================================
        df_clean = df.dropna().copy()
        if len(df_clean) < 100:
            return None
            
        features = ['SMI_Hist', 'Vol_Surge', 'RSI', 'Dist_EMA20', 'Dist_EMA50', 'Volatilitas']
        X = df_clean[features]
        
        df_clean['Target_TP'] = df_clean['Close'] + (1.5 * df_clean['ATR'])
        df_clean['Target_SL'] = df_clean['Close'] - (1.0 * df_clean['ATR'])
        
        labels = []
        for i in range(len(df_clean)):
            if i >= len(df_clean) - 5:
                labels.append(np.nan)
                continue
            win = 0
            for day in range(1, 6):
                if df_clean['High'].iloc[i+day] >= df_clean['Target_TP'].iloc[i]:
                    win = 1
                    break
                elif df_clean['Low'].iloc[i+day] <= df_clean['Target_SL'].iloc[i]:
                    break
            labels.append(win)
        
        y = pd.Series(labels).dropna()
        X = X.loc[y.index]
        
        if len(X) < 50:
            return None
        
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X, y)
        
        today_features = df.iloc[-1:][features]
        ai_confidence = model.predict_proba(today_features)[0][1] * 100
        
        if ai_confidence < 60:  # Diturunin ke 60%
            return None

        # ==========================================
        # DYNAMIC TP/SL
        # ==========================================
        current_atr = float(df['ATR'].iloc[-1])
        avg_atr_20 = float(df['ATR_MA20'].iloc[-1])
        vol_ratio = current_atr / avg_atr_20 if avg_atr_20 > 0 else 1
        
        if vol_ratio > 1.3:
            tp_mult, sl_mult = 2.0, 1.5
        else:
            tp_mult, sl_mult = 1.5, 1.0
        
        target_tp = round(last_price + (tp_mult * current_atr))
        target_pct = round(((target_tp - last_price) / last_price) * 100, 1)
        
        stop_ls = round(last_price - (sl_mult * current_atr))
        stop_pct = round(((last_price - stop_ls) / last_price) * 100, 1)

        return {
            "ticker": ticker.replace(".JK", ""),
            "price": last_price,
            "target": target_tp,
            "target_pct": target_pct,
            "stop": stop_ls,
            "stop_pct": stop_pct,
            "confidence": int(ai_confidence),
            "smi_val": round(current_smi, 2),
            "win_rate": round(wr, 1),
            "total_trades": total,
            "profit_factor": profit_factor,
            "vol_ratio": round(vol_ratio, 2),
            "avg_value_m": round(avg_value / 1e9, 1)
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
    
    print("🤖 Memindai dengan SMIIO + AI + Backtest...")
    
    results = []
    for i, t in enumerate(tickers, 1):
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
                print(f"  [{i}/{len(tickers)}] ✅ {res['ticker']}: WR={res['win_rate']}% | PF={res['profit_factor']} | Conf={res['confidence']}%")
        except Exception:
            continue

    results = sorted(results, key=lambda x: (x['profit_factor'], x['win_rate'], x['confidence']), reverse=True)[:3]

    if not results:
        message = "❌ Tidak ada saham yang lolos filter hari ini.\n\n"
        message += "💡 AI memutuskan: Better no trade than bad trade!"
        send_telegram_message(message)
        print("Tidak ada saham yang lolos.")
        return

    message = "🕌 *SMIIO GOLDEN CROSS - AI VERIFIED* 🕌\n"
    message += f"📅 {pd.Timestamp.now().strftime('%d %B %Y')}\n"
    message += "━━━━━━━━━━━━━━━━━━\n\n"

    for i, r in enumerate(results, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        
        message += f"{emoji} *{r['ticker']}* | Score: {r['confidence']}%\n"
        message += f"💰 Entry     : Rp {r['price']:,}\n"
        message += f"🎯 TP        : Rp {r['target']:,} (+{r['target_pct']}%)\n"
        message += f"🛑 SL        : Rp {r['stop']:,} (-{r['stop_pct']}%)\n"
        message += f"⚖️  R/R       : 1:{r['target_pct']/r['stop_pct']:.1f}\n\n"
        
        message += f"📊 *Backtest:*\n"
        message += f"• WR: {r['win_rate']}% | PF: {r['profit_factor']}\n"
        message += f"• SMI: {r['smi_val']} | Vol: {r['vol_ratio']}x\n\n"
        
        message += f"⏰ Max Hold: 5 Hari\n\n"

    message += "💡 *Disclaimer:* Tools bantu, tetap DYOR!"
    
    send_telegram_message(message)
    print("\n✅ Laporan berhasil dikirim ke Telegram!")

if __name__ == "__main__":
    main()
