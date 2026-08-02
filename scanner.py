import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("CHAT_ID")

def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Token/Chat ID Telegram belum diset!")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Error Telegram: {e}")

def analyze_stock(ticker, df_saham):
    try:
        df = df_saham.dropna().copy()
        if df.empty or len(df) < 50:
            return None

        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        # Filter
        last_price = float(close.iloc[-1])
        if not (60 <= last_price <= 1000):
            return None
        
        avg_value = (close * volume).rolling(20).mean().iloc[-1]
        if avg_value < 1_000_000_000:
            return None

        # SMI
        price_diff = close.diff()
        e1d = price_diff.ewm(span=20, adjust=False).mean()
        e2d = e1d.ewm(span=5, adjust=False).mean()
        e1a = price_diff.abs().ewm(span=20, adjust=False).mean()
        e2a = e1a.ewm(span=5, adjust=False).mean()
        
        df['SMI'] = np.where(e2a == 0, 0, (e2d / e2a) * 100)
        df['SMI_Signal'] = df['SMI'].ewm(span=5, adjust=False).mean()

        # Golden Cross
        gc0 = (df['SMI'].iloc[-1] > df['SMI_Signal'].iloc[-1]) and (df['SMI'].iloc[-2] <= df['SMI_Signal'].iloc[-2])
        gc1 = (df['SMI'].iloc[-2] > df['SMI_Signal'].iloc[-2]) and (df['SMI'].iloc[-3] <= df['SMI_Signal'].iloc[-3])
        gc2 = (df['SMI'].iloc[-3] > df['SMI_Signal'].iloc[-3]) and (df['SMI'].iloc[-4] <= df['SMI_Signal'].iloc[-4])
        
        if not (gc0 or gc1 or gc2):
            return None

        # ATR
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        df['ATR'] = pd.DataFrame({'t1':tr1,'t2':tr2,'t3':tr3}).max(axis=1).rolling(14).mean()
        
        atr = float(df['ATR'].iloc[-1])
        
        # TP/SL
        tp = round(last_price + (1.5 * atr))
        sl = round(last_price - (1.0 * atr))

        # Win Rate
        wins = total = 0
        for i in range(50, len(df)-5):
            if (df['SMI'].iloc[i] > df['SMI_Signal'].iloc[i]) and \
               (df['SMI'].iloc[i-1] <= df['SMI_Signal'].iloc[i-1]):
                ret = (df['Close'].iloc[i+5]/df['Close'].iloc[i] - 1)*100
                if ret > 0: wins += 1
                total += 1
        
        wr = wins/total*100 if total > 0 else 50
        
        gc_day = "Hari Ini" if gc0 else "Kemarin" if gc1 else "2 Hari Lalu"

        return {
            "ticker": ticker.replace(".JK", ""),
            "price": last_price,
            "target": tp,
            "target_pct": round((tp/last_price-1)*100,1),
            "stop": sl,
            "stop_pct": round((last_price-sl)/last_price*100,1),
            "smi": round(df['SMI'].iloc[-1],2),
            "win_rate": round(wr,1),
            "total_trades": total,
            "gc_day": gc_day,
            "volume_m": round(avg_value/1e9,1)
        }

    except:
        return None

def main():
    if not os.path.exists("tickers.txt"):
        return

    with open("tickers.txt", "r") as f:
        tickers = [line.strip() + ".JK" for line in f if line.strip()]

    print(f"📥 Download {len(tickers)} saham...")
    data_all = yf.download(tickers, period="6mo", interval="1d", progress=False)
    
    print("🔍 Scan Golden Cross...")
    
    results = []
    for t in tickers:
        try:
            df = pd.DataFrame({
                'Close': data_all['Close'][t],
                'High': data_all['High'][t],
                'Low': data_all['Low'][t],
                'Volume': data_all['Volume'][t]
            })
            res = analyze_stock(t, df)
            if res:
                results.append(res)
                print(f"  ✅ {res['ticker']}: WR {res['win_rate']}% | GC {res['gc_day']}")
        except:
            continue

    results = sorted(results, key=lambda x: x['win_rate'], reverse=True)[:2]

    if not results:
        msg = "❌ *Tidak ada Golden Cross hari ini*\n\n💡 Hold cash dulu."
    else:
        msg = "🟢 *GOLDEN CROSS - TOP PICKS*\n"
        msg += f"📅 {pd.Timestamp.now().strftime('%d %B %Y')}\n"
        msg += "━━━━━━━━━━━━━━━━━━\n\n"
        
        for i, r in enumerate(results, 1):
            emoji = "🥇" if i == 1 else "🥈"
            msg += f"{emoji} *{r['ticker']}*\n"
            msg += f"💰 Rp {r['price']:,} | WR: {r['win_rate']}%\n"
            msg += f"🎯 +{r['target_pct']}% | 🛑 -{r['stop_pct']}%\n"
            msg += f"📊 SMI: {r['smi']} | GC: {r['gc_day']}\n\n"
        
        msg += "⏰ Hold: 5 Hari\n💡 Tetap DYOR!"

    send_telegram_message(msg)
    print("\n✅ Done!")

if __name__ == "__main__":
    main()
