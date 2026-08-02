import os
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

def analyze_stock_debug(ticker, df_saham):
    """Version debug - return alasan ditolak"""
    try:
        df = df_saham.dropna().copy()
        if df.empty or len(df) < 100:
            return None, "Data < 100"

        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        # Filter 1: Likuiditas
        value_txn = close * volume
        avg_value = value_txn.rolling(window=20).mean().iloc[-1]
        if avg_value < 2_000_000_000:
            return None, f"Volume kecil: Rp {avg_value/1e9:.1f}M"

        # Filter 2: Harga
        last_price = float(close.iloc[-1])
        if not (50 <= last_price <= 1000):
            return None, f"Harga diluar: Rp {last_price}"

        # SMI Calculation
        price_diff = close.diff()
        ema1_diff = price_diff.ewm(span=20, adjust=False).mean()
        ema2_diff = ema1_diff.ewm(span=5, adjust=False).mean()
        ema1_abs = price_diff.abs().ewm(span=20, adjust=False).mean()
        ema2_abs = ema1_abs.ewm(span=5, adjust=False).mean()
        
        df['SMI'] = np.where(ema2_abs == 0, 0, (ema2_diff / ema2_abs) * 100)
        df['SMI_Signal'] = df['SMI'].ewm(span=5, adjust=False).mean()

        # Filter 3: Golden Cross
        gc_h0 = (df['SMI'].iloc[-1] > df['SMI_Signal'].iloc[-1]) and (df['SMI'].iloc[-2] <= df['SMI_Signal'].iloc[-2])
        gc_h1 = (df['SMI'].iloc[-2] > df['SMI_Signal'].iloc[-2]) and (df['SMI'].iloc[-3] <= df['SMI_Signal'].iloc[-3])
        gc_h2 = (df['SMI'].iloc[-3] > df['SMI_Signal'].iloc[-3]) and (df['SMI'].iloc[-4] <= df['SMI_Signal'].iloc[-4])
        is_gc = gc_h0 or gc_h1 or gc_h2

        # Filter 4: Sweet Spot
        current_smi = df['SMI'].iloc[-1]
        is_sweet = -0.2 <= current_smi <= 0.4

        if not is_gc:
            return None, f"No GC (SMI:{current_smi:.2f})"
        
        if not is_sweet:
            return None, f"Not Sweet (SMI:{current_smi:.2f})"

        # Filter 5: Backtest
        wins, total = 0, 0
        for i in range(100, len(df)-5):
            past_gc = (df['SMI'].iloc[i] > df['SMI_Signal'].iloc[i]) and \
                      (df['SMI'].iloc[i-1] <= df['SMI_Signal'].iloc[i-1])
            past_sweet = -0.2 <= df['SMI'].iloc[i] <= 0.4
            
            if past_gc and past_sweet:
                entry = df['Close'].iloc[i]
                exit_p = df['Close'].iloc[i+5]
                ret = (exit_p/entry - 1) * 100
                if ret > 0: wins += 1
                total += 1

        if total < 3:
            return None, f"Trades < 3 ({total})"

        wr = wins / total * 100
        if wr < 40:
            return None, f"WR < 40% ({wr:.0f}%, {total}t)"

        # Filter 6: AI
        df['ATR'] = pd.DataFrame({
            'tr1': high - low,
            'tr2': (high - close.shift(1)).abs(),
            'tr3': (low - close.shift(1)).abs()
        }).max(axis=1).rolling(14).mean()
        
        df['Vol_Surge'] = volume / volume.rolling(20).mean()
        df['EMA20'] = close.ewm(span=20, adjust=False).mean()
        df['EMA50'] = close.ewm(span=50, adjust=False).mean()
        df['Dist_EMA20'] = close / df['EMA20']
        df['Dist_EMA50'] = close / df['EMA50']
        df['SMI_Hist'] = df['SMI'] - df['SMI_Signal']
        df['Volatilitas'] = df['ATR'] / close
        
        gain = (price_diff.where(price_diff > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-price_diff.where(price_diff < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        df_clean = df.dropna()
        if len(df_clean) < 100:
            return None, "Data < 100"

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
            return None, f"ML < 50 data"

        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X, y)

        today_features = df.iloc[-1:][features]
        ai_conf = model.predict_proba(today_features)[0][1] * 100

        if ai_conf < 65:
            return None, f"AI < 65% ({ai_conf:.0f}%)"

        return {"ticker": ticker, "wr": wr, "conf": ai_conf, "smi": current_smi}, "✅ LOLOS"

    except Exception as e:
        return None, f"Error: {str(e)[:50]}"


def debug_main():
    if not os.path.exists("tickers.txt"):
        print("File tickers.txt tidak ditemukan!")
        return

    with open("tickers.txt", "r") as f:
        tickers = [line.strip() + ".JK" for line in f if line.strip()]

    print(f"🔍 Debug {len(tickers)} saham...")
    print("="*60)
    
    # Test 50 saham dulu biar cepet
    test_tickers = tickers[:50]
    
    print(f"📥 Download {len(test_tickers)} saham (test)...")
    data_all = yf.download(test_tickers, period="18mo", interval="1d", progress=False)
    
    stats = {}
    lolos_list = []
    
    for t in test_tickers:
        try:
            df_saham = pd.DataFrame({
                'Close': data_all['Close'][t],
                'High': data_all['High'][t],
                'Low': data_all['Low'][t],
                'Volume': data_all['Volume'][t]
            })
            
            result, reason = analyze_stock_debug(t, df_saham)
            
            # Count reasons
            if reason not in stats:
                stats[reason] = 0
            stats[reason] += 1
            
            if result:
                print(f"  ✅ {t.replace('.JK','')}: WR={result['wr']:.0f}% | Conf={result['conf']:.0f}% | SMI={result['smi']:.2f}")
                lolos_list.append(result)
            else:
                print(f"  ❌ {t.replace('.JK','')}: {reason}")
                
        except Exception as e:
            reason = f"Error: {str(e)[:50]}"
            if reason not in stats:
                stats[reason] = 0
            stats[reason] += 1
    
    # Summary
    print("\n" + "="*60)
    print("📊 DEBUG SUMMARY")
    print("="*60)
    
    # Sort by count
    sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)
    for reason, count in sorted_stats:
        bar = "█" * (count // 2)
        print(f"  {reason:<40} {count:>3} {bar}")
    
    print(f"\n📈 Total LOLOS: {len(lolos_list)}/{len(test_tickers)}")
    
    if lolos_list:
        print("\n🏆 Yang Lolos:")
        for r in lolos_list:
            print(f"  ✅ {r['ticker'].replace('.JK','')}: WR={r['wr']:.0f}% | Conf={r['conf']:.0f}%")

if __name__ == "__main__":
    debug_main()
