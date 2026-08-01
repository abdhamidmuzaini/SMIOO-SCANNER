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

def calculate_dynamic_tp_sl(df, idx):
    try:
        high = df['High'].iloc[:idx+1]
        low = df['Low'].iloc[:idx+1]
        close = df['Close'].iloc[:idx+1]
        current_price = float(close.iloc[-1])
        
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean().iloc[-1]
        
        if pd.isna(atr) or atr <= 0:
            sl = current_price * 0.93
            tp = current_price * 1.15
        else:
            sl = current_price - (2.0 * atr)
            tp = current_price * 1.15
            
        return round(sl, 2), round(tp, 2)
    except Exception:
        cp = float(df['Close'].iloc[idx])
        return round(cp * 0.93, 2), round(cp * 1.15, 2)

def evaluate_stock_adaptive(df):
    try:
        if len(df) < 60:
            return False, 0, 0, 0, 0, 0
            
        df['SMIOO_Blue'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['SMIOO_Orange'] = df['SMIOO_Blue'].ewm(span=9, adjust=False).mean()
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
        
        wins = 0
        total_trades = 0
        
        # Backtest dari awal data (Jan 2024) sampai 30 hari sebelum hari ini
        for i in range(30, len(df) - 30):
            b_prev = df['SMIOO_Blue'].iloc[i-1]
            b_curr = df['SMIOO_Blue'].iloc[i]
            o_prev = df['SMIOO_Orange'].iloc[i-1]
            o_curr = df['SMIOO_Orange'].iloc[i]
            
            if (b_prev <= o_prev) and (b_curr > o_curr) and (b_curr > b_prev):
                sl, tp = calculate_dynamic_tp_sl(df, i)
                trade_win = False
                
                # Simulasi swing hingga 30 hari ke depan
                for j in range(1, 31):
                    if i + j >= len(df):
                        break
                    high_future = float(df['High'].iloc[i+j])
                    low_future = float(df['Low'].iloc[i+j])
                    
                    if high_future >= tp:
                        trade_win = True
                        break
                    if low_future <= sl:
                        trade_win = False
                        break
                
                total_trades += 1
                if trade_win:
                    wins += 1
                    
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        current_price = float(df['Close'].iloc[-1])
        b_prev_today = df['SMIOO_Blue'].iloc[-2]
        b_curr_today = df['SMIOO_Blue'].iloc[-1]
        o_prev_today = df['SMIOO_Orange'].iloc[-2]
        o_curr_today = df['SMIOO_Orange'].iloc[-1]
        
        is_crossover = (b_prev_today <= o_prev_today) and (b_curr_today > o_curr_today)
        is_slope_up = b_curr_today > b_prev_today
        
        if not (is_crossover and is_slope_up):
            return False, 0, 0, 0, 0, 0
            
        ma5 = float(df['MA5'].iloc[-1])
        ma20 = float(df['MA20'].iloc[-1])
        vol_today = float(df['Volume'].iloc[-1])
        vol_ma20 = float(df['Vol_MA20'].iloc[-1])
        
        sl_today, tp_today = calculate_dynamic_tp_sl(df, len(df)-1)
        risk = current_price - sl_today
        reward = tp_today - current_price
        rr = reward / risk if risk > 0 else 0
        
        score = 0
        if is_crossover and is_slope_up:
            score += 30
        if current_price > ma5 and current_price > ma20:
            score += 25
        elif current_price > ma20:
            score += 15
        if vol_today > vol_ma20:
            score += 25
        if rr >= 2.0:
            score += 20
        elif rr >= 1.5:
            score += 10
            
        return True, score, win_rate, total_trades, sl_today, tp_today
    except Exception:
        return False, 0, 0, 0, 0, 0

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
            # Tarik data historis mutlak mulai dari 1 Januari 2024
            ticker_obj = yf.Ticker(symbol)
            df = ticker_obj.history(start="2024-01-01")
            if df is None or len(df) < 60:
                continue
                
            current_price = float(df['Close'].iloc[-1])
            val_ma20 = float((df['Close'] * df['Volume']).rolling(window=20).mean().iloc[-1])
            
            if not (70 <= current_price < 1000 and val_ma20 > 2_000_000_000):
                continue
                
            has_signal, score, win_rate, total_trades, sl, tp = evaluate_stock_adaptive(df)
            
            if has_signal and score >= 85 and total_trades >= 2 and win_rate >= 75.0:
                risk = current_price - sl
                reward = tp - current_price
                rr = reward / risk if risk > 0 else 0
                
                ticker_name = symbol.replace(".JK", "")
                signals.append(f"🎯 *{ticker_name}* (Score: {score} | Backtest 2024 WR: {win_rate:.0f}% | RR: {rr:.1f})\n  • Harga Masuk: {current_price:.0f}\n  • 🔴 SL: {sl}\n  • 🟢 TP: {tp}")
                
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    if signals:
        message = "🚀 **SWING SIGNALS (Backtest from Jan 2024 | Score >= 85 | WR >= 75%)** 🚀\n📅 *Tanggal:* Hari Ini\n\n" + "\n\n".join(signals) + "\n\n_Diuji berdasarkan rekam jejak historis sejak Januari 2024._"
        send_telegram(message)
    else:
        print("Tidak ada saham yang memenuhi kriteria backtest sejak Januari 2024 hari ini.")

if __name__ == "__main__":
    run_scanner()
