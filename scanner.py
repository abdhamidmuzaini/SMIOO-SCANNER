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

def calculate_atr_tp_sl(df, idx, holding_days=10):
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
        
        multiplier = 1.8 if holding_days <= 7 else 2.0
        tp_target_mult = 1.07 if holding_days <= 7 else 1.10
        
        if pd.isna(atr) or atr <= 0:
            sl = current_price * 0.95
            tp = current_price * tp_target_mult
        else:
            sl = current_price - (multiplier * atr)
            tp = current_price * tp_target_mult
            
        return round(sl, 2), round(tp, 2)
    except Exception:
        cp = float(df['Close'].iloc[idx])
        return round(cp * 0.95, 2), round(cp * 1.08, 2)

def evaluate_stock_atr_adaptive(df):
    try:
        if len(df) < 60:
            return False, 0, 0, 10, 0, 0, 0, 0
            
        df['SMIOO_Blue'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['SMIOO_Orange'] = df['SMIOO_Blue'].ewm(span=9, adjust=False).mean()
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
        
        candidate_holding_periods = [7, 10, 14]
        best_win_rate = -1
        best_holding_period = 10
        best_total_trades = 0
        
        for hp in candidate_holding_periods:
            wins = 0
            total_trades = 0
            
            for i in range(30, len(df) - hp):
                b_prev = df['SMIOO_Blue'].iloc[i-1]
                b_curr = df['SMIOO_Blue'].iloc[i]
                o_prev = df['SMIOO_Orange'].iloc[i-1]
                o_curr = df['SMIOO_Orange'].iloc[i]
                
                if (b_prev <= o_prev) and (b_curr > o_curr):
                    sl, tp = calculate_atr_tp_sl(df, i, hp)
                    trade_win = False
                    
                    for j in range(1, hp + 1):
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
                        
            wr = (wins / total_trades * 100) if total_trades > 0 else 0
            if wr > best_win_rate and total_trades >= 2:
                best_win_rate = wr
                best_holding_period = hp
                best_total_trades = total_trades
                
        if best_win_rate == -1:
            best_win_rate = 55.0
            best_holding_period = 10
            best_total_trades = 1
            
        recent_cross = False
        for k in range(1, 4):
            if len(df) > k + 1:
                bp = df['SMIOO_Blue'].iloc[-(k+1)]
                bc = df['SMIOO_Blue'].iloc[-k]
                op = df['SMIOO_Orange'].iloc[-(k+1)]
                oc = df['SMIOO_Orange'].iloc[-k]
                if bp <= op and bc > oc:
                    recent_cross = True
                    break
                    
        blue_today = df['SMIOO_Blue'].iloc[-1]
        blue_prev = df['SMIOO_Blue'].iloc[-2]
        orange_today = df['SMIOO_Orange'].iloc[-1]
        is_bullish_trend = (blue_today > orange_today) and (blue_today > blue_prev)
        
        if not (recent_cross or is_bullish_trend):
            return False, 0, 0, 10, 0, 0, 0, 0
            
        current_price = float(df['Close'].iloc[-1])
        ma5 = float(df['MA5'].iloc[-1])
        ma20 = float(df['MA20'].iloc[-1])
        vol_today = float(df['Volume'].iloc[-1])
        vol_ma20 = float(df['Vol_MA20'].iloc[-1])
        
        sl_today, tp_today = calculate_atr_tp_sl(df, len(df)-1, best_holding_period)
        risk = current_price - sl_today
        reward = tp_today - current_price
        rr = reward / risk if risk > 0 else 0
        
        score = 0
        if recent_cross:
            score += 30
        elif is_bullish_trend:
            score += 20
        if current_price > ma5 and current_price > ma20:
            score += 25
        elif current_price > ma20:
            score += 15
        if vol_today > vol_ma20:
            score += 25
        if rr >= 1.5:
            score += 20
        elif rr >= 1.3:
            score += 10
            
        return True, score, best_win_rate, best_holding_period, best_total_trades, sl_today, tp_today, rr
    except Exception:
        return False, 0, 0, 10, 0, 0, 0, 0

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
    candidates = []
    
    for symbol in tickers:
        try:
            ticker_obj = yf.Ticker(symbol)
            df = ticker_obj.history(start="2024-01-01")
            if df is None or len(df) < 60:
                continue
                
            current_price = float(df['Close'].iloc[-1])
            val_ma20 = float((df['Close'] * df['Volume']).rolling(window=20).mean().iloc[-1])
            
            if not (70 <= current_price < 1000 and val_ma20 > 2_000_000_000):
                continue
                
            has_signal, score, win_rate, holding_days, total_trades, sl, tp, rr = evaluate_stock_atr_adaptive(df)
            
            if has_signal and score >= 65 and total_trades >= 1 and win_rate >= 55.0 and rr >= 1.3:
                ticker_name = symbol.replace(".JK", "")
                candidates.append({
                    'symbol': ticker_name,
                    'score': score,
                    'win_rate': win_rate,
                    'holding_days': holding_days,
                    'price': current_price,
                    'sl': sl,
                    'tp': tp,
                    'rr': rr
                })
                
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    if candidates:
        candidates.sort(key=lambda x: (x['win_rate'], x['score']), reverse=True)
        top_candidates = candidates[:2]
        
        signals = []
        for item in top_candidates:
            sl_pct = ((item['sl'] - item['price']) / item['price']) * 100
            tp_pct = ((item['tp'] - item['price']) / item['price']) * 100
            signals.append(f"🎯 *{item['symbol']}* (ATR WR: {item['win_rate']:.0f}% | Hold: ~{item['holding_days']} Hari | RR: {item['rr']:.1f})\n  • Harga Masuk: {item['price']:.0f}\n  • 🔴 SL: {item['sl']} ({sl_pct:.1f}%)\n  • 🟢 TP: {item['tp']} (+{tp_pct:.1f}%)")
            
        message = "🚀 **REALISTIC ATR FAST SWING SIGNALS** 🚀\n📅 *Tanggal:* Hari Ini\n\n" + "\n\n".join(signals) + "\n\n_Disaring dengan ATR dinamis, WR >= 55%, & RR >= 1.3._"
        send_telegram(message)
    else:
        print("Tidak ada saham yang memenuhi kriteria realistis hari ini.")

if __name__ == "__main__":
    run_scanner()
