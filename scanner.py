import os
import requests
import numpy as np
import pandas as pd
import yfinance as yf

# Konfigurasi Telegram
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
CHAT_ID = os.environ.get('CHAT_ID')


def send_telegram_message(message):
  if not TELEGRAM_TOKEN or not CHAT_ID:
    print('Token atau Chat ID Telegram belum diset!')
    return
  url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'

  if len(message) > 4000:
    chunks = [message[i : i + 4000] for i in range(0, len(message), 4000)]
    for chunk in chunks:
      requests.post(
          url, json={'chat_id': CHAT_ID, 'text': chunk, 'parse_mode': 'Markdown'}
      )
  else:
    payload = {'chat_id': CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    try:
      response = requests.post(url, json=payload)
      if response.status_code != 200:
        print(f'Gagal kirim Telegram: {response.text}')
    except Exception as e:
      print(f'Error koneksi Telegram: {e}')


def main():
  report = (
      '🚀 *Top Sinyal Saham (SMIOO GC + AI Historical Win Rate Rank):*\n\n'
  )

  try:
    with open('tickers.txt', 'r') as f:
      raw_tickers = [line.strip() for line in f if line.strip()]
  except FileNotFoundError:
    raw_tickers = ['BSML', 'ADRO', 'ANTM']

  valid_signals = []

  for raw in raw_tickers:
    ticker = raw.replace('$', '').strip().upper()
    if not ticker.endswith('.JK'):
      ticker += '.JK'

    try:
      df = yf.download(ticker, period='6mo', interval='1d', progress=False)
      if df.empty or len(df) < 50:
        continue

      if isinstance(df.columns, pd.MultiIndex):
        close = df['Close'].iloc[:, 0]
        open_p = df['Open'].iloc[:, 0]
        high = df['High'].iloc[:, 0]
        low = df['Low'].iloc[:, 0]
        volume = df['Volume'].iloc[:, 0]
      else:
        close = df['Close']
        open_p = df['Open']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

      last_close = float(close.iloc[-1])
      last_open = float(open_p.iloc[-1])

      # 1. Filter Rentang Harga: 70 - 1000
      if not (70 <= last_close <= 1000):
        continue

      # 2. Filter Transaksi 20 Hari > 2 Miliar
      traded_value = close * volume
      avg_value_20 = traded_value.rolling(window=20).mean().iloc[-1]
      if avg_value_20 < 2_000_000_000:
        continue

      # 3. Candle Bullish: C > O minimal 3%
      candle_pct = ((last_close - last_open) / last_open) * 100
      if candle_pct < 3.0:
        continue

      # 4. Harga > MA5
      ma5 = close.rolling(window=5).mean().iloc[-1]
      if last_close <= ma5:
        continue

      # 5. Indikator SMIOO Golden Cross (GC) hari ini
      high_n = high.rolling(14).max()
      low_n = low.rolling(14).min()
      midpoint = (high_n + low_n) / 2
      diff = close - midpoint
      ema2_diff = (
          diff.ewm(span=3, adjust=False).mean().ewm(span=3, adjust=False).mean()
      )
      ema2_hl = (
          (high_n - low_n)
          .ewm(span=3, adjust=False)
          .mean()
          .ewm(span=3, adjust=False)
          .mean()
      )

      smioo = (ema2_diff / (ema2_hl / 2)) * 100
      smioo_signal = smioo.ewm(span=4, adjust=False).mean()

      if len(smioo) < 2:
        continue
      is_gc = (smioo.iloc[-1] > smioo_signal.iloc[-1]) and (
          smioo.iloc[-2] <= smioo_signal.iloc[-2]
      )
      if not is_gc:
        continue

      # Hitung ATR & TP/SL hari ini
      tr1 = high - low
      tr2 = abs(high - close.shift())
      tr3 = abs(low - close.shift())
      tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
      atr = tr.rolling(14).mean().iloc[-1]

      tp_price = last_close + (1.5 * atr)
      sl_price = last_close - (1.0 * atr)
      tp_pct = ((tp_price - last_close) / last_close) * 100
      sl_pct = ((last_close - sl_price) / last_close) * 100

      # ---- AI BACKTEST / EVALUASI WIN RATE HISTORIS ----
      past_wins = 0
      past_losses = 0
      for i in range(50, len(df) - 14):
        h_sub = high.iloc[: i + 1]
        l_sub = low.iloc[: i + 1]
        c_sub = close.iloc[: i + 1]
        o_sub = open_p.iloc[: i + 1]
        v_sub = volume.iloc[: i + 1]

        c_val = float(c_sub.iloc[-1])
        o_val = float(o_sub.iloc[-1])

        if not (70 <= c_val <= 1000):
          continue
        if (
            (c_sub * v_sub).rolling(window=20).mean().iloc[-1]
            < 2_000_000_000
        ):
          continue
        if ((c_val - o_val) / o_val) * 100 < 3.0:
          continue
        if c_val <= c_sub.rolling(window=5).mean().iloc[-1]:
          continue

        hn = h_sub.rolling(14).max()
        ln = l_sub.rolling(14).min()
        mid = (hn + ln) / 2
        d = c_sub - mid
        e2d = (
            d.ewm(span=3, adjust=False).mean().ewm(span=3, adjust=False).mean()
        )
        e2hl = (
            (hn - ln)
            .ewm(span=3, adjust=False)
            .mean()
            .ewm(span=3, adjust=False)
            .mean()
        )
        sm = (e2d / (e2hl / 2)) * 100
        sms = sm.ewm(span=4, adjust=False).mean()

        if len(sm) < 2:
          continue
        if (sm.iloc[-1] > sms.iloc[-1]) and (sm.iloc[-2] <= sms.iloc[-2]):
          tr_sub = pd.concat(
              [
                  h_sub - l_sub,
                  abs(h_sub - c_sub.shift()),
                  abs(l_sub - c_sub.shift()),
              ],
              axis=1,
          ).max(axis=1)
          atr_v = tr_sub.rolling(14).mean().iloc[-1]
          tp_v = c_val + (1.5 * atr_v)
          sl_v = c_val - (1.0 * atr_v)

          hit = False
          for f in range(1, 15):
            if i + f >= len(df):
              break
            fh = float(high.iloc[i + f])
            fl = float(low.iloc[i + f])
            if fh >= tp_v:
              past_wins += 1
              hit = True
              break
            elif fl <= sl_v:
              past_losses += 1
              hit = True
              break
          if not hit:
            if float(close.iloc[min(i + 14, len(df) - 1)]) > c_val:
              past_wins += 1
            else:
              past_losses += 1

      total_trades = past_wins + past_losses
      win_rate = (past_wins / total_trades * 100) if total_trades > 0 else 50.0

      valid_signals.append({
          'ticker': ticker.replace('.JK', ''),
          'price': int(last_close),
          'candle_pct': candle_pct,
          'tp_price': int(tp_price),
          'tp_pct': tp_pct,
          'sl_price': int(sl_price),
          'sl_pct': sl_pct,
          'win_rate': win_rate,
          'trades': total_trades,
      })

    except Exception as e:
      continue

  # Urutkan berdasarkan Win Rate historis tertinggi
  valid_signals.sort(key=lambda x: x['win_rate'], reverse=True)

  # Ambil Top 2 terbaik saja
  top_signals = valid_signals[:2]

  if not top_signals:
    report += (
        'Belum ada emiten yang lolos kriteria AI Win Rate hari'
        ' ini. *Cash is King!*'
    )
  else:
    for sig in top_signals:
      report += f"📌 *{sig['ticker']}* (Harga: Rp {sig['price']})\n"
      report += f"   • AI Win Rate Historis: *{sig['win_rate']:.1f}%* ({sig['trades']} sampel uji masa lalu)\n"
      report += f"   • Kenaikan Harian: +{sig['candle_pct']:.1f}%\n"
      report += f"   • TP: Rp {sig['tp_price']} (+{sig['tp_pct']:.1f}%)\n"
      report += f"   • SL: Rp {sig['sl_price']} (-{sig['sl_pct']:.1f}%)\n\n"

  send_telegram_message(report)


if __name__ == '__main__':
  main()
