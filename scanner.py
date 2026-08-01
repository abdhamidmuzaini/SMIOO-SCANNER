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
  report = '🚀 *Sinyal Saham (SMIOO GC + Filter Ketat):*\n\n'

  try:
    with open('tickers.txt', 'r') as f:
      raw_tickers = [line.strip() for line in f if line.strip()]
  except FileNotFoundError:
    raw_tickers = ['BSML', 'ADRO', 'ANTM']

  found_signal_count = 0

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

      # 1. FILTER RENTANG HARGA: 70 - 1000
      if not (70 <= last_close <= 1000):
        continue

      # 2. FILTER TRANSAKSI 20 HARI > 2 MILIAR
      traded_value = close * volume
      avg_value_20 = traded_value.rolling(window=20).mean().iloc[-1]
      if avg_value_20 < 2_000_000_000:
        continue

      # 3. CANDLE BULLISH: C > O minimal 3%
      candle_pct = ((last_close - last_open) / last_open) * 100
      if candle_pct < 3.0:
        continue

      # 4. HARGA > MA5
      ma5 = close.rolling(window=5).mean().iloc[-1]
      if last_close <= ma5:
        continue

      # 5. INDIKATOR SMIOO GOLDEN CROSS (GC)
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

      # Cek Golden Cross
      is_gc = (smioo.iloc[-1] > smioo_signal.iloc[-1]) and (
          smioo.iloc[-2] <= smioo_signal.iloc[-2]
      )
      if not is_gc:
        continue

      # Hitung ATR untuk TP & SL
      tr1 = high - low
      tr2 = abs(high - close.shift())
      tr3 = abs(low - close.shift())
      tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
      atr = tr.rolling(14).mean().iloc[-1]

      tp_price = last_close + (1.5 * atr)
      sl_price = last_close - (1.0 * atr)

      tp_pct = ((tp_price - last_close) / last_close) * 100
      sl_pct = ((last_close - sl_price) / last_close) * 100

      found_signal_count += 1
      clean_name = ticker.replace('.JK', '')
      report += f'📌 *{clean_name}* (Harga: Rp {int(last_close)})\n'
      report += f'   • Kenaikan Harian: +{candle_pct:.1f}%\n'
      report += f'   • TP: Rp {int(tp_price)} (+{tp_pct:.1f}%)\n'
      report += f'   • SL: Rp {int(sl_price)} (-{sl_pct:.1f}%)\n\n'

      if found_signal_count >= 10:
        break

    except Exception as e:
      continue

  if found_signal_count == 0:
    report += (
        'Belum ada emiten yang lolos kriteria SMIOO GC & likuiditas hari'
        ' ini. *Cash is King!*'
    )

  send_telegram_message(report)


if __name__ == '__main__':
  main()
