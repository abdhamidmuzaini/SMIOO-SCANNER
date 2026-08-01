import os
import requests
import numpy as np
import pandas as pd
import yfinance as yf

# Konfigurasi Telegram & Modal Dasar
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
CHAT_ID = os.environ.get('CHAT_ID')
BASE_MODAL = 5000000  # Modal dasar Rp 5.000.000


def send_telegram_message(message):
  if not TELEGRAM_TOKEN or not CHAT_ID:
    print('Token atau Chat ID Telegram belum diset!')
    return
  url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'

  # Pecah pesan jika melebihi batas 4000 karakter Telegram
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


def check_market_regime():
  try:
    ihsg = yf.download('^JKSE', period='3mo', progress=False)
    if ihsg.empty:
      return 1.0, 'Netral (Default)'

    if isinstance(ihsg.columns, pd.MultiIndex):
      close_col = ihsg['Close'].iloc[:, 0]
    else:
      close_col = ihsg['Close']

    current_close = close_col.iloc[-1]
    ma20 = close_col.rolling(window=20).mean().iloc[-1]
    ma50 = close_col.rolling(window=50).mean().iloc[-1]

    if current_close > ma20 and ma20 > ma50:
      return 1.0, '🟢 BULLISH KUAT (Alokasi 100%)'
    elif current_close > ma50:
      return 0.6, '🟡 SIDEWAYS / WSP (Alokasi 60%)'
    else:
      return 0.2, '🔴 BEARISH / DOWN (Mode Defensif 20%)'
  except Exception as e:
    print(f'Gagal cek IHSG: {e}')
    return 0.5, 'Netral (Fallback)'


def main():
  regime_multiplier, regime_status = check_market_regime()
  adjusted_modal = BASE_MODAL * regime_multiplier

  report = f'📊 *STATUS REZIM PASAR:* {regime_status}\n'
  report += f'💰 *Alokasi Modal Efektif:* Rp {int(adjusted_modal):,}\n\n'
  report += '🚀 *Daftar Sinyal Saham (SMIOO + ATR):*\n\n'

  try:
    with open('tickers.txt', 'r') as f:
      raw_tickers = [line.strip() for line in f if line.strip()]
  except FileNotFoundError:
    raw_tickers = ['BSML', 'ADRO', 'ANTM']

  found_signal_count = 0

  for raw in raw_tickers:
    # Bersihkan ticker: hapus tanda $, spasi, dan pastikan berakhiran .JK
    ticker = raw.replace('$', '').strip().upper()
    if not ticker.endswith('.JK'):
      ticker += '.JK'

    try:
      df = yf.download(ticker, period='6mo', interval='1d', progress=False)
      if df.empty or len(df) < 50:
        continue

      if isinstance(df.columns, pd.MultiIndex):
        close = df['Close'].iloc[:, 0]
        high = df['High'].iloc[:, 0]
        low = df['Low'].iloc[:, 0]
      else:
        close = df['Close']
        high = df['High']
        low = df['Low']

      tr1 = high - low
      tr2 = abs(high - close.shift())
      tr3 = abs(low - close.shift())
      tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
      atr = tr.rolling(14).mean().iloc[-1]

      last_close = float(close.iloc[-1])
      tp_price = last_close + (1.5 * atr)
      sl_price = last_close - (1.0 * atr)

      tp_pct = ((tp_price - last_close) / last_close) * 100
      sl_pct = ((last_close - sl_price) / last_close) * 100

      max_alloc_per_stock = adjusted_modal * 0.25
      shares_to_buy = max_alloc_per_stock / last_close
      lots = int(shares_to_buy // 100)
      if lots < 1:
        lots = 1

      found_signal_count += 1
      clean_name = ticker.replace('.JK', '')
      report += f'📌 *{clean_name}* (Harga: Rp {int(last_close)})\n'
      report += f'   • TP: Rp {int(tp_price)} (+{tp_pct:.1f}%)\n'
      report += f'   • SL: Rp {int(sl_price)} (-{sl_pct:.1f}%)\n'
      report += f'   • Rekomendasi Beli: *{lots} Lot*\n\n'

    except Exception as e:
      continue

  if found_signal_count == 0:
    report += (
        'Belum ada emiten yang memenuhi kriteria kuat hari ini. *Cash is King!*'
    )

  send_telegram_message(report)


if __name__ == '__main__':
  main()
