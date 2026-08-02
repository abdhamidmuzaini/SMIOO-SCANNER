"""
SMI Ergodic + AI Backtest Engine
For SMIOO-SCANNER
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List

class SMIErgodic:
    def __init__(self, long_period=20, short_period=5, signal_period=5):
        self.long_period = long_period
        self.short_period = short_period
        self.signal_period = signal_period
    
    def calculate(self, df):
        # SMI
        price_change = df['Close'].diff()
        abs_change = abs(price_change)
        
        tsi_num = price_change.ewm(span=self.long_period, adjust=False).mean()
        tsi_num = tsi_num.ewm(span=self.short_period, adjust=False).mean()
        
        tsi_den = abs_change.ewm(span=self.long_period, adjust=False).mean()
        tsi_den = tsi_den.ewm(span=self.short_period, adjust=False).mean()
        
        df['SMI'] = 100 * (tsi_num / tsi_den)
        df['SMI_Signal'] = df['SMI'].ewm(span=self.signal_period, adjust=False).mean()
        df['SMI_Histogram'] = df['SMI'] - df['SMI_Signal']
        
        # ATR
        high, low, close = df['High'], df['Low'], df['Close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = df['TR'].rolling(14).mean()
        df['ATR_pct'] = (df['ATR'] / df['Close']) * 100
        
        # Cross Detection
        df['SMI_Cross'] = 0
        df.loc[(df['SMI'].shift(1) <= df['SMI_Signal'].shift(1)) & 
               (df['SMI'] > df['SMI_Signal']), 'SMI_Cross'] = 1  # Golden Cross
        df.loc[(df['SMI'].shift(1) >= df['SMI_Signal'].shift(1)) & 
               (df['SMI'] < df['SMI_Signal']), 'SMI_Cross'] = -1  # Death Cross
        
        # EMA & Volume
        df['EMA20'] = df['Close'].ewm(span=20).mean()
        df['Volume_MA20'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']
        
        return df
    
    def get_signal(self, df):
        latest = df.iloc[-1]
        
        signal = {
            'smi': latest['SMI'],
            'signal': latest['SMI_Signal'],
            'golden_cross': latest['SMI_Cross'] == 1,
            'death_cross': latest['SMI_Cross'] == -1,
            'atr': latest['ATR'],
            'atr_pct': latest['ATR_pct'],
        }
        
        # Position
        if latest['SMI'] > 40:
            signal['position'] = 'OVERBOUGHT'
        elif latest['SMI'] < -40:
            signal['position'] = 'OVERSOLD'
        elif 0 < latest['SMI'] <= 40:
            signal['position'] = 'BULLISH'
        else:
            signal['position'] = 'BEARISH'
        
        # Score
        score = 0
        if signal['golden_cross']: score += 40
        if latest['SMI'] > 0: score += 20
        if latest['SMI'] > latest['SMI_Signal']: score += 20
        if latest['Volume_Ratio'] > 1.5: score += 20
        
        signal['score'] = min(score, 100)
        
        return signal
    
    def calculate_targets(self, df, entry_price):
        latest = df.iloc[-1]
        atr = latest['ATR']
        
        sl_price = max(entry_price * 0.95, entry_price - (atr * 1.5))
        tp_price = entry_price + (atr * 2.25)
        
        return {
            'tp_price': tp_price,
            'tp_pct': (tp_price/entry_price - 1) * 100,
            'sl_price': sl_price,
            'sl_pct': (sl_price/entry_price - 1) * 100,
            'rr_ratio': ((tp_price/entry_price - 1) * 100) / abs((sl_price/entry_price - 1) * 100)
        }


class AIBacktestEngine:
    def __init__(self, smi):
        self.smi = smi
    
    def backtest_setup(self, ticker, df):
        """Backtest setup teknikal"""
        setups = {
            'golden_cross_only': {'trades': 0, 'wins': 0, 'returns': []},
            'golden_cross_above_zero': {'trades': 0, 'wins': 0, 'returns': []},
            'golden_cross_high_volume': {'trades': 0, 'wins': 0, 'returns': []},
            'golden_cross_above_ema': {'trades': 0, 'wins': 0, 'returns': []},
            'golden_cross_perfect': {'trades': 0, 'wins': 0, 'returns': []},
        }
        
        for i in range(50, len(df) - 10):
            if df['SMI_Cross'].iloc[i] == 1:
                entry = df['Close'].iloc[i]
                
                above_zero = df['SMI'].iloc[i] > 0
                high_volume = df['Volume_Ratio'].iloc[i] > 1.5
                above_ema = df['Close'].iloc[i] > df['EMA20'].iloc[i]
                
                exit_price = None
                for j in range(i+1, min(i+11, len(df))):
                    if df['SMI_Cross'].iloc[j] == -1 or j == i+10:
                        exit_price = df['Close'].iloc[j]
                        break
                
                if exit_price:
                    ret = (exit_price / entry - 1) * 100
                    
                    setup_list = ['golden_cross_only']
                    if above_zero: setup_list.append('golden_cross_above_zero')
                    if high_volume: setup_list.append('golden_cross_high_volume')
                    if above_ema: setup_list.append('golden_cross_above_ema')
                    if above_zero and high_volume and above_ema:
                        setup_list.append('golden_cross_perfect')
                    
                    for setup in setup_list:
                        setups[setup]['trades'] += 1
                        setups[setup]['returns'].append(ret)
                        if ret > 0:
                            setups[setup]['wins'] += 1
        
        # Calculate stats
        results = {}
        for name, data in setups.items():
            if data['trades'] >= 3:
                results[name] = {
                    'trades': data['trades'],
                    'win_rate': data['wins'] / data['trades'] * 100,
                    'avg_return': np.mean(data['returns']),
                    'max_return': max(data['returns']),
                    'min_return': min(data['returns']),
                }
        
        return results
    
    def get_best_setup(self, ticker, df):
        setups = self.backtest_setup(ticker, df)
        if not setups:
            return None
        
        best = max(setups.items(), 
                  key=lambda x: x[1]['win_rate'] * x[1]['avg_return'] 
                  if x[1]['avg_return'] > 0 else 0)
        
        return {'name': best[0], 'stats': best[1]}


def check_current_setup(df):
    """Cek setup aktif sekarang"""
    latest = df.iloc[-1]
    
    active = []
    if latest['SMI_Cross'] == 1:
        active.append('golden_cross_only')
    if latest['SMI'] > 0:
        active.append('golden_cross_above_zero')
    if latest['Volume_Ratio'] > 1.5:
        active.append('golden_cross_high_volume')
    if latest['Close'] > latest['EMA20']:
        active.append('golden_cross_above_ema')
    
    perfect = all([
        latest['SMI_Cross'] == 1,
        latest['SMI'] > 0,
        latest['Volume_Ratio'] > 1.5,
        latest['Close'] > latest['EMA20']
    ])
    
    if perfect:
        active.append('golden_cross_perfect')
    
    return {
        'active_setups': active,
        'perfect_setup': perfect,
        'golden_cross': latest['SMI_Cross'] == 1
    }
