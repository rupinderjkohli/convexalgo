# CONVEXALGOS
import pandas as pd
import numpy as np

import yfinance as yf       #install
import datetime
from datetime import datetime
import time
import globals
# import streamlit as st

def get_hist_info(ticker, period, interval):
  # get historical market data
  print(ticker, period, interval)
  hist = ticker.history(period=period, 
                        interval=interval, 
                        # back_adjust=True, 
                        #auto_adjust=True
                        )
#   print("get_hist_info",hist)

  return hist

def candle_reversal_431(df):
    # 431
    df['strategy_431_long'] = ((df['Close'].shift(4) > df['Close'].shift(3)) &
                (df['Close'].shift(3) > df['Close'].shift(2)) &
                (df['Close'].shift(2) < df['Close'].shift(1))
                )
                
    df['strategy_431_short'] = ((df['Close'].shift(4) < df['Close'].shift(3)) &
                (df['Close'].shift(3) < df['Close'].shift(2)) &
                (df['Close'].shift(2) > df['Close'].shift(1))
                )

    df['candle_reversal_431'] = np.where(df['strategy_431_long'], 'Buy', 
                                        np.where(df['strategy_431_short'], 'Sell', None))
    return df

def ema_new(df,short_window, long_window):
    # EMA
    # Create short exponential moving average column
    # column names for long and short moving average columns
    short_window_col = str(short_window) + '_ema'
    long_window_col = str(long_window) + '_ema'
    df[short_window_col] = df['Close'].ewm(span = short_window, adjust = True).mean()

    # Create a long exponential moving average column
    df[long_window_col] = df['Close'].ewm(span = long_window, adjust = True).mean()

    # create a new column 'Signal' such that if faster moving average is greater than slower moving average 
    # then set Signal as 1 else 0.
    df['Signal'] = 0.0  
    df['Signal'] = np.where(df[short_window_col] > df[long_window_col], 1.0, 0.0) 

    # create a new column 'Position' which is a day-to-day difference of the 'Signal' column. 
    df['ema_crossover'] = df['Signal'].diff()
    return df

def ema_continual(df,short_window, long_window):
    # EMA CONTINUAL
    # Create short exponential moving average column
    short_window_col = str(short_window) + '_ema'
    long_window_col = str(long_window) + '_ema'
    df[short_window_col] = df['Close'].ewm(span = short_window, adjust = True).mean()

    # Create a long exponential moving average column
    df[long_window_col] = df['Close'].ewm(span = long_window, adjust = True).mean()

    df['ema_5above8'] = (df[short_window_col] > df[long_window_col])

    df['ema_continual_long'] = ((df[short_window_col] > df[long_window_col]) & #Ema 5 is above Ema  8
                                        # Last candle (C0) closes above ema5 with low below ema 5 or ema 8 (green candle) and 
                                        # close of C0 candle is less than high of the last two candles
                                        ((df['Close'].shift(1) > df[short_window_col]) &
                                        ((df['Low'].shift(1) < df[short_window_col]) |
                                        (df['Low'].shift(1) < df[long_window_col])) &
                                        (df['Close'].shift(1) <  df['High'].shift(1)) & 
                                        (df['Close'].shift(1) <  df['High'].shift(2))) &
                                        # Low of Candle before C0 (C1) < ema 5 or <  ema 8 
                                        # with high above ema 5(red candle) 
                                        (((df['Low'].shift(2) < df[short_window_col]) |
                                        (df['Low'].shift(2) < df[long_window_col])) &
                                        (df['High'].shift(2) > df[short_window_col]))
                                        )

    df['ema_5below8'] = (df[short_window_col] < df[long_window_col]) 
                                
    df['ema_continual_short'] = ((df[short_window_col] < df[long_window_col]) & #Ema 5 is above Ema  8
                                        # Last candle (C0) closes above ema5 with low below ema 5 or ema 8 (green candle) and 
                                        # close of C0 candle is less than high of the last two candles
                                        ((df['Close'].shift(1) < df[short_window_col]) &
                                        ((df['High'].shift(1) > df[short_window_col]) |
                                        (df['High'].shift(1) > df[long_window_col])) &
                                        (df['Close'].shift(1) >  df['Low'].shift(1)) & 
                                        (df['Close'].shift(1) >  df['Low'].shift(2))) &
                                        # Low of Candle before C0 (C1) < ema 5 or <  ema 8 
                                        # with high above ema 5(red candle) 
                                        (((df['High'].shift(2) > df[short_window_col]) |
                                        (df['High'].shift(2) > df[long_window_col])) &
                                        (df['Low'].shift(2) < df[short_window_col]))
                                        )

    df['ema_continual'] = np.where(df['ema_continual_long'], 'Buy', 
                                        np.where(df['ema_continual_short'], 'Sell', None))
    return df


def main():
# def convexalgos_standalone():    
    # Need to see the output of these functions
    period = "1mo"
    interval= "15m"

    file_path = "symbolList.py" 

    symbols = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            lineSplit = line.split(",")
            symbols.append({'sym': lineSplit[0].strip(), 'qty': lineSplit[1].strip()})

    for sym in symbols:
        stock = sym['sym']
        qty = float(sym['qty'])
        
        print(stock, period, interval)
        
        yf_data = yf.Ticker(stock)
        
        print("stock history",yf_data)
        
        df = get_hist_info(yf_data, period, interval)
        # df = get_selected_stock_history(known_options,selected_period, selected_interval)  
        
        print("stock history",df.head())
        
        print(f'currently doing {stock} with the amount of {qty}')
    
        # CONVEXALGOS
        # Need to see the output of these functions
        print(f'currently doing {stock} with the amount of {qty}')
    
        print("CONVEXALGOS - 4-3-1 Candle Reversal")
        df_candle_reversal_431 = candle_reversal_431(df)
        
        print(df_candle_reversal_431.head(2))
        
        if(df_candle_reversal_431.strategy_431_long.iloc[-1] == 'Buy'):
            print("Long (Buy)")
        elif(df_candle_reversal_431.strategy_431_short.iloc[-1] == 'Sell'):
            print("Short (Sell)")
        
        print("CONVEXALGOS - EMA Crossover")
        ema_new_df = ema_new(df,globals.emaSlowLength,globals.emaFastLength)
        print(ema_new_df.head(2))
        
        if(ema_new_df.ema_crossover.iloc[-1] == '1'):
            print("Long (Buy)")
        elif(ema_new_df.ema_crossover.iloc[-1] == '-1'):
            print("Short (Sell)")
        
        print("CONVEXALGOS - EMA Continual")
        ema_continual_df = ema_continual(df,globals.emaSlowLength,globals.emaFastLength)
        print(ema_continual_df.head(2))
        
        if(ema_continual_df.ema_continual.iloc[-1] == 'Buy'):
            print("Long (Buy)")
        elif(ema_continual_df.ema_continual.iloc[-1] == 'Sell'):
            print("Short (Sell)")
        
if __name__ == '__main__':
  main()    