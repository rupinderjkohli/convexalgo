
# ##########################################################  
# Purpose: 
# ##########################################################
import pandas as pd
import numpy as np

import yfinance as yf       #install
import datetime
from datetime import datetime
import time
import pytz

# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import csv

import plotly   #install
import plotly.io as pio

import plotly.figure_factory as ff
#importing pybase64 module
import pybase64

# For plotting
import plotly.graph_objects as go
from plotly.subplots import make_subplots


import streamlit as st      #install
from streamlit_js_eval import streamlit_js_eval

# from streamlit_autorefresh import st_autorefresh
# from schedule import every, repeat, run_pending
import streamlit_extras #.metric_cards #import style_metric_cards # beautify metric card with css

# from lightweight_charts import Chart
import time
import asyncio
import nest_asyncio


import base64
from base64 import b64encode

from millify import millify # shortens values (10_000 ---> 10k)

from algotrading_class import *

# from IPython.core.display import HTML # note the library
# from tabulate import tabulate
# from config import Config

# Using plotly dark template
TEMPLATE = 'plotly_dark'

# st.set_page_config(layout='wide', page_title='Stock Dashboard', page_icon=':dollar:')
st.set_page_config(
    page_title="Convex Trades Dashboard",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://convextrades.com/',
        # 'Report a bug': "mailto:rupinder.johar.kohli@gmail.com",
        'About': "#An *extremely* cool app displaying your GoTo Trading Dashboard!"
    }
  ) 

# update every 5 mins
# st_autorefresh(interval=5 * 60 * 1000, key="dataframerefresh")

nest_asyncio.apply()

# print("Plotly Version : {}".format(plotly.__version__))

pd.set_option('display.max_columns', None,
              'display.max_rows', None,
              'display.max_colwidth', None)

pd.options.display.float_format = '${:,.2f}'.format


# ##########################################################  
# Purpose: 
# """## stocks"""
# # ##########################################################
def get_all_stock_info(ticker):
  # get all stock info

  info = ticker.info
  info_df = pd.DataFrame.from_dict([info])
  info_df_short = info_df[['symbol', 'shortName', 'exchange', 'quoteType', 'currency',
                           'previousClose', 'open', 'dayLow', 'dayHigh',
                          #  'category', 
                          # 'navPrice',    # dc, don't know why this is failing?
                          #  'regularMarketPreviousClose', 'regularMarketOpen',
                          #  'regularMarketDayLow', 'regularMarketDayHigh',
                           'fiftyTwoWeekLow', 'fiftyTwoWeekHigh', 'fiftyDayAverage',
                          #  'regularMarketVolume',
                          #  'twoHundredDayAverage',
                          #  'trailingPE', 'volume',
                          #  'averageVolume', 'averageVolume10days',
                          #  'averageDailyVolume10Day', 'bid', 'ask', 'bidSize', 'askSize', 'yield',
                          #  'totalAssets', 'trailingAnnualDividendRate',
                          #  'trailingAnnualDividendYield',
                          #  'ytdReturn', 'beta3Year', 'fundFamily', 'fundInceptionDate',
                          #  'legalType', 'threeYearAverageReturn', 'fiveYearAverageReturn',
                          'underlyingSymbol',
                          #  'longName', 'firstTradeDateEpochUtc', 
                          'timeZoneFullName',
                          #  'timeZoneShortName', 'uuid', 'messageBoardId', 'gmtOffSetMilliseconds',
                          #  'trailingPegRatio'
                            ]]
  info_df_short.reset_index(inplace=True)
  # st.write (info_df_short.to_dict(orient='dict'))
  return info_df_short

# ##########################################################  
# Purpose: 
# ##########################################################
def get_hist_info(ticker, period, interval):
  # get historical market data
  # print(ticker, period, interval)
  hist = ticker.history(period=period, interval=interval, 
                        # back_adjust=True, 
                        auto_adjust=True)

  return hist

# ##########################################################  
# Purpose: 
# ##########################################################
def sma_buy_sell_trigger(df, sma_p1, sma_p2):
  # get historical market data
  # print(ticker, period, interval)
  df['SMA_p1'] = df['Close'].rolling(sma_p1).mean()
  df['SMA_p2'] = df['Close'].rolling(sma_p2).mean()
  df['SMA_Signal'] = 0.0

  #DC review: revisit the rules below
  # If 5 period ema crosses over 10 period ema (note: ema not sma) then go long

  df['SMA_Signal'] = np.where(df['SMA_p1'] > df['SMA_p2'], 1.0, 0.0)

  df['SMA_Position'] = df['SMA_Signal'].diff()

  return df

# ##########################################################  
# Purpose: 
# ##########################################################
# def plot_stk_charts(df):
#   sns.set(style="whitegrid")
#   fig,axs = plt.subplots(3,2, figsize = (8,10))
#   sns.histplot(data=df, x="Open", kde=True, color="skyblue", ax=axs[0, 0])
#   sns.histplot(data=df, x="High", kde=True, color="olive", ax=axs[0, 1])
#   sns.histplot(data=df, x="Low", kde=True, color="gold", ax=axs[1, 0])
#   sns.histplot(data=df, x="Close", kde=True, color="teal", ax=axs[1, 1])
#   sns.histplot(data=df, x="Volume", kde=True, color="teal", ax=axs[2, 0])
#   sns.histplot(data=df, x="Dividends", kde=True, color="blue", ax=axs[2, 1])
#   fig.tight_layout()
#   return

# ##########################################################  
# Purpose: 
# ##########################################################
def get_stk_news(ticker):

  news_df = pd.DataFrame(ticker.news)

  # note the new way of creating column
  news_df = news_df.assign(providerPublishTime_n=lambda x: pd.to_datetime(x.providerPublishTime, unit='s'))

  # display(news_df.info())

  news_df_select = news_df[['title',	'publisher',	'link',	'providerPublishTime_n',	'type'	,'relatedTickers']]

  return news_df_select

# https://coderzcolumn.com/tutorials/data-science/candlestick-chart-in-python-mplfinance-plotly-bokeh#2


# ##########################################################  
# Purpose: 
# ##########################################################
def MovingAverageCrossStrategy(symbol, 
                               stock_df,
                               short_window,
                               long_window, 
                               moving_avg, 
                               display_table = True):
    # st.write("IN MovingAverageCrossStrategy")
    '''
    The function takes the stock symbol, time-duration of analysis, 
    look-back periods and the moving-average type(SMA or EMA) as input 
    and returns the respective MA Crossover chart along with the buy/sell signals for the given period.
    '''
    # stock_symbol - (str)stock ticker as on Yahoo finance. Eg: 'ULTRACEMCO.NS' 
    # start_date - (str)start analysis from this date (format: 'YYYY-MM-DD') Eg: '2018-01-01'
    # end_date - (str)end analysis on this date (format: 'YYYY-MM-DD') Eg: '2020-01-01'
    # short_window - (int)lookback period for short-term moving average. Eg: 5, 10, 20 
    # long_window - (int)lookback period for long-term moving average. Eg: 50, 100, 200
    # moving_avg - (str)the type of moving average to use ('SMA' or 'EMA')
    # display_table - (bool)whether to display the date and price table at buy/sell positions(True/False)

    # column names for long and short moving average columns
    short_window_col = str(short_window) + '_' + moving_avg
    long_window_col = str(long_window) + '_' + moving_avg  
  
    if moving_avg == 'SMA':
        # Create a short simple moving average column
        stock_df[short_window_col] = stock_df['Close'].rolling(window = short_window, min_periods = 1).mean()

        # Create a long simple moving average column
        stock_df[long_window_col] = stock_df['Close'].rolling(window = long_window, min_periods = 1).mean()

    elif moving_avg == 'EMA':
        # Create short exponential moving average column
        stock_df[short_window_col] = stock_df['Close'].ewm(span = short_window, adjust = True).mean()

        # Create a long exponential moving average column
        stock_df[long_window_col] = stock_df['Close'].ewm(span = long_window, adjust = True).mean()
        
        # calculate the stop loss / stop profit
        # Determine Stop-Loss Order
        # A stop-loss order is a request to a broker to sell stocks at a certain price. 
        # These orders aid in minimizing an investor’s loss in a security position.

    # create a new column 'Signal' such that if faster moving average is greater than slower moving average 
    # then set Signal as 1 else 0.
    stock_df['Signal'] = 0.0  
    stock_df['Signal'] = np.where(stock_df[short_window_col] > stock_df[long_window_col], 1.0, 0.0) 

    # create a new column 'Position' which is a day-to-day difference of the 'Signal' column. 
    stock_df['Position'] = stock_df['Signal'].diff()
    
    # ########################################
    # plot close price, short-term and long-term moving averages
    # https://towardsdatascience.com/making-a-trade-call-using-simple-moving-average-sma-crossover-strategy-python-implementation-29963326da7a
    # ########################################
    df_pos = pd.DataFrame()
    previous_triggers = pd.DataFrame()
    
    stock_df, buy_short, sell_long = calculate_atr_buy_sell(stock_df)
    
    if display_table == True:
        df_pos = stock_df[(stock_df['Position'] == 1) | (stock_df['Position'] == -1)]
        df_pos['Position'] = df_pos['Position'].apply(lambda x: 'Buy' if x == 1 else 'Sell')
        previous_triggers = df_pos[['Position']][-6:]
        
    
    # #########################
    # BEGIN: DEBUG_INFO
    st.write(symbol)
    st.write("base data")
    # stock_df = stock_df.reset_index()
    # stock_df.Datetime = stock_df.Datetime.dt.strftime('%Y/%m/%d %H:%M')
    stock_df.index = stock_df.index.strftime('%Y/%m/%d %H:%M')
    st.write(stock_df.sort_index(ascending=False)[:10])
    # stock_df = stock_df.set_index('Datetime')
    
    # END: DEBUG_INFO
    # #########################
    return stock_df, df_pos, previous_triggers

# ##########################################################  
# Purpose: 
# ##########################################################
def get_current_price(symbol, selected_period, selected_interval):
    try:
      ticker = yf.Ticker(symbol)
      todays_data = ticker.history(period = selected_period, interval = selected_interval)
      
    except:
      print("unable to load the ticker current price") 
      return 
    return todays_data['Close'].iloc[-1]

# ##########################################################  
# Purpose: 
# ##########################################################
def show_snapshot(all_tickers):
    # print(str(all_tickers))
    # ticker = yf.Ticker("AAPL", "MSFT")
    ticker = yf.Ticker(all_tickers)
    
    return
  


# ##########################################################  
# Purpose: set up the stock ticker watchlist (user customisation)
# ##########################################################
def save_user_selected_options(selected_tickers):
  df_tickers = pd.DataFrame(selected_tickers)
  try:
    df_tickers.to_csv("user_selected_options.csv", mode='w', index=False, header=True)
  except pd.errors.EmptyDataError:
    print('CSV file is empty save')
  except FileNotFoundError:
    print('CSV file not found save')
  return
  
def load_user_selected_options():
  user_list = []
  try :
    df = pd.read_csv("user_selected_options.csv", header=0)
    
    if (df.empty):
      user_list = []
      
    else: 
      user_list = df['0'].unique()
      print(df['0'].unique())
  except pd.errors.EmptyDataError:
    print('CSV file is empty load')
  except FileNotFoundError:
    print('CSV file not found load')
  
  return user_list
  
def update_selection():
  print("options changed")
  # Session State also supports the attribute based syntax
  if 'ticker_list' not in st.session_state:
      st.session_state.key = 'ticker_list'
      
  print (st.session_state['ticker_list'])
  
  return st.session_state.key 
  
  
# ##########################################################  
# Purpose: timezone challenges
# ##########################################################
#  // you could use this function to convert all your times to required time zone
def timeToTz(originalTime, timeZone): 
  st.write(originalTime)
  # zonedDate = new Date(new Date(originalTime * 1000).toLocaleString('en-US', { timeZone }))
  zonedDate = pd.to_datetime(originalTime)#.dt.tz_localize('UTC').dt.tz_convert(timeZone)
  timestamp_utc = zonedDate.replace(tzinfo=timeZone.utc).timestamp()

  return zonedDate, timestamp_utc

from datetime import datetime, timezone

def unix_timestamp(local_timestamp, local_timezone):
    """turn the input timestamp into a UTC `datetime` object, even though
    the timestamp is not in UTC time, we must do this to construct a datetime
    object with the proper date/time values"""
    dt_fake_utc = datetime.fromtimestamp(local_timestamp, tz=timezone.utc)

    """remove the (incorrect) timezone info that we supplied """
    dt_naive = dt_fake_utc.replace(tzinfo=None)

    """localize the datetime object to our `timezone`. You cannot use
    datetime.replace() here, because it does not account for daylight savings
    time"""
    dt_local = local_timezone.localize(dt_naive)

    """Convert our datetime object back to a timestamp"""
    return int(dt_local.timestamp())
  
  
# ##########################################################  
# Purpose: Function to calculate the Average True Range (ATR)
# For each time period (bar), the true range is simply the greatest of the three price differences:
# High - Low
# High - Previous close
# Previous close - Low
# ########################################################## 
def calculate_atr(data, window=14):
    high_low = data['High'] - data['Low']
    high_close = abs(data['High'] - data['Close'].shift())
    low_close = abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=window).mean()
    
    return atr
    
# ##########################################################  
# Purpose:  Calculate Average True Range (ATR) and its moving average
# ##########################################################  
def calculate_atr_buy_sell(data):
  data['atr'] = calculate_atr(data)
  data['atr_ma'] = data['atr'].rolling(window=14).mean()  # 14-day moving average of ATR

  # NOT IN USE
  # Define buy and sell signals
  buy_signal = (data['atr'] > data['atr_ma']) & (data['atr'].shift(1) <= data['atr_ma'].shift(1))
  sell_signal = (data['atr'] < data['atr_ma']) & (data['atr'].shift(1) >= data['atr_ma'].shift(1))
  
  buy_long_idx = data.index[buy_signal]
  sell_short_idx = data.index[sell_signal]
  
  buy_long = buy_signal.loc[buy_signal==True]
  sell_short = sell_signal.loc[sell_signal==True]
  
  return data, buy_long, sell_short



# Bullish Candle — Green / Bull / Long CandleStick
# Bearish Candle — Red / Bear / Short CandleStick
# https://medium.com/@letspython3.x/learn-and-implement-candlestick-patterns-python-6de09854fa3e
def is_candle_properties(df):
  # st.write(open, close)
  # df['candle_type'] = np.where(df['Open'] < df['Close'], "green", "red") 
  df['candle_length'] = df['High'] - df['Low']
  df['bodyLength'] = abs(df['Open'] - df['Close'])
  """Calculate and return the length of lower shadow or wick."""
  df['lowerWick'] = (df['Open'] if df['Open'] <= df['Close'] else df['Close']) - df['Low']
  """Calculate and return the length of upper shadow or wick."""                
  df['upperWick'] = df['High'] - (df['Open'] if df['Open'] >= df['Close'] else df['Close'])

  return df

def strategy_431(
  # symbol,
                 df, #to find the prev 3 candles
                #  candle_obj,
                #  is_sorted, #if the df is sorted in reverse order of dates
                 
                #  selected_short_window,
                #  selected_long_window,
                #  algo_strategy
                 ):
  
  # If ((close of previous candle(c1) > Close of the candle before (c2))
  # AND (close of the candle before (c2) is > the close of candle before (c3))
  # AND (last candle (c0) close < close of c1)
  # AND (last candle(c0) close > low of c2)
  # AND (last candle close < last candle open)
  
  # for long - three white soldiers
  # https://trendspider.com/learning-center/thestrat-candlestick-patterns-a-traders-guide/
  # https://www.babypips.com/learn/forex/triple-candlestick-patterns
  # https://bullishbears.com/3-bar-reversal-pattern/

  # close of 4th less than close of 3rd - define the trend; should be same - down / up
  # close of 3rd less than close of 2nd - define the trend
  # 1st candle should now close below the close of the second
  
  # first candle = candle at the top of the frame (now - 5 min (interval)
  # second candle is now - 10 min (interval *2
  # third candle is now - 15 min (interval * 3)
  # fourth candle is now - 20 min (interval * 4)
  
  # for short 
  # close of 3rd less than close of 2nd
  # close of 2nd less than close of 1st
  
  df_3_whites = candle_three_white_soldiers(df)
  st.write(df_3_whites)
  return

# https://eodhd.com/financial-academy/technical-analysis-examples/practical-guide-to-automated-detection-trading-patterns-with-python
# 
def candle_three_white_soldiers(df) -> pd.Series:
  """*** Candlestick Detected: Three White Soldiers ("Strong - Reversal - Bullish Pattern - Up")"""

  # Fill NaN values with 0
  df = df.fillna(0)

  return (
      ((df["Open"] > df["Open"].shift(1)) & (df["Open"] < df["Close"].shift(1)))
      & (df["Close"] > df["High"].shift(1))
      & (df["High"] - np.maximum(df["Open"], df["Close"]) < (abs(df["Open"] - df["Close"])))
      & ((df["Open"].shift(1) > df["Open"].shift(2)) & (df["Open"].shift(1) < df["Close"].shift(2)))
      & (df["Close"].shift(1) > df["High"].shift(2))
      & (
          df["High"].shift(1) - np.maximum(df["Open"].shift(1), df["Close"].shift(1))
          < (abs(df["Open"].shift(1) - df["Close"].shift(1)))
      )
  )


def candle_three_black_crows(df) -> pd.Series:
  """* Candlestick Detected: Three Black Crows ("Strong - Reversal - Bearish Pattern - Down")"""

  # Fill NaN values with 0
  df = df.fillna(0)

  return (
      ((df["Open"] < df["Open"].shift(1)) & (df["Open"] > df["Close"].shift(1)))
      & (df["Close"] < df["Low"].shift(1))
      & (df["Low"] - np.maximum(df["Open"], df["Close"]) < (abs(df["Open"] - df["Close"])))
      & ((df["Open"].shift(1) < df["Open"].shift(2)) & (df["Open"].shift(1) > df["Close"].shift(2)))
      & (df["Close"].shift(1) < df["Low"].shift(2))
      & (
          df["Low"].shift(1) - np.maximum(df["Open"].shift(1), df["Close"].shift(1))
          < (abs(df["Open"].shift(1) - df["Close"].shift(1)))
      )
  )
  
  
def candle_four_three_one_soldiers(df, is_sorted) -> pd.Series:
  """*** Candlestick Detected: Three White Soldiers ("Strong - Reversal - Bullish Pattern - Up")
  # close of 4th less than close of 3rd - define the trend; should be same - down / up
  # close of 3rd less than close of 2nd - define the trend
  # 1st candle should now close below the close of the second
  

  # for long
  # close of 4th greater than close of 3rd
  # close of 3rd greater than close of 2nd -
  # close of 2nd less than close of 1st
  
  # for short
  # close of 4th less than close of 3rd
  # close of 3rd less than close of 2nd -
  # close of 2nd higher than close of 1st

  """
  if(~is_sorted):
    df = df.sort_index(ascending = False)
  # Fill NaN values with 0
  df = df.fillna(0)
  # print(df.head())
  df_evaluate = df[['Open','Close', 'High', 'Low']]
  df_evaluate['t3'] = df_evaluate['Close'].shift(3)
  df_evaluate['t2'] = df_evaluate['Close'].shift(2)
  df_evaluate['t1'] = df_evaluate['Close'].shift(1)
  df_evaluate['t0'] = df_evaluate['Close'].shift(0)
  
  df_evaluate = df_evaluate.fillna(0)
  
  df_evaluate['candle_type'] = np.where(df_evaluate['Open'] - df_evaluate['Close']<0, "Green", "Red")
  # close of 4th less than close of 3rd - define the trend; should be same - down / up
  df_evaluate['trend_4_R'] = np.where ((df_evaluate['Close'].shift(3) - df_evaluate['Close'].shift(2))>0, "Up" , "Down")
  # close of 3rd less than close of 2nd - define the trend
  df_evaluate['trend_3_R'] = np.where ((df_evaluate['Close'].shift(2) - df_evaluate['Close'].shift(1))>0, "Up" , "Down")
  # 1st candle should now close below the close of the second
  df_evaluate['trend_1_R'] = np.where ((df_evaluate['Close'].shift(0) - df_evaluate['Close'].shift(1))<0, "Up" , "Down")
  
  df_evaluate['trend_4'] = df_evaluate['Close'].shift(3) - df_evaluate['Close'].shift(2)
  df_evaluate['trend_3'] = df_evaluate['Close'].shift(2) - df_evaluate['Close'].shift(1) 
  df_evaluate['trend_1'] = df_evaluate['Close'].shift(1) - df_evaluate['Close'].shift(0) 
  
  # for long
  # close of 4th greater than close of 3rd
  # close of 3rd greater than close of 2nd -
  # close of 2nd less than close of 1st
  df_evaluate['strategy_431_long'] = ((df['Close'].shift(3) > df['Close'].shift(2)) &
              (df['Close'].shift(2) > df['Close'].shift(1)) &
              (df['Close'].shift(1) < df['Close'].shift(0))
              )
  
  # for short
  # close of 4th less than close of 3rd
  # close of 3rd less than close of 2nd -
  # close of 2nd higher than close of 1st

  df_evaluate['strategy_431_short'] = ((df['Close'].shift(3) < df['Close'].shift(2)) &
              (df['Close'].shift(2) < df['Close'].shift(1)) &
              (df['Close'].shift(1) > df['Close'].shift(0))
              )
  
  # print(position)
  

  return df_evaluate #, position
