
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

# To read external property file
from jproperties import Properties

from algotrading_class import *

from algotrading_algos import *


# Using plotly dark template
TEMPLATE = 'plotly_dark'

# st.set_page_config(layout='wide', page_title='Stock Dashboard', page_icon=':dollar:')


# update every 5 mins
# st_autorefresh(interval=5 * 60 * 1000, key="dataframerefresh")

nest_asyncio.apply()

# print("Plotly Version : {}".format(plotly.__version__))

pd.set_option('display.max_columns', None,
              'display.max_rows', None,
              'display.max_colwidth', None)

pd.options.display.float_format = '${:,.2f}'.format

@st.cache_resource
def load_config(refresh):
  configs = Properties()

  with open('./config.properties', 'rb') as config_file:
      configs.load(config_file)

  SYMBOLS = configs.get('SYMBOLS').data.split(',') 
  
  # get the following config variables to session state
  PERIOD = configs.get('PERIOD')
  INTERVAL = configs.get('INTERVAL')
  STOP_LOSS = configs.get('STOP_LOSS')
  TAKE_PROFIT = configs.get('TAKE_PROFIT')
  MOVING_AVERAGE_BASED = configs.get('MOVING_AVERAGE_BASED').data.split(',')
  TREND_BASED = configs.get('TREND_BASED').data.split(',')
  
  
  # print("SYMBOLS")
  # print(PERIOD)
  # SYMBOLS = SYMBOLS.sort()
  if refresh:
    return SYMBOLS
  else:
    return SYMBOLS, PERIOD, INTERVAL, STOP_LOSS, TAKE_PROFIT, MOVING_AVERAGE_BASED, TREND_BASED

  
  

# ##########################################################  
# Purpose: 
# """## stocks"""
# # ##########################################################
def get_all_stock_info(ticker):
  # get all stock info

  info = ticker.info
  info_df = pd.DataFrame.from_dict([info])
  # info_df_short = info_df[['symbol', 'shortName', 'exchange', 'quoteType', 'currency',
  #                          'previousClose', 'open', 'dayLow', 'dayHigh',
  #                          'category', 
  #                         # 'navPrice',    # dc, don't know why this is failing?
  #                          'regularMarketPreviousClose', 'regularMarketOpen',
  #                          'regularMarketDayLow', 'regularMarketDayHigh',
  #                          'fiftyTwoWeekLow', 'fiftyTwoWeekHigh', 'fiftyDayAverage',
  #                          'regularMarketVolume',
  #                          'twoHundredDayAverage',
  #                          'trailingPE', 'volume',
  #                          'averageVolume', 'averageVolume10days',
  #                          'averageDailyVolume10Day', 'bid', 'ask', 'bidSize', 'askSize', 'yield',
  #                          'totalAssets', 'trailingAnnualDividendRate',
  #                          'trailingAnnualDividendYield',
  #                          'ytdReturn', 'beta3Year', 'fundFamily', 'fundInceptionDate',
  #                          'legalType', 'threeYearAverageReturn', 'fiveYearAverageReturn',
  #                         'underlyingSymbol',
  #                          'longName', 'firstTradeDateEpochUtc', 
  #                         'timeZoneFullName',
  #                          'timeZoneShortName', 'uuid', 'messageBoardId', 'gmtOffSetMilliseconds',
  #                          'trailingPegRatio',
  #                           ]]
  info_df.reset_index(inplace=True)
  # info_df_short.reset_index(inplace=True)
  # st.write (info_df_short.to_dict(orient='dict'))
  return info_df #info_df_short

# ##########################################################  
# Purpose: 
# ##########################################################
def get_hist_info(ticker, period, interval):
  # get historical market data
  # print(ticker, period, interval)
  hist = ticker.history(period=period, 
                        interval=interval, 
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

  try:
    news_df = pd.DataFrame(ticker.news)
  except:
    st.write("ERROR")
    return pd.DataFrame()

  # note the new way of creating column
  news_df = news_df.assign(providerPublishTime_n=lambda x: pd.to_datetime(x.providerPublishTime, unit='s'))

  # display(news_df.info())

  news_df_select = news_df[['title',	'publisher',	'link',	'providerPublishTime_n',	'type'	,'relatedTickers']]

  return news_df_select

# https://coderzcolumn.com/tutorials/data-science/candlestick-chart-in-python-mplfinance-plotly-bokeh#2


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
  
  



# Bullish Candle — Green / Bull / Long CandleStick
# Bearish Candle — Red / Bear / Short CandleStick
# https://medium.com/@letspython3.x/learn-and-implement-candlestick-patterns-python-6de09854fa3e
def candle_properties(df):
  # st.write(open, close)
  # df['candle_type'] = np.where(df['Open'] < df['Close'], "green", "red") 
  # df['candle_length'] = df['High'] - df['Low']
  # df['bodyLength'] = abs(df['Open'] - df['Close'])
  # """Calculate and return the length of lower shadow or wick."""
  # df['lowerWick'] = (df['Open'] if df['Open'] <= df['Close'] else df['Close']) - df['Low']
  # """Calculate and return the length of upper shadow or wick."""                
  # df['upperWick'] = df['High'] - (df['Open'] if df['Open'] >= df['Close'] else df['Close'])
  
  df['candle_type'] = np.where(df['Open'] < df['Close'], "green", "red") 
  df['candle_length'] = df['High'] - df['Low']
  df['bodyLength'] = abs(df['Open'] - df['Close'])
  
  # # """Calculate and return the length of lower shadow or wick."""
  df['lowerWick'] = np.where(df['Open'] <= df['Close'], 
                                        df['Open'], 
                                        df['Close']) - df['Low']
  
  # # """Calculate and return the length of upper shadow or wick."""                
  df['upperWick'] = df['High'] - np.where(df['Open'] >= df['Close'], 
                                        df['Open'], 
                                        df['Close'])

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
  
  
 

# ##########################################################  
# Purpose: Basic EDA
# ##########################################################
def historical_overview(df): 
  df_overview = {}
  df_overview_df = pd.DataFrame() 
  # print(df.columns)
  
  # df_overview = pd.DataFrame(df['Open','Close'].describe(include='all'))
  # print(df.describe())
  
  period_high = df['High'].mean()
  period_low = df['Low'].mean()
  
  # print(df['Open'].min(), df['Open'].max(), df['Open'].std())
  period_close_min = df['Close'].min()
  period_close_max = df['Close'].max()
  period_close_std = df['Close'].std()
  
  period_open_max = df['Open'].max()
  period_open_min = df['Open'].min()
  period_open_std = df['Open'].std()
  
  for variable in ["period_close_min",
                  "period_close_max",
                  "period_close_std",
                  "period_open_min",
                  "period_open_min",
                  "period_open_std",
                  "period_high",
                  "period_low"
                  ]:
    df_overview[variable] = eval(variable)
    # print(df_overview)
  df_overview_df = pd.DataFrame([df_overview])
  # print(x)
  
  # df_overview_df = pd.concat([x, df_overview], ignore_index=True)
  
  return df_overview_df



  
  
  
def display_watchlist():
  # expander = st.expander("Selected Stocks")
  
  user_sel_list = []

  # load_user_selected_options()
  
  user_sel_list = load_user_selected_options()
  
  st.session_state.user_watchlist = user_sel_list
  
  return user_sel_list

def customize(expander):
    ticker_list = ""
    ticker_list = expander.text_area(":red[enter the ticker list seperated with commas]",
                                key='new_ticker'
        )
    
    if (expander.button("Update Ticker")):
        with open('config.properties', 'r', encoding='utf-8') as file: 
            data = file.readlines() 
        
            # print(data[1]) 
            data[1] = data[1].replace('\n', '')
            # print("postsplit", data[1])
            
            data[1] = data[1]+","+ticker_list+"\n"
            # print(data[1])
            
            # print(data)
        
        with open('config.properties', 'w', encoding='utf-8') as file: 
            file.writelines(data) 
        
        # ticker_list = ""
    return ticker_list
  

def setup_day(user_sel_list, period, interval, symbol_list, algo_functions_map):
  st.markdown(
      """
      Welcom to Convex Trades, a one stop solution enabling you to 
      - find stocks, 
      - enter and exit trades based on predefined and proven strategies
  """
  )
  
  # st.markdown(
  #     """
  #     ### You are currently setup as:
  #     """
  # )
  with st.container(): # CONTAINER: current day setup
    # Display list horizontally with HTML/CSS
    
    
    # st.markdown("<div style='display:flex;'>Convex Algo Strategy:  {} <div> "
    #             .format(algo_strategy), unsafe_allow_html=True)

    st.write("")
    
  st.markdown(
      """
      ### Customise your trading day
      """
  )
  # **👈 Select your focus of the day sidebar** 
  
  st.write("---")  # Add a horizontal rule
    
  with st.container(): # CONTAINER: ticker selection         
    
    st.markdown("<div style='display:flex;'>The current selected Stocks watchlist:  {} <div> "
                .format(" , ".join(["<div> {} </div>".format(item) for item in user_sel_list])), unsafe_allow_html=True)

    st.write("")
    
    expander = st.expander("Select Stocks Watchlist")
  
    # ticker selection
    multiselect_placeholder = expander.empty()
    ticker = multiselect_placeholder.multiselect('Selected Ticker(s)', options=symbol_list,
                                  help = 'Select a ticker', 
                                  key='ticker_list',
                                  max_selections=8,
                                  default= user_sel_list, #["TSLA"],
                                  placeholder="Choose an option",
                                  # on_change=update_selection(),
                                  )
    # print(ticker)
    # print(st.session_state)
    known_options = ticker
    save_user_selected_options(ticker)
    refresh = False
      
  is_customize = expander.toggle("Customize List")
  with st.container():
    if is_customize:
      with st.expander("Customize Stocks Watchlist"):
        # st.write("Customize Stocks List.")
        new_ticker_list = customize(expander)
        if (len(new_ticker_list)!=0):
          refresh = True
        if refresh:
          symbol_list = load_config(refresh)
          symbol_list = np.sort(symbol_list)
          # Clear the existing multiselect widget
          multiselect_placeholder.empty()
          ticker = multiselect_placeholder.multiselect('Selected Ticker(s)', options=symbol_list,
                                  # help = 'Select a ticker', 
                                  # key='ticker_list',
                                  # max_selections=8,
                                  default= user_sel_list, #["TSLA"],
                                  # placeholder="Choose an option",
                                  # on_change=update_selection(),
                                  )

          # Recreate the multiselect with updated options
          toast_message = (":red[Ticker list updated]"
                      )
          st.toast(toast_message, icon='🏃')
          return #symbol_list, stop_loss, take_profit
        else:
          return
  # CONTAINER: ticker selection
  st.write("---")  # Add a horizontal rule

  st.markdown("<div style='display:flex;'>The current selected Trading Period:  {}  & Interval:  {} <div> "
                .format(period,interval), unsafe_allow_html=True)
  st.write("")  
    # st.markdown("<div style='display:flex;'>Trading Interval:  {} <div> "
    #             .format(interval), unsafe_allow_html=True)
    
  with st.container(): # CONTAINER: Strategy selection 
    expander = st.expander("Select Trading Period")
    # period selection
    selected_period = expander.selectbox(
        'Select Period', options=['1d','5d','1mo','3mo', '6mo', 'YTD', '1y', 'all'], index=1)
    
    # interval selection
    selected_interval = expander.selectbox(
        'Select Intervals', options=['1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo'], index=2)
    
    
  st.write("---")  # Add a horizontal rule

  with st.container(): # CONTAINER: Strategy selection 
    expander = st.expander("Select Trading Strategy")
    
    algo_name, algo_functions = list(algo_functions_map)
    
    # Checkbox selection
    selected_algos = []
    selected_options = [expander.checkbox(option, value=False) for option in algo_functions_map[0]]
    for option, selected in zip(algo_name, selected_options):    
        if selected:
          selected_algos.append(option)
  
  print("selected_algos ",selected_algos)
  st.write("---")  # Add a horizontal rule
  return known_options, selected_algos
  
async def signals_view(known_options, selected_algos, selected_period, selected_interval):
  # generate summary
  df_summary_view = pd.DataFrame()
  
  combined_trading_summary = []
  combined_trading_summary_df = pd.DataFrame()
  
  tasks = []
  
  if (len(selected_algos) == 0):
    selected_algos = ['5/8 SMA', '5/8 EMA', '5/8 EMA 1-2 candle price','4-3-1 candle price reversal']
  
  st.markdown("<div style='display:flex;'>Stocks watchlist:  {} <div> "
              .format(" , ".join(["<div> {} </div>".format(item) for item in known_options])), unsafe_allow_html=True)
  st.write("")
  
  st.markdown("<div style='display:flex;'>The current selected Trading Period:  {}  & Interval:  {} <div> "
                .format(st.session_state.period,st.session_state.interval), unsafe_allow_html=True)
  st.write("")  
  
  st.markdown("<div style='display:flex;'>Trading Strategy:  {} <div> "
              .format(st.session_state.selected_algos), unsafe_allow_html=True)
  st.write("")
  
  # st.session_state.page_subheader = '{0} ({1})'.format(yf_data.info['shortName'], yf_data.info['symbol'])
  # st.subheader(st.session_state.page_subheader)
    
          
  for symbol in known_options:
    # get ticker data
    yf_data = yf.Ticker(symbol) #initiate the ticker
    # st.session_state.page_subheader = '{0} ({1})'.format(yf_data.info['shortName'], yf_data.info['symbol'])
    # st.subheader(st.session_state.page_subheader)
    
    # generate summary
    etf_summary_info = get_all_stock_info(yf_data)
    # st.write(etf_summary_info.T)
    df_details = etf_summary_info[['symbol', 'shortName','quoteType','financialCurrency',
                                    'industry','sector','currentPrice','recommendationKey', 
                                    'fiftyTwoWeekHigh','fiftyTwoWeekLow', 
                                    'grossMargins','ebitdaMargins']]
    df_details['52w Range'] = ((df_details['currentPrice'] - df_details['fiftyTwoWeekLow'])/(df_details['fiftyTwoWeekHigh'] - df_details['fiftyTwoWeekLow']))*100
    
    df_summary_view = pd.concat([df_summary_view, df_details], ignore_index=True)
    # Add 52 week price range
    
    # generate trading summary
    # based on the selected algo strategy call the selected functions
    # st.write("getting summary for: ", symbol)
    stock_hist_df = get_hist_info(yf_data, selected_period, selected_interval)
    
    # await asyncio.sleep(1)
    # use gather instead of run
    
    
    # trading_summary = await asyncio.gather(algo_trading_summary(symbol, 
    #                                  stock_hist_df,
    #                                  selected_algos, 
    #                                  selected_period, 
    #                                  selected_interval,
    #                                  )
    # )
    # print("summary_view", stock_hist_df.columns)
    tasks.append(algo_trading_summary(symbol, 
                                     stock_hist_df,
                                     selected_algos, 
                                     selected_period, 
                                     selected_interval,
                                     )
                 )
    
    # # Combine results into a single list of dictionaries
    # # st.write(trading_summary)
    # # st.write(type(trading_summary))
    # # st.write(len(trading_summary))
    # for i, result in enumerate(trading_summary):
    #   # st.write("len(result)",len(result))
    #   combined_trading_summary.append(result[i])
    # # st.write(combined_trading_summary)
    
  results = await asyncio.gather(*tasks)
  
      
  # present view  
  # st.write(df_summary_view.sort_values(by='symbol',ascending=True))
  # st.write("getting trading view for: ", selected_algos)
  # st.write(combined_trading_summary)
  # st.write("Results:", results)
  
  # Flatten the list nested structure
  flattened_data = [item for sublist in results for item in sublist]
  
  # Create a DataFrame from the list of dictionaries
  combined_trading_summary_df = pd.DataFrame(flattened_data)
  # print(combined_trading_summary_df.columns)
  # Index(['symbol', 'algo_strategy', 'stock_trigger_at', 'stock_trigger_state',
  #      'stock_price_at_trigger', 'stock_stop_loss_atr',
  #      'stock_take_profit_atr', 'stock_atr_ma', 'stock_ema_p1', 'stock_ema_p2',
  #      'stock_previous_triggers'],
  #     dtype='object')
  combined_trading_summary_df = combined_trading_summary_df[['symbol', 
                                                             'stock_trigger_state',
                                                             'stock_trigger_at', 
                                                             'stock_price_at_trigger', 
                                                             'stock_stop_loss_atr',
                                                             'stock_take_profit_atr',
                                                             'algo_strategy', 
                                                             'stock_previous_triggers',
                                                             ]].sort_values(by = ['stock_trigger_at', 'symbol'], ascending=[False, True])
  st.data_editor(
    combined_trading_summary_df,
    column_config={"symbol": st.column_config.TextColumn(
        "Ticker",
        width="small"
    ),
                   "algo_strategy": st.column_config.TextColumn(
        "Strategy Name",
        width="small"
    ),
                   "stock_trigger_state": st.column_config.TextColumn(
        "Trigger",
        width="small"
    ),
                    "stock_take_profit_atr": st.column_config.NumberColumn(
        "Take Profit Price",
        format="%.2f",
    ),
                    "stock_stop_loss_atr": st.column_config.NumberColumn(
        "Stop Loss Price",
        format="%.2f",
    ),
                    "stock_price_at_trigger": st.column_config.NumberColumn(
        "Trigger Price",
        format="%.2f",
    ),
                    "stock_atr_ma": st.column_config.NumberColumn(
        "ATR MA",
        format="%.2f",
    ),
                    "stock_previous_triggers": st.column_config.ListColumn(
        "Previous Triggers",
        # format="DD MMM YYYY, HH:MM"
        # width="medium",
    ),
                    "stock_trigger_at": st.column_config.DatetimeColumn(
        "Trigger Time",
        format = "YYYY-MM-DD HH:mm"
        # format="DD MMM YYYY, HH:MM"
    ),
        # "stock_view_details": st.column_config.LinkColumn
        # (
        #     "Stock Details",
        #     help="The top trending Streamlit apps",
        #     max_chars=100,
        #     display_text="view table",
        #     # default=add_container(etf_data[symbol], quick_explore_df[symbol])
        # ),
        
    },
    height=None,
    use_container_width=True,
    hide_index=True,
    )
  
  return
  
  
  
async def stock_status(known_options, selected_algos, selected_period, selected_interval):
  # generate stocks list view
  # st.write(known_options, selected_algos, selected_period, selected_interval)
  
  # await asyncio.sleep(1)
  for symbol in known_options:
    # get ticker data
    yf_data = yf.Ticker(symbol) #initiate the ticker
    st.write("fetching status for: ", symbol )
    stock_hist_df = get_hist_info(yf_data, selected_period, selected_interval)
    
    # func1 = strategy_sma(symbol,
    #              stock_hist_df,
    #              selected_period, 
    #              selected_interval,
    #              algo_strategy = "SMA",
    #              selected_short_window = 5,
    #              selected_long_window = 8
    #              )
    # st.write(func1)
    
    status_strategy_ema = await strategy_ema(symbol,
                 stock_hist_df,
                 selected_period, 
                 selected_interval,
                 algo_strategy = "EMA",
                 selected_short_window = 5,
                 selected_long_window = 8,
                 is_summary = False,
                 )
    # EMA quick_explore + the following columns
    # stock_df[short_window_col]; stock_df[long_window_col]
    # stock_df['Signal']; stock_df['Position']
    # print("status_strategy_ema")
    # print(status_strategy_ema.columns)
    # print()
    # status_strategy_ema
    # Index(['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits',
    #    '5_EMA', '8_EMA', 'Signal', 'Position', 'atr', 'atr_ma',
    #    'stop_loss_atr', 'take_profit_atr'],
    #   dtype='object')
    status_strategy_ema = status_strategy_ema[['Close', 
       '5_EMA', '8_EMA', 'Signal', 'Position', 'atr_ma',
       'stop_loss_atr', 'take_profit_atr']]
    # st.write("EMA", status_strategy_ema.sort_index(ascending=False))
    
    status_strategy_ema_continual = await strategy_ema_continual(symbol,
                                 stock_hist_df,
                                 selected_period, 
                                 selected_interval,
                                 algo_strategy = "EMA 1-2 candle price",
                                 selected_short_window = 5,
                                 selected_long_window = 8,
                                 is_summary = False,
                                 )
    # ema_continual + the following columns
    # stock_df['ema_5above8'];stock_df['t0_close_aboveema5'];stock_df['t0_low_belowema5'];stock_df['ema_continual_long'];
    # stock_df['ema_5below8'];stock_df['t0_close_belowema5'];stock_df['t0_low_aboveema5'];stock_df['ema_continual_short']
    # print("status_strategy_ema_continual")
    # print(status_strategy_ema_continual.columns)
    # print()
    # Index(['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits',
    #    '5_EMA', '8_EMA', 'Signal', 'Position', 'atr', 'atr_ma',
    #    'stop_loss_atr', 'take_profit_atr', '5_EMA 1-2 candle price',
    #    '8_EMA 1-2 candle price', 'ema_5above8', 't0_close_aboveema5',
    #    't0_low_belowema5', 'ema_continual_long', 'ema_5below8',
    #    't0_close_belowema5', 't0_low_aboveema5', 'ema_continual_short'],
    #   dtype='object')
    status_strategy_ema_continual = status_strategy_ema_continual[['Close', 
       '5_EMA', '8_EMA', 'Signal', 'Position', 'atr', 'atr_ma',
       'stop_loss_atr', 'take_profit_atr', '5_EMA 1-2 candle price',
       '8_EMA 1-2 candle price', 'ema_5above8', 't0_close_aboveema5',
       't0_low_belowema5', 'ema_continual_long', 'ema_5below8',
       't0_close_belowema5', 't0_low_aboveema5', 'ema_continual_short']]
    # st.write("EMA 1-2 candle price", status_strategy_ema_continual.sort_index(ascending=False))
    
    
    status_strategy_431_reversal = await strategy_431_reversal(symbol,
                                 stock_hist_df,
                                 selected_period, 
                                 selected_interval,
                                 is_summary = False,
                                 algo_strategy = "4-3-1 candle price reversal",
                                 )
    
    print("status_strategy_431_reversal")
    print(status_strategy_431_reversal.columns)
    print()
    # Index(['Open', 'Close', 'High', 'Low', 'strategy_431_long',
    #    'strategy_431_short', 'position', 'atr', 'atr_ma', 'stop_loss_atr',
    #    'take_profit_atr'],
    #   dtype='object')
    status_strategy_431_reversal = status_strategy_431_reversal
    # [['Close', 'strategy_431_long_c1',
    #    'strategy_431_long_c2', 'strategy_431_long_c3', 'strategy_431_long',
    #    'strategy_431_short_c1', 'strategy_431_short_c2',
    #    'strategy_431_short_c3', 'strategy_431_short', 'atr', 'position',
    #    'atr_ma', 'stop_loss_atr', 'take_profit_atr']]
    # st.write("4-3-1 candle price reversal", status_strategy_431_reversal.sort_index(ascending=False))
    
    st.write("---")
    
    # Merge on index and selected columns
    status_ema_merged_df = pd.merge(status_strategy_ema, #[selected_columns_df1], 
                                    status_strategy_ema_continual, #[selected_columns_df2], 
                                    left_index=True, right_index=True)
    status_ema_merged_df = pd.merge(status_ema_merged_df, #[selected_columns_df1], 
                                    status_strategy_431_reversal, #[selected_columns_df2], 
                                    left_index=True, right_index=True)

    st.write(status_ema_merged_df.sort_index(ascending=False))
    st.write("---")
    
  return
  

def show_trading_charts():
  # known_options, 
  #                             selected_algos, 
  #                             period, 
  #                             interval,):
  # # show visualizations
  # st.write("hello")
  
  return
  
        
def show_change_logs():
  # generate change log
  st.subheader("Change Log")
  st.write("- Implemented Moving Averages EMA strategy.")
  st.write("- Ability to add more stocks to the existing watchlist from the universe of all stocks allowed by the app.")
  st.write("- Add your own stock tickers through the Customisation tab.")
  st.write("- Added 4-3-1 candle price reversal Strategy.")
  st.write("- News about the selected stocks is listed.")
  
  return

# #############################################

async def algo_trading_summary(symbol,
                               stock_hist_df,
                               selected_algos,
                               selected_period, 
                               selected_interval,
                               ):
    print("algo_trading_summary function is running")
    # st.write(symbol, selected_algos, st.session_state.algo_functions_map)
    
    algo_name, algo_functions = list(st.session_state.algo_functions_map)
    
    # Extract the second element from each list using list comprehension
    extracted_functions = [y for x, y in zip(algo_name, algo_functions) if x in selected_algos]
    # [lst[1] for lst in [algo_name, algo_functions]]
    
    # print("extracted_functions")
    # print(extracted_functions[0], extracted_functions[1])
    
    # results = await asyncio.gather(func_a(), func_b())
    print("getting into functions")
    await asyncio.sleep(1)
    func1 = strategy_sma(symbol,
                 stock_hist_df,
                 selected_period, 
                 selected_interval,
                 algo_strategy = "SMA",
                 selected_short_window = 5,
                 selected_long_window = 8,
                 is_summary = True,
                 )
    func2 = strategy_ema(symbol,
                 stock_hist_df,
                 selected_period, 
                 selected_interval,
                 algo_strategy = "EMA",
                 selected_short_window = 5,
                 selected_long_window = 8,
                 is_summary = True,
                 )
    
    func3 = strategy_ema_continual(symbol,
                                 stock_hist_df,
                                 selected_period, 
                                 selected_interval,
                                 algo_strategy = "EMA 1-2 candle price",
                                 selected_short_window = 5,
                                 selected_long_window = 8,
                                 is_summary = True,
                                 )
    
    func4 = strategy_431_reversal(symbol,
                                 stock_hist_df,
                                 selected_period, 
                                 selected_interval,
                                 is_summary = True,
                                 algo_strategy = "4-3-1 candle price reversal",
                                 
                                 )
    
    # results = await asyncio.gather(extracted_functions[0], extracted_functions[1])
    results = await asyncio.gather(func1, func2, func3, func4)
    await asyncio.sleep(1)
    
    # st.write("algo_trading_summary function is done")
    # st.write(results)
    
    # Combine results into a single list of dictionaries
    combined_results = []
    combined_results_df = pd.DataFrame()
    for result in results:
        combined_results.append(result)
    # print(type(combined_results))
    # st.write(combined_results)

    # Create a DataFrame from the list of dictionaries
    print("generated trading summary for ", symbol)
    # combined_results_df = pd.DataFrame(combined_results)
    # st.write(combined_results_df)
    
    # PLACEHOLDER TO TEST FOR NEW ALGOS
    
    return (combined_results)

    # Get the object allocation traceback
    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics('lineno')

    # # Print the top statistics
    # for stat in top_stats[:10]:
    #     print(stat)