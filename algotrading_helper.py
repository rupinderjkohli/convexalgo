
# ##########################################################  
# Purpose: 
# ##########################################################
import pandas as pd
import numpy as np
from IPython.display import display

import yfinance as yf       #install
import datetime
from datetime import datetime
import time
import pytz
from millify import millify # shortens values (10_000 ---> 10k)


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

from algotrading_algos import *

nest_asyncio.apply()

# print("Plotly Version : {}".format(plotly.__version__))

pd.set_option('display.max_columns', None,
              'display.max_rows', None,
              'display.max_colwidth', None)

pd.options.display.float_format = '${:,.2f}'.format

# import redis
# ssl_connection = redis.Redis(host='localhost', port=6666, ssl=True, ssl_cert_reqs="none")
# ssl_connection.ping()

def setup_default_state():
  print(">>>>>>>>>>>> setup_session_state\n")
  
  period = period[0]
  interval = interval[0]
  stop_loss_factor = float(stop_loss[0])
  take_profit_factor = float(take_profit[0])
  
  moving_average = trading_strategy_ma
  trend_based = trading_strategy_trend 

  selected_algos = convex_trade_algos_list
  
  return

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
  
  if refresh:
    return SYMBOLS
  else:
    return SYMBOLS, PERIOD, INTERVAL, STOP_LOSS, TAKE_PROFIT, MOVING_AVERAGE_BASED, TREND_BASED


# https://coderzcolumn.com/tutorials/data-science/candlestick-chart-in-python-mplfinance-plotly-bokeh#2

# ##########################################################  
# Purpose: Get user specific tickers list
# ##########################################################
def load_user_selected_options(username):
  user_list = []
  try :
    user_options = "user_selected_options.csv"
    df = pd.read_csv(user_options, header=0)
    
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

# ##########################################################  
# Purpose: set up the stock ticker watchlist (user customisation)
# ##########################################################
def save_user_selected_options(username, selected_tickers):
  df_tickers = pd.DataFrame(selected_tickers)
  try:
    user_options = "user_selected_options.csv"
    df_tickers.to_csv(user_options, mode='w', index=False, header=True)
  except pd.errors.EmptyDataError:
    print('CSV file is empty save')
  except FileNotFoundError:
    print('CSV file not found save')
  return
  
# ##########################################################  
# Purpose: reload the user watchlist
# ##########################################################
def display_watchlist():
  user_sel_list = []
  user_sel_list = load_user_selected_options("demo")
  
  return user_sel_list
  

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
  # print (info_df_short.to_dict(orient='dict'))
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
# Purpose: Retrieve the ticker news
# ##########################################################
def get_stk_news(ticker):

  try:
    news_df = pd.DataFrame(ticker.news)
  except:
    # st.write("ERROR")
    return pd.DataFrame()

  # note the new way of creating column
  news_df = news_df.assign(providerPublishTime_n=lambda x: pd.to_datetime(x.providerPublishTime, unit='s'))

  # display(news_df.info())

  news_df_select = news_df[['title',	'publisher',	'link',	'providerPublishTime_n',	'type'	,'relatedTickers']]

  return news_df_select

# ##########################################################  
# Purpose: Retrive the ticker signals
# ##########################################################
def signals_view(known_options, selected_algos, selected_period, selected_interval):
  # generate summary
  print("####### signals_view run time #######")
  df_summary_view = pd.DataFrame()
  
  combined_trading_summary = []
  combined_trading_summary_df = pd.DataFrame()
  # await asyncio.sleep(1)
  tasks = []
  
  if (len(selected_algos) == 0):
    #RK load from config
    selected_algos = ['5/8 EMA', '5/8 EMA 1-2 candle price','4-3-1 candle price reversal'] 
    
  # RK051424: getting stock history from a central function in 2 steps - 
  # load the history for all tickers and 
  # then process for individual ticker
  yf_ticker_history = get_selected_stock_history(known_options,selected_period, selected_interval)    
  
  for symbol in known_options:
    # get ticker data
    # yf_data = yf.Ticker(symbol) #initiate the ticker
    
    
    # st.session_state.page_subheader = '{0} ({1})'.format(yf_data.info['shortName'], yf_data.info['symbol'])
    # st.subheader(st.session_state.page_subheader)
    
    # # generate summary
    # etf_summary_info = get_all_stock_info(yf_data)
    # # st.write(etf_summary_info.T)
    # df_details = etf_summary_info[['symbol', 'shortName','quoteType','financialCurrency',
    #                                 'industry','sector','currentPrice','recommendationKey', 
    #                                 'fiftyTwoWeekHigh','fiftyTwoWeekLow', 
    #                                 'grossMargins','ebitdaMargins']]
    # df_details['52w Range'] = ((df_details['currentPrice'] - df_details['fiftyTwoWeekLow'])/(df_details['fiftyTwoWeekHigh'] - df_details['fiftyTwoWeekLow']))*100
    
    # df_summary_view = pd.concat([df_summary_view, df_details], ignore_index=True)
    # Add 52 week price range
    
    # generate trading summary
    # based on the selected algo strategy call the selected functions
    # st.write("getting summary for: ", symbol)
    # stock_hist_df = get_hist_info(yf_data, selected_period, selected_interval)
    # st.write(stock_hist_df[:10])
    # await asyncio.sleep(1)
    # use gather instead of run
    
    # RK051424: getting stock history from a central function in 2 steps -
    stock_hist_df = yf_ticker_history[symbol]
    
    # tasks.append(algo_trading_summary(symbol, 
    #                                  stock_hist_df,
    #                                  selected_algos, 
    #                                  selected_period, 
    #                                  selected_interval,
    #                                  )
    #              )
    # print("algo_trading_summary", symbol)
    results = algo_trading_summary(symbol, 
                                     stock_hist_df,
                                     selected_algos, 
                                     selected_period, 
                                     selected_interval,
                                     )
    # print ("results algo_trading_summary >>>>>>>>>>>>\n", results)
    # print ("results algo_trading_summary >>>>>>>>>>>>\n")
    combined_trading_summary.append(results)
    
                 
  # results = await asyncio.gather(*tasks)
  # await asyncio.sleep(1)

  # Flatten the list nested structure
  flattened_data = [item for sublist in combined_trading_summary for item in sublist]
  
  print("##################################### RESULTS #################### \n", combined_trading_summary)
  
  print("##################################### flattened_data #################### ", flattened_data)
  # Create a DataFrame from the list of dictionaries
  combined_trading_summary_df = pd.DataFrame(combined_trading_summary)
  # print(combined_trading_summary_df.columns)
  # Index(['symbol', 'algo_strategy', 'stock_trigger_at', 'stock_trigger_state',
  #      'stock_price_at_trigger', 'stock_stop_loss_atr',
  #      'stock_take_profit_atr', 'stock_atr_ma', 'stock_ema_p1', 'stock_ema_p2',
  #      'stock_previous_triggers'],
  #     dtype='object')
  # combined_trading_summary_df = combined_trading_summary_df[['symbol', 
  #                                                            'stock_trigger_state',
  #                                                            'stock_trigger_at', 
  #                                                            'stock_price_at_trigger', 
  #                                                            'stock_stop_loss_atr',
  #                                                            'stock_take_profit_atr',
  #                                                            'algo_strategy', 
  #                                                            'tweet_post',
  #                                                            'stock_previous_triggers',
  #                                                            ]].sort_values(by = ['stock_trigger_at', 'symbol'], ascending=[False, True])
  
  display("............. combined_trading_summary_df .............\n", combined_trading_summary_df)
  # # Add a column with links
  # link_column = []
  # for symbol in combined_trading_summary_df['symbol']:
  #   link = f'https://example.com/{symbol}'
  #   link_column.append(f'[Link]({link})')
  # # Add the link column to the DataFrame
  # combined_trading_summary_df['share'] = link_column
  
  # # Create a column for popover toggles
  # toggle_col = st.columns(combined_trading_summary_df.shape[0]) # Create a column for each row 
  # print("toggle_col#########################", toggle_col)
  
  # for i, col in enumerate(combined_trading_summary_df.shape[0]):
  #   if col.popover('ℹ️ {i}'): #('ℹ️', key=i):
  #       col.write(combined_trading_summary_df.iloc[i]['symbol'])
  
  # st.data_editor(
  #   combined_trading_summary_df,
  #   column_config={"symbol": st.column_config.TextColumn(
  #       "Ticker",
  #       width="small"
  #   ),
  #                  "algo_strategy": st.column_config.TextColumn(
  #       "Strategy Name",
  #       width="small"
  #   ),
  #                  "stock_trigger_state": st.column_config.TextColumn(
  #       "Trigger",
  #       width="small"
  #   ),
  #                   "stock_take_profit_atr": st.column_config.NumberColumn(
  #       "Take Profit Price",
  #       format="%.2f",
  #   ),
  #                   "stock_stop_loss_atr": st.column_config.NumberColumn(
  #       "Stop Loss Price",
  #       format="%.2f",
  #   ),
  #                   "stock_price_at_trigger": st.column_config.NumberColumn(
  #       "Trigger Price",
  #       format="%.2f",
  #   ),
  #                   "stock_atr_ma": st.column_config.NumberColumn(
  #       "ATR MA",
  #       format="%.2f",
  #   ),
  #                   "stock_previous_triggers": st.column_config.ListColumn(
  #       "Previous Triggers",
  #       # format="DD MMM YYYY, HH:MM"
  #       # width="medium",
  #   ),
  #                   "stock_trigger_at": st.column_config.DatetimeColumn(
  #       "Trigger Time",
  #       format = "YYYY-MM-DD HH:mm"
  #       # format="DD MMM YYYY, HH:MM"
  #   ),
  #       # "share": st.column_config.LinkColumn
  #       # (
  #       #     "share",
  #       #     help="social media",
  #       #     max_chars=100,
  #       #     # display_text="view table",
  #       #     # default=add_container(etf_data[symbol], quick_explore_df[symbol])
  #       # ),
        
  #   },
  #   height=None,
  #   use_container_width=True,
  #   hide_index=True,
  #   )
  # app_refresh(selected_interval, "signals_view")
  return None
  
# save output in cache to be used by the trading charts
# @st.cache_resource
# @st.experimental_fragment(run_every='1m')
def stock_status(known_options, selected_algos, selected_period, selected_interval, run_count):
  # generate stocks list view
  # st.write(known_options, selected_algos, selected_period, selected_interval)
  run_count +=1
  # await asyncio.sleep(1)
  stock_status_data = {}
  status_ema_merged_df = {}
  etf_multi_index = pd.MultiIndex.from_product([known_options,
                                                selected_algos],
                                               names=['tickers', 'algo_strategy'])
  # Create a DataFrame with the MultiIndex
  etf_processed_signals = pd.DataFrame(index=etf_multi_index,
                                       columns = ['Value']
                                       )
  # st.write("from stock_status",etf_processed_signals)
  # RK 051424
  yf_ticker_history = get_selected_stock_history(known_options,selected_period, selected_interval) 
  
  for symbol in known_options:
    # get ticker data
    
    # yf_data = yf.Ticker(symbol) #initiate the ticker
    # st.write("fetching status for: ", symbol )
    # stock_hist_df = get_hist_info(yf_data, selected_period, selected_interval)
    # RK051424: getting stock history from a central function in 2 steps -
    stock_hist_df = yf_ticker_history[symbol]
    
    func1 = strategy_sma(symbol,
                 stock_hist_df,
                 selected_period, 
                 selected_interval,
                 algo_strategy = "SMA",
                 selected_short_window = 5,
                 selected_long_window = 8,
                 is_summary = False,
                 )
    # st.write(func1)
    
    status_strategy_ema = strategy_ema(symbol,
                 stock_hist_df,
                 selected_period, 
                 selected_interval,
                 algo_strategy = "EMA",
                 selected_short_window = 5,
                 selected_long_window = 8,
                 is_summary = False,
                 )
    algo_strategy = "5/8 EMA"
    
    print("status_strategy_ema")
    # print(status_strategy_ema.columns)
    print()
    
    status_strategy_ema = status_strategy_ema [[ 
       'Close','5_EMA', '8_EMA', 'Position_c',]]
    # st.write("EMA", status_strategy_ema.sort_index(ascending=False))
    # Rename the column Position_c
    status_strategy_ema.rename(columns={'Position_c': 'Trigger:EMA crossover',
                                        'Close': 'EMA Close Price'}, inplace=True)
    
    status_strategy_ema_continual = strategy_ema_continual(symbol,
                                 stock_hist_df,
                                 selected_period, 
                                 selected_interval,
                                 algo_strategy = "5/8 EMA 1-2 candle price",
                                 selected_short_window = 5,
                                 selected_long_window = 8,
                                 is_summary = False,
                                 )
    algo_strategy = "5/8 EMA 1-2 candle price"
    # etf_processed_signals[symbol][algo_strategy] = status_strategy_ema
    # st.write(etf_processed_signals) #[symbol][:10])
    
    # ema_continual + the following columns
    # stock_df['ema_5above8'];stock_df['t0_close_aboveema5'];stock_df['t0_low_belowema5'];stock_df['ema_continual_long'];
    # stock_df['ema_5below8'];stock_df['t0_close_belowema5'];stock_df['t0_low_aboveema5'];stock_df['ema_continual_short']
    print("status_strategy_ema_continual")
    print(status_strategy_ema_continual.columns)
    print()
    # Index(['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits',
      #  '5_EMA', '8_EMA', 'Signal', 'Position', 'atr', 'atr_ma', 'position_c',
      #  'stop_loss_atr', 'take_profit_atr', '5_EMA 1-2 candle price',
      #  '8_EMA 1-2 candle price', 'ema_5above8', 't0_close_aboveema5',
      #  't0_low_belowema5', 'ema_continual_long', 'ema_5below8',
      #  't0_close_belowema5', 't0_low_aboveema5', 'ema_continual_short'],
    #   dtype='object')
    status_strategy_ema_continual = status_strategy_ema_continual[[ 
       'Close','5_5/8 EMA 1-2 candle price','8_5/8 EMA 1-2 candle price',  
       'ema_5above8','t0_close_aboveema5','t0_low_belowema5', 'ema_continual_long', 
       'ema_5below8','t0_close_belowema5', 't0_low_aboveema5', 'ema_continual_short','position']]
    # st.write("EMA 1-2 candle price", status_strategy_ema_continual.sort_index(ascending=False))
    # Rename the column position
    status_strategy_ema_continual.rename(columns={'position': 'Trigger:EMA 1-2 Continual',
                                        'Close': 'EMA_C Close Price'}, inplace=True)
    
    # status_strategy_431_reversal = strategy_431_reversal(symbol,
    #                              stock_hist_df,
    #                              selected_period, 
    #                              selected_interval,
    #                              is_summary = False,
    #                              algo_strategy = "4-3-1 candle price reversal",
    #                              )
    # algo_strategy = "4-3-1 candle price reversal"
  
    # print("status_strategy_431_reversal")
    # # print(status_strategy_431_reversal.columns)
    # print()
    # # ['Open', 'Close', 'High', 'Low', 't3', 't2', 't1', 't0',
    #   #  'strategy_431_long_c1', 'strategy_431_long_c2', 'strategy_431_long_c3',
    #   #  'strategy_431_long', 'strategy_431_short_c1', 'strategy_431_short_c2',
    #   #  'strategy_431_short_c3', 'strategy_431_short', 'position', 'atr',
    #   #  'atr_ma', 'stop_loss_atr', 'take_profit_atr']
    # #   dtype='object')
    # status_strategy_431_reversal = status_strategy_431_reversal[['Close','strategy_431_long_c1',
    #    'strategy_431_long_c2', 'strategy_431_long_c3', 'strategy_431_long',
    #    'strategy_431_short_c1', 'strategy_431_short_c2',
    #    'strategy_431_short_c3', 'strategy_431_short','position',]]
    # # st.write("4-3-1 candle price reversal", status_strategy_431_reversal.sort_index(ascending=False))
    # # Rename the column position
    # status_strategy_431_reversal.rename(columns={'position': 'Trigger:4-3-1 Reversal',
    #                                     'Close': '4-3-1 Close Price'}, inplace=True)
    
    
    # status_strategy_candle_hammer =  await strategy_candle_hammer(symbol,
    #                              stock_hist_df,
    #                              selected_period, 
    #                              selected_interval,
    #                              is_summary = False,
    #                              algo_strategy = "candle hammer",
    #                              )
    # algo_strategy = "candle hammer"
    # st.write("strategy_candle_inverted_hammer:stock_status")
    
    # strategy_candle_inverted_hammer = await strategy_candle_inverted_hammer(symbol,
    #                              stock_hist_df,
    #                              selected_period, 
    #                              selected_interval,
    #                              is_summary = False,
    #                              algo_strategy = "candle inverted hammer",)
    
    # algo_strategy = "candle inverted hammer"
    # st.write("---")
    # st.write("strategy_candle_inverted_hammer:stock_status")
    # Merge on index and selected columns
    status_ema_merged_df = pd.DataFrame()
    status_ema_merged_df = pd.merge(status_strategy_ema, #[selected_columns_df1], 
                                    status_strategy_ema_continual, #[selected_columns_df2], 
                                    left_index=True, right_index=True, how='outer')
    # status_ema_merged_df = pd.merge(status_ema_merged_df, #[selected_columns_df1], 
    #                                 status_strategy_431_reversal, #[selected_columns_df2], 
    #                                 left_index=True, right_index=True, how='outer')
    # status_ema_merged_df = pd.merge(status_ema_merged_df, #[selected_columns_df1], 
    #                                 status_strategy_candle_hammer, #[selected_columns_df2], 
    #                                 left_index=True, right_index=True, how='outer')
    # status_ema_merged_df = pd.merge(status_ema_merged_df, #[selected_columns_df1], 
    #                                 strategy_candle_inverted_hammer, #[selected_columns_df2], 
    #                                 left_index=True, right_index=True, how='outer')

    stock_status_data[symbol] = status_ema_merged_df
    # print('#################################')
    # print(status_ema_merged_df.columns)
    # print('#################################')
    
    # data = {}
    # for idx in etf_processed_signals.index:
    #   # st.write(idx)
    #   data[idx] = status_ema_merged_df 
    #   # Reassign names if they are missing or incorrect
      
      
    # # Concatenate the DataFrames along axis=0 to create a DataFrame with MultiIndex
    # etf_processed_signals_df = pd.concat(data, axis=0)
    # # st.write("etf_processed_signals_df.index",etf_processed_signals_df.index)
    # etf_processed_signals_df.index.names = ['ticker', 'algo_name', 'Datetime']
    # # st.write("etf_processed_signals_df.index",etf_processed_signals_df.index)
    
    # # st.write("from stock_status function")
    # # st.write(etf_processed_signals_df[symbol]) #[:10])
  # app_refresh(selected_interval, "stock_status")
  return stock_status_data, status_ema_merged_df #etf_processed_signals_df, stock_status_data
        
def show_change_logs():
  # generate change log
  st.subheader("Change Log")
  st.write("- Implemented Moving Averages EMA strategy.")
  st.write("- Ability to add more stocks to the existing watchlist from the universe of all stocks allowed by the app.")
  st.write("- Add your own stock tickers through the Customisation tab.")
  st.write("- Added 4-3-1 candle price reversal Strategy.")
  # st.write("- News about the selected stocks is listed.")
  st.write("- Added visualizations with filter on date and stock.")
  
  return

def algo_playground():
  # st.session_state.user_watchlist, # known_options, 
  #                             st.session_state.selected_algos, 
  #                             st.session_state.period, 
  #                             st.session_state.interval
  df_hist = get_selected_stock_history(st.session_state.user_watchlist,st.session_state.period, 
                                  st.session_state.interval)
  
  strategy_hammer_summary_tasks = []
  strategy_hammer_details_tasks = []
  
  for symbol in st.session_state.user_watchlist:
    print(df_hist[symbol].columns)
    df = df_hist[symbol]
    
    # # st.write("################")
    # st.write(candle_hammer(df))
    # # st.write("################")
    
    strategy_hammer_summary_tasks.append(strategy_candle_hammer(symbol,
                                  df,
                                  st.session_state.period, 
                                  st.session_state.interval,
                                  is_summary = True,
                                  algo_strategy = "candle hammer",)
                                     )

    
    # strategy_candle_hammer_summary = asyncio.run(strategy_candle_hammer(symbol,
    #                               df,
    #                               st.session_state.period, 
    #                               st.session_state.interval,
    #                               is_summary = True,
    #                               algo_strategy = "candle hammer",))
    
    # pd.DataFrame(strategy_candle_hammer_summary)
    # st.write(strategy_candle_hammer_summary)
    
    # st.write("strategy_candle_hammer - details",)
    # strategy_hammer_details_tasks.append(strategy_candle_hammer(symbol,
    #                               df,
    #                               st.session_state.period, 
    #                               st.session_state.interval,
    #                               is_summary = False,
    #                               algo_strategy = "candle hammer",)
    #                                  )
    st.write("strategy_candle_hammer - details",symbol)
    # results_strategy_hammer_details = await asyncio.gather(*strategy_hammer_details_tasks)
    # st.write(type(results_strategy_hammer_details))
    strategy_candle_hammer_detailed = (strategy_candle_hammer(symbol,
                                  df,
                                  st.session_state.period, 
                                  st.session_state.interval,
                                  is_summary = False,
                                  algo_strategy = "candle hammer",))
    
    # await asyncio.sleep(1)
    st.write(strategy_candle_hammer_detailed)
    
    # strategy_hammer(df)
  
  st.write("strategy_candle_hammer - summary")
  # results_strategy_hammer_summary = await asyncio.gather(*strategy_hammer_summary_tasks)
  # st.write(pd.DataFrame(results_strategy_hammer_summary))
  
  return
  
# #############################################

def algo_trading_summary(symbol,
                               stock_hist_df,
                               selected_algos,
                               selected_period, 
                               selected_interval,
                               ):
    print("********* algo_trading_summary **********", symbol)
    trading_summary_results = []
    
    # st.write(symbol, selected_algos)
    
    # NameError: name 'algo_functions_map' is not defined
    # algo_name, algo_functions, algo_functions_args = list(algo_functions_map)
    
    # # Extract the second element from each list using list comprehension
    # #RK load from config
    # extracted_functions = [y for x, y in zip(algo_name, algo_functions) if x in selected_algos]
    # # [lst[1] for lst in [algo_name, algo_functions]]
    
    # print("extracted_functions")
    # print(extracted_functions[0], extracted_functions[1])
    
    # results = await asyncio.gather(func_a(), func_b())
    # print("getting into functions")
    # await asyncio.sleep(1)
    # func1 =  strategy_sma(symbol,
    #              stock_hist_df,
    #              selected_period, 
    #              selected_interval,
    #              algo_strategy = "SMA",
    #              selected_short_window = 5,
    #              selected_long_window = 8,
    #              is_summary = True,
    #              )
    # st.write("func1", func1)
    
    func2 =  strategy_ema(symbol,
                 stock_hist_df,
                 selected_period, 
                 selected_interval,
                 algo_strategy = "EMA",
                 selected_short_window = 5,
                 selected_long_window = 8,
                 is_summary = True,
                 )
    # print("func2 >>>>>>>>>      >>>>>>>>>>>>  ", func2)
    # Combine results into a single list of dictionaries
    trading_summary_results.append(func2)
    
    func3 =  strategy_ema_continual(symbol,
                                 stock_hist_df,
                                 selected_period, 
                                 selected_interval,
                                 algo_strategy = "5/8 EMA 1-2 candle price",
                                 selected_short_window = 5,
                                 selected_long_window = 8,
                                 is_summary = True,
                                 )
    # print("func3 >>>>>>>>>      >>>>>>>>>>>>  ", func3)
    # Combine results into a single list of dictionaries
    trading_summary_results.append(func3)

    # func4 =  strategy_431_reversal(symbol,
    #                              stock_hist_df,
    #                              selected_period, 
    #                              selected_interval,
    #                              is_summary = True,
    #                              algo_strategy = "4-3-1 candle price reversal",
    #                              )
    
    # print("func4 >>>>>>>>>      >>>>>>>>>>>>  ", func4)
    # # Combine results into a single list of dictionaries
    # trading_summary_results.append(func4)

    # func5 =  strategy_candle_hammer(symbol,
    #                              stock_hist_df,
    #                              selected_period, 
    #                              selected_interval,
    #                              is_summary = True,
    #                              algo_strategy = "candle hammer",
    #                              )
    
    # func6 = strategy_candle_inverted_hammer(symbol,
    #                              stock_hist_df,
    #                              selected_period, 
    #                              selected_interval,
    #                              is_summary = True,
    #                              algo_strategy = "candle inverted hammer",)
    # st.write("func4", func4)
    
    
    # results = await asyncio.gather(func2) #, func3, func4)
    # st.write("results", results)

    # results = func2
    
    # await asyncio.sleep(1)
    
    # print("algo_trading_summary function is done")
    # print(results)
    
    # Combine results into a single list of dictionaries
    # trading_summary_results = []
    # trading_summary_results.append(results)
    
    # trading_summary_results_df = pd.DataFrame()
    # for result in results:
    #   print("trading_summary_results ================\n", result)
    #   trading_summary_results.append(result)
    # print("trading_summary_results ================\n",trading_summary_results)
    # print("trading_summary_results ================\n")
    # # st.write(trading_summary_results)

    # Create a DataFrame from the list of dictionaries
    print(" ================ generated trading summary for ================>>>", symbol, "\n", trading_summary_results)
    # trading_summary_results_df = pd.DataFrame(trading_summary_results)
    # st.write(trading_summary_results_df)
    
    # PLACEHOLDER TO TEST FOR NEW ALGOS
  
    return (trading_summary_results)

    # Get the object allocation traceback
    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics('lineno')

    # Print the top statistics
    # for stat in top_stats[:10]:
    #     print(stat)

    
def to_twitter(post):
  # Twitter API credentials
  consumer_key = "WGZXajFpNExQRVdDazBIX2Fiblc6MTpjaQ" #'your_consumer_key'
  consumer_secret = "JFc7PYNVn6HZY4HyIXGphXIPm1PyRhmyD5sBCWLIJA3RDTti7p" #'your_consumer_secret'
  access_token = "1342321856523726848-jd8sBpTd4l0h9SSzC1kIWUFXb6Nj6D" #'your_access_token'
  access_token_secret = "kbLUtKW4QjWyteK7vOXDn4NqowPbVcfk5qLzsx1879JnJ" #'your_access_token_secret'
  api_key = "l3JyizUiL2A4l45N0bPgloDv2"
  api_secret = "yluErKd8PSwPhLGFRCB3R3kkbpOJZ8YaERJn4JJyu9inyn1DLO"

  # Authenticate to Twitter
  auth = tweepy.OAuth1UserHandler(#consumer_key, consumer_secret, 
                                  api_key, api_secret, 
                                  access_token, access_token_secret)
  api = tweepy.API(auth)

  # Streamlit app
  st.sidebar.title("")
  st.sidebar.subheader('Tweet from ConvexTrades!')

  # Get tweet text from user input
  tweet_text = st.sidebar.text_area('Enter your tweet:', '')

  # Button to send tweet
  if st.sidebar.button('Send Tweet'):
      # Check if tweet text is not empty
      if tweet_text.strip():
          # Send tweet
          api.update_status(tweet_text)
          st.sidebar.success('Tweet sent successfully!')
      else:
          st.sidebar.warning('Please enter some text for your tweet.')


def get_selected_stock_history(known_options,selected_period, selected_interval):
  selected_etf_data = {}
  for symbol in known_options:
    # get ticker data
    yf_data = yf.Ticker(symbol) #initiate the ticker
    # st.write("fetching status for: ", symbol )
    stock_hist_df = get_hist_info(yf_data, selected_period, selected_interval)  
    selected_etf_data[symbol] = stock_hist_df
    
     
  return selected_etf_data

def app_refresh(selected_interval, process):
   # streamlit autorefresh
  st.write("app_refresh function", st.session_state['last_run'],  datetime.now()) 
  stock_history_refresh_cnt = st_autorefresh(interval=selected_interval, limit=100, key="stock_history_refresh") 
  last_update = datetime.now()
  st.session_state['last_run'] = int(time.time())
  # last_refresh_log = last_refresh_log.append(st.session_state['last_run'])
  user_refresh_log = st.session_state.username + "_last_refresh_log.csv"
  with open(user_refresh_log, 'a+') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([process, st.session_state['last_run']])
        
  # try:
  #   user_refresh_log = st.session_state.username + "_last_refresh_log.csv"
  #   st.session_state['last_run'].to_csv(user_refresh_log, mode='w', index=False, header=True)
  # except pd.errors.EmptyDataError:
  #   print('CSV file is empty save')
  # except FileNotFoundError:
  #   st.session_state['last_run'].to_csv(user_refresh_log, index=False)
  
  # app_refresh 1716571443 2024-05-24 13:26:31.803263
  return last_update

# def extract_number(time_string):
#   number = ''
#   for char in time_string:
#       if char.isdigit():
#           number += char
#   return (number)
