
# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np

import yfinance as yf       #install
import datetime
from datetime import datetime
import time

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
# from streamlit_autorefresh import st_autorefresh
# from schedule import every, repeat, run_pending

import base64
from base64 import b64encode

from millify import millify # shortens values (10_000 ---> 10k)

import streamlit_extras #.metric_cards #import style_metric_cards # beautify metric card with css
# from streamlit_extras.metric_cards import style_metric_cards # beautify metric card with css
# from streamlit_extras import chart_container

# from IPython.core.display import HTML # note the library
# from tabulate import tabulate
# from config import Config

# Using plotly dark template
TEMPLATE = 'plotly_dark'

st.set_page_config(layout='wide', page_title='Stock Dashboard', page_icon=':dollar:')

# update every 5 mins
# st_autorefresh(interval=5 * 60 * 1000, key="dataframerefresh")


# print("Plotly Version : {}".format(plotly.__version__))

pd.set_option('display.max_columns', None,
              'display.max_rows', None,
              'display.max_colwidth', None)

pd.options.display.float_format = '${:,.2f}'.format



# """## stocks"""

def get_all_stock_info(ticker):
  # get all stock info

  info = ticker.info
  info_df = pd.DataFrame.from_dict([info])
  info_df_short = info_df[['symbol', 'shortName', 'exchange', 'quoteType', 'currency',
                           'previousClose', 'open', 'dayLow', 'dayHigh',
                          #  'category', 'navPrice',    # dc, don't know why this is failing?
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

def get_hist_info(ticker, period, interval):
  # get historical market data
  # print(ticker, period, interval)
  hist = ticker.history(period=period, interval=interval)
  # hist['SMA'] = hist['Close'].rolling(20).mean()
  # hist['EMA_5day'] = hist['Close'].ewm(span=5, adjust=False).mean()
  # hist['EMA_10day'] = hist['Close'].ewm(span=10, adjust=False).mean()
  # hist['Signal'] = 0.0

  # #DC review: revisit the rules below
  # # If 5 period ema crosses over 10 period ema (note: ema not sma) then go long

  # hist['Signal'] = np.where(hist['EMA_5day'] > hist['EMA_10day'], 1.0, 0.0)

  # hist['Position'] = hist['Signal'].diff()

  return hist



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

def get_stk_news(ticker):

  news_df = pd.DataFrame(ticker.news)

  # note the new way of creating column
  news_df = news_df.assign(providerPublishTime_n=lambda x: pd.to_datetime(x.providerPublishTime, unit='s'))

  # display(news_df.info())

  news_df_select = news_df[['title',	'publisher',	'link',	'providerPublishTime_n',	'type'	,'relatedTickers']]

  return news_df_select

# https://coderzcolumn.com/tutorials/data-science/candlestick-chart-in-python-mplfinance-plotly-bokeh#2

def draw_candle_stick_chart(df,symbol):
  candlestick = go.Candlestick(
                            x=df.index,       # choosing the Datetime column for the x-axis disrupts the graph to show the gaps
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                            #increasing_line_color= 'green', decreasing_line_color= 'red'
                            )


  fig = go.Figure(data=[candlestick])

  fig.update_layout(
      xaxis_rangeslider_visible=True,
      #width=800, height=600,
      title=symbol,
      yaxis_title= symbol 
  )
  #fig.show()
  return fig

def draw_candle_stick_chart_ma(df, symbol):
  candlestick = go.Candlestick(
                            x=df.index,       # choosing the Datetime column for the x-axis disrupts the graph to show the gaps
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                            #increasing_line_color= 'green', decreasing_line_color= 'red'
                            )
  sma = go.Scatter(x=df.index,
                  y=df["SMA"],
                  #yaxis="y1",
                  name="SMA",
                  # fillcolor = 'black',

                  )
  ema_5day = go.Scatter(x=df.index,
                   y=df["EMA_5day"],
                   name="EMA_5day"
                  )

  ema_10day = go.Scatter(x=df.index,
                   y=df["EMA_10day"],
                   name="EMA_10day"
                  )

  fig = go.Figure(data=[candlestick, sma, ema_5day, ema_10day])

  # fig = go.Figure(data=[candlestick])

  fig.update_layout(
      xaxis_rangeslider_visible=True,
      #width=800, height=600,
      title= symbol,
      yaxis_title= symbol # selected ticker
  )
  # fig.show()
  return fig

def draw_candle_stick_triggers(df, symbol, short_window, long_window, algo_strategy):
  
  # https://plotly.com/python/reference/scattergl/
  # column names for long and short moving average columns
  # print("draw_candle_stick_triggers")
  # print(df.info())
  short_window_col = str(short_window) + '_' + algo_strategy
  long_window_col = str(long_window) + '_' + algo_strategy

  candlestick = go.Candlestick(
                              x=df.index,       # choosing the Datetime column for the x-axis disrupts the graph to show the gaps
                              open=df['Open'],
                              high=df['High'],
                              low=df['Low'],
                              close=df['Close'],
                              #increasing_line_color= 'green', decreasing_line_color= 'red'8
                              )

  short_window = go.Scatter(x=df.index,
                    y=df[short_window_col],
                    name=short_window_col,
                    # fillcolor = 'azure'
                  )

  long_window = go.Scatter(x=df.index,
                    y=df[long_window_col],
                    name=long_window_col
                  )

  # plot 'buy' signals
  N = 100000
  position_buy = go.Scattergl(x=df[df['Position'] == 1].index,
                    y=df[short_window_col][df['Position'] == 1],
                    name="Buy",
                    mode='markers',
                    marker=dict(
                        color=np.random.randn(N),
                        colorscale='greens',
                        line_width=1,
                        symbol = 'triangle-up',
                        size = 10
                        )
                  )

  # plot 'sell' signals
  position_sell = go.Scattergl(x=df[df['Position'] == -1].index,
                            y=df[long_window_col][df['Position'] == -1],
                            name = 'sell',
                            mode='markers',
                            marker=dict(
                              color= np.random.randn(N+1000),
                              colorscale='reds',
                              line_width=1,
                              symbol = 'triangle-down',
                              size = 10
                              # hovertext = df_hist['Open','Close','High','Low']
                              )
                            )

  # # plot ‘buy’ signals
  # plt.plot(df_hist[df_hist['Position'] == 1].index,
  #          df_hist['EMA_p1'][df_hist['Position'] == 1],
  #          '^', markersize = 15, color = 'g', label = 'buy')
  # # plot ‘sell’ signals
  # plt.plot(df_hist[df_hist['Position'] == -1].index,
  #          df_hist['EMA_p2'][df_hist['Position'] == -1],
  #          'v', markersize = 15, color = 'r', label = 'sell')

  # # plot 'buy' signals
  # position_buy = go.Scatter(x=df_hist[df_hist['Position'] == 1].index,
  #          df_hist['EMA_5day'][df_hist['Position'] == 1],
  #          '^', markersize = 15, color = 'c', label = 'buy')

  # # plot 'sell' signals
  # position_sell = go.Scatter(df_hist[df_hist['Position'] == -1].index,
  #          df_hist['EMA_10day'][df_hist['Position'] == -1],
  #          'v', markersize = 15, color = 'k', label = 'sell')

  fig = go.Figure(data=[candlestick, short_window, long_window
                        , position_buy, position_sell
                        ])

  # fig = go.Figure(data=[candlestick])
  
  fig.update_layout(
      xaxis_rangeslider_visible=True,
      #width=800, height=600,
      # title= symbol #  "NVDA, Today - Dec 2023",
      yaxis_title= symbol #'NVDA Stock'
  )
  # fig.show()
  return fig

def sma_trigger_plot(df):
  import plotly.express as px
  df = df.reset_index()
  
  plt.figure(figsize = (20,10))
  # plot close price, short-term and long-term moving averages 
  df['Close'].plot(color = 'k', label= 'Close') 
  df['SMA_p1'].plot(color = 'b',label = 'SMA_p1') 
  df['SMA_p2'].plot(color = 'g', label = 'SMA_p2')
  
  # fig = px.line(df, x="Datetime", y="Close", title='Close')
  
  # st.plotly_chart(fig, theme="streamlit")
  
  # plot ‘buy’ signals
  go_long = df[df['SMA_Position'] == 1].index, df['SMA_p1'][df['SMA_Position'] == 1]
  # print("go_long")
  # print(go_long)
  
  # plot ‘sell’ signals
  go_short = df[df['SMA_Position'] == -1].index, df['SMA_p2'][df['SMA_Position'] == -1]
  # print("go_short")
  # print(go_short)
  
  
  return 

def plot_stk_hist(df):
  # print("plotting Data and Histogram")
  plt.figure(figsize=(12, 5))
  plt.plot(df.Close, color='green')
  plt.plot(df.Open, color='red')
  plt.ylabel("Counts")
  plt.xlabel("Date")

  return


def sparkline(df, col): #, Avg):
    fig = go.Figure()

    # Plot the Close price
    fig.add_trace(go.Scatter(x=df.index, y=df[col], line_color='blue', line_width=1))

    # # Plot the Avg line
    # fig.add_hline(y=ewm, line_color='indianred', line_width=1)

    # hide and lock down axes
    fig.update_xaxes(visible=False) #, fixedrange=True)
    fig.update_yaxes(visible=True, #fixedrange=True,
                     autorange=True,
                     anchor="free",
                     autoshift=True)

    # plt.ylim()

    # strip down the rest of the plot
    fig.update_layout(
        # template=TEMPLATE,
        width=250,
        height=80,
        showlegend=False,
        margin=dict(t=1,l=1,b=1,r=1)
    )

    # disable the modebar for such a small plot - fig.show commented out for debugging purposes
    # fig.show(config=dict(displayModeBar=False))

    png = plotly.io.to_image(fig)

    png_base64 = base64.b64encode(png).decode() #('ascii')
    # png_base64 = pybase64.b64encode(fig.to_image()).decode('utf-8') #'ascii')
    # display(png_base64)
    
    #decode base64 string data
    # decoded_data = base64.b64decode(fig.to_image()).decode('utf-8')
    
    # print(type(png_base64))
    sparkline_url = '<img src="data:image/png;pybase64,{}"/>'.format(png_base64)
    # print(type(sparkline_url))
    # print (sparkline_url)
    
    #open file with base64 string data
    # file = open('file1.txt', 'rb')
    # encoded_data = file.read()
    # file.close()
    #decode base64 string data
    decoded_data=pybase64.b64decode((png_base64))
    #write the decoded data back to original format in  file
    img_file = open('image.jpeg', 'wb')
    img_file.write(decoded_data)
    img_file.close()

    #print ('<img src="data:image/png;base64,{}"/>'.format(png_base64))
    return png_base64 #png_base64 #('<img src="data:image/png;base64,{}"/>'.format(decoded_data))
    # return ('<img src="data:/png;pybase64,{}"/>'.format(png_base64))

def MovingAverageCrossStrategy(symbol, 
                               stock_df,
                               #stock_symbol, 
                               #start_date = '2018-01-01', 
                               #end_date = '2020-01-01',
                               short_window,
                               long_window, 
                               moving_avg, 
                               display_table = True):
    
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

    # import the closing price data of the stock for the aforementioned period of time in Pandas dataframe
    # start = datetime.datetime(*map(int, start_date.split('-')))
    # end = datetime.datetime(*map(int, end_date.split('-'))) 
    # stock_df = web.DataReader(stock_symbol, 'yahoo', start = start, end = end)['Close']
    # stock_df = pd.DataFrame(stock_df) # convert Series object to dataframe 
    # stock_df.columns = {'Close'} # assign new column name
    # stock_df.dropna(axis = 0, inplace = True) # remove any null rows 
                        
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
        stock_df[short_window_col] = stock_df['Close'].ewm(span = short_window, adjust = False).mean()

        # Create a long exponential moving average column
        stock_df[long_window_col] = stock_df['Close'].ewm(span = long_window, adjust = False).mean()

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
    if display_table == True:
        df_pos = stock_df[(stock_df['Position'] == 1) | (stock_df['Position'] == -1)]
        df_pos['Position'] = df_pos['Position'].apply(lambda x: 'Buy' if x == 1 else 'Sell')
        # print(tabulate(df_pos, headers = 'keys', tablefmt = 'psql'))
        # print(df_pos)
    return stock_df, df_pos

def get_current_price(symbol, selected_period, selected_interval):
    ticker = yf.Ticker(symbol)
    todays_data = ticker.history(period = selected_period, interval = selected_interval)
    # print(todays_data)
    
    return todays_data['Close'].iloc[-1]

def show_snapshot(all_tickers):
    
    fig = px.line(df, x="date", y=df.columns,
              hover_data={"date": "|%B %d, %Y"},
              title='custom tick labels with ticklabelmode="period"')
    fig.update_xaxes(
        dtick="M1",
        tickformat="%b\n%Y",
        ticklabelmode="period")
    fig.show()  #st.pyplot(plt.gcf())
    st.pyplot(fig)
    return