# -*- coding: utf-8 -*-
"""AlgoTrading_v3.2_panel.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18MuK4_G2Nf8oow21NW_a3pHg_35JgVSI
"""

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


# from IPython.core.display import HTML # note the library

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
                          #  'fiftyTwoWeekLow', 'fiftyTwoWeekHigh', 'fiftyDayAverage',
                          #  'regularMarketVolume',
                          #  'twoHundredDayAverage',
                          #  'trailingPE', 'volume',
                          #  'averageVolume', 'averageVolume10days',
                          #  'averageDailyVolume10Day', 'bid', 'ask', 'bidSize', 'askSize', 'yield',
                          #  'totalAssets', 'trailingAnnualDividendRate',
                          #  'trailingAnnualDividendYield',
                          #  'ytdReturn', 'beta3Year', 'fundFamily', 'fundInceptionDate',
                          #  'legalType', 'threeYearAverageReturn', 'fiveYearAverageReturn',
                          # 'underlyingSymbol',
                          #  'longName', 'firstTradeDateEpochUtc', 'timeZoneFullName',
                          #  'timeZoneShortName', 'uuid', 'messageBoardId', 'gmtOffSetMilliseconds',
                          #  'trailingPegRatio'
                            ]]
  info_df_short.reset_index(inplace=True)
  # st.write (info_df_short.to_dict(orient='dict'))
  return info_df_short

def get_hist_info(ticker, period, interval):
  # get historical market data
  hist = ticker.history(period=period, interval=interval)
  hist['SMA'] = hist['Close'].rolling(20).mean()
  hist['EMA_5day'] = hist['Close'].ewm(span=5, adjust=False).mean()
  hist['EMA_10day'] = hist['Close'].ewm(span=10, adjust=False).mean()
  hist['Signal'] = 0.0

  # If 5 period ema crosses over 10 period ema (note: ema not sma) then go long

  hist['Signal'] = np.where(hist['EMA_5day'] > hist['EMA_10day'], 1.0, 0.0)

  hist['Position'] = hist['Signal'].diff()

  return hist

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

def draw_candle_stick_triggers(df, symbol):
  
  # https://plotly.com/python/reference/scattergl/

  candlestick = go.Candlestick(
                              x=df.index,       # choosing the Datetime column for the x-axis disrupts the graph to show the gaps
                              open=df['Open'],
                              high=df['High'],
                              low=df['Low'],
                              close=df['Close'],
                              #increasing_line_color= 'green', decreasing_line_color= 'red'8
                              )

  ema_5day = go.Scatter(x=df.index,
                    y=df["EMA_5day"],
                    name="EMA_5day",
                    # fillcolor = 'azure'
                  )

  ema_10day = go.Scatter(x=df.index,
                    y=df["EMA_10day"],
                    name="EMA_10day"
                  )

  # plot 'buy' signals
  N = 100000
  position_buy = go.Scattergl(x=df[df['Position'] == 1].index,
                    y=df['EMA_5day'][df['Position'] == 1],
                    name="Buy",
                    mode='markers',
                    marker=dict(
                        color=np.random.randn(N),
                        colorscale='greens',
                        line_width=1,
                        symbol = 'triangle-up',
                        size = 12
                        )
                  )

  # plot 'sell' signals
  position_sell = go.Scattergl(x=df[df['Position'] == -1].index,
                            y=df['EMA_10day'][df['Position'] == -1],
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

  fig = go.Figure(data=[candlestick, ema_5day, ema_10day
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


def plot_stk_hist(df):
  print("plotting Data and Histogram")
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


def main():
      
  # """### Select Stock and Time interval"""
  # https://github.com/smudali/stocks-analysis/blob/main/dasboard/01Home.py
  symbol_list = ["TSLA","NVDA","AMZN", "NFLX","BA","GS","SPY","QQQ","IWM","SMH","RSP"]

  st.sidebar.header("Choose your Stock filter: ")
  # ticker = st.sidebar.selectbox(
  #     'Select Ticker', options=symbol_list)
  
  #implement multi-selection for tickers
  # st.write(st.__version__)
  
  ticker = st.sidebar.multiselect('Choose Ticker', options=symbol_list,
                                help = 'Select a ticker', 
                                key='ticker',
                                max_selections=4,
                                default= ["TSLA"]
                                )
  selected_period = st.sidebar.selectbox(
      'Select Period', options=['1d','5d','1mo','3mo', '6mo', 'YTD', '1y', 'all'], index=2)
  selected_interval = st.sidebar.selectbox(
      'Select Intervals', options=['1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo'], index=8)
      

  #         Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
  # Either Use period parameter or use start and end
  #     interval : str
  #         Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo

  # period = "1mo"
  # interval= "1d"
  ema_period1 = 5
  ema_period2 = 10

  known_options = ticker 
  # st.write (known_options)
  
  if len(known_options) == 0:
    st.write ("Please select a ticker in the sidebar")
    return
  else:
    tab = st.tabs(["Summary","🗃 List View","📈 Visualisations", "🗃 Details"])
    # ###################################################
    # Summary: 
    # # of stocks being watched; 
    # Algo being used
    # # of winining vs losing trades
    # # best stocks
    # ###################################################
    with tab[0]:    
      st.write("Showing Summary for the following stocks:")
      # st.write(selected_period, selected_interval)
      
      st.write(known_options)
      etf_summary = {} #pd.DataFrame()
      etf_summary_info = {} # dictionary
      
      for symbol in known_options:
        yf_data = yf.Ticker(symbol) #initiate the ticker
        etf_summary_info[symbol] = get_all_stock_info(yf_data) #get_hist_info(yf_data, selected_period, selected_interval)
        # etf_summary = pd.concat([etf_summary, etf_summary_info])
        # etf_summary = pd.DataFrame(etf_summary_info.items())
        
        etf_summary.update(etf_summary_info)
        # etf_summary = pd.Series(etf_summary_info).to_frame()
        
        # print(type(etf_summary_info))
        print("df:", etf_summary)
        
      # etf_summary_df = pd.DataFrame.from_dict([etf_summary_info])
      
      st.write(etf_summary['TSLA']) #.to_html(escape=False, index=False), unsafe_allow_html=True) 
      st.divider()
      st.write(etf_summary['TSLA'])
      
      df = pd.json_normalize(etf_summary)
      
      st.write(df)
       
        # # Subheader with company name and symbol
        # st.session_state.page_subheader = '{0} ({1})'.format(yf_data.info['shortName'], yf_data.info['symbol'])
        # st.subheader(st.session_state.page_subheader)
        # st.write(symbol)
        # stock_info_df = get_all_stock_info(yf_data)
        # st.write(stock_info_df.to_html(escape=False, index=False), unsafe_allow_html=True)

    
    # ###################################################
    # List View: 
    # # of all stocks; 
    # ###################################################
    with tab[1]:
      st.write("Showing the List View of the selected stocks")
      ## Create two columns
      # col1, col2 = st.columns(2)
      # etf_info = pd.DataFrame()
      # etf_data = {} # dictionary
        
      # for symbol in known_options:
      #   yf_data = yf.Ticker(symbol) #initiate the ticker
      #   ticker = yf.Ticker(symbol)
      #   # df = get_all_stock_info(ticker)

      #   etf_data[symbol] = get_hist_info(yf_data, selected_period, selected_interval)
      #   etf_info = pd.concat([etf_data, etf_info], ignore_index=True)

      #   # History data for 12 months
      #   history = yf_data.history(period=selected_period)[['Open', 'Close']]
      #   # Convert the history series to a DF
      #   history_df = history #.to_frame()
      #   # display (ticker)
      #   # display (history_df.head(5))

      #   # Add the sparkline for 12 month Open history data
      #   spark_img = sparkline(history_df, 'Open')
      #   spark_img_url =  ('<img src="data:/png;pybase64,{}"/>'.format(spark_img))
      #   # etf_info.loc[etf_info['symbol'] == symbol, 'last_12_months_Open'] = (spark_img_url)
      #   # etf_data.loc[etf_data['symbol'] == symbol, 'last_12_months_Open'] = (spark_img_url)


      #   # Add the sparkline for 12 month Close history data
      #   # etf_info.loc[etf_info['symbol'] == symbol, 'last_12_months_Close'] = sparkline(history_df, 'Close')

      # # etf_data_df = pd.DataFrame.from_dict(etf_data,orient='index')
      
      # # st.write(etf_data) #.to_html(render_links=True))
          
      # etf_info = etf_info.drop(columns=['index']   )
      # etf_info_df = pd.DataFrame.from_dict(etf_info)
      
      # # st.write(etf_info_df) #.to_html(render_links=True))
      
      st.divider()
    # ###################################################
    # Charts: 
    # # of stocks being watched; 
    # ###################################################
    with tab[2]:    
      for symbol in known_options:
        yf_data = yf.Ticker(symbol) #initiate the ticker
        
        st.session_state.page_subheader = '{0} ({1})'.format(yf_data.info['shortName'], yf_data.info['symbol'])
        st.subheader(st.session_state.page_subheader)
        # st.write(yf_data)
        st.write(symbol)
        st.write("Historical data per period (Showing EMA-5day period vs EMA-10day period)")
        
        # st.write("(Showing EMA-5day period vs EMA-10day period)")
        stock_hist_df = get_hist_info(yf_data, selected_period, selected_interval)
        
        # ## Display the data table in the first column
        # st.dataframe(stock_hist_df.head(10))

        fig = draw_candle_stick_triggers(stock_hist_df, symbol)
        # Plot!
        # Create and display the bar chart in the second column
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        
        st.divider()

    # ###################################################
    # Details: 
    # Details of all stocks individually being watched; 
    # ###################################################
    with tab[3]:    
      st.write("Placeholder for details on the individual stocks")
      # stock_news_df = get_stk_news(yf_data)
      # st.write("News")
      # st.write(stock_news_df.to_html(escape=False, index=True), unsafe_allow_html=True)
      # st.divider()


    # tab1 = st.tabs(["🗃 Base Data"])
    # with tab1:

    

    #     # stock_news_df = get_stk_news(yf_data)
    #     # st.write("News")
    #     # st.write(stock_news_df.to_html(escape=False, index=True), unsafe_allow_html=True)
    #     # st.divider()
      
    #     st.write("Historical data per period")
    #     st.write("Showing EMA-5day period vs EMA-10day period")
    #     stock_hist_df = get_hist_info(yf_data, selected_period, selected_interval)
    #     st.write(stock_hist_df.to_html(escape=False, index=True), unsafe_allow_html=True)
    #     st.divider()
    
  # for symbol in symbol_list:
# #     units = hist_df.query("Symbol == @symbol")['Units'].sum()

# #     ticker = yf.Ticker(symbol)

# #     data['symbol'].append(ticker.info['symbol'])
# #     data['industry'].append(ticker.info['industry'])
# #     data['units'].append(units)
# #     current_price = ticker.info['currentPrice']
# #     data['current_price'].append(current_price)
# #     # Round to 2 decimal points
# #     market_value = round(units * current_price, 2)
# #     data['market_value'].append(market_value)

# #     # Prev close value to calculate day change
# #     prev_close = ticker.info['previousClose']
# #     day_change = (current_price - prev_close) * units
# #     data['day_change'].append(day_change)
# #     data['day_change_pct'].append(((current_price/prev_close) - 1) * 100)

# #     # History data for 12 months
# #     history = ticker.history(period='1y')['Close']

# # showing for just 1 ticker
# yf_data = yf.Ticker(ticker) #initiate the ticker

  # tab1, tab2 = st.tabs(["🗃 Data","📈 Chart"])

  # with tab1:

  #     # Subheader with company name and symbol
  #     st.session_state.page_subheader = '{0} ({1})'.format(yf_data.info['shortName'], yf_data.info['symbol'])
  #     st.subheader(st.session_state.page_subheader)
  #     # st.write(yf_data)

  #     stock_info_df = get_all_stock_info(yf_data)
  #     st.write("Overview")
  #     st.write(stock_info_df.to_html(escape=False, index=False), unsafe_allow_html=True)

  #     st.divider()

      
  #     st.write("Historical data per period")
  #     st.write("Showing EMA-5day period vs EMA-10day period")
  #     stock_hist_df = get_hist_info(yf_data, selected_period, selected_interval)
  #     st.write(stock_hist_df.to_html(escape=False, index=True), unsafe_allow_html=True)
  #     st.divider()
      
  # with tab2:
  #     fig = draw_candle_stick_triggers(stock_hist_df, ticker)
  #     # Plot!
  #     st.plotly_chart(fig, theme="streamlit", use_container_width=True)

  return

if __name__ == '__main__':
  main()

# # https://www.quantstart.com/articles/candlestick-subplots-with-plotly-and-the-alphavantage-api/

# etf_info = pd.DataFrame()
# etf_data = {} # dictionary
# for symbol in symbol_list:
#     ticker = yf.Ticker(symbol)
#     # df = get_all_stock_info(ticker)

#     etf_data[symbol] = get_hist_info(ticker, period, interval)
#     etf_info = pd.concat([get_all_stock_info(ticker), etf_info], ignore_index=True)

#     # History data for 12 months
#     history = ticker.history(period='1y')[['Open', 'Close']]
#     # Convert the history series to a DF
#     history_df = history #.to_frame()
#     # display (ticker)
#     # display (history_df.head(5))

#     # # Add the sparkline for 12 month Open history data
#     # spark_img = sparkline(history_df, 'Open')
#     # spark_img_url =  ('<img src="data:/png;pybase64,{}"/>'.format(spark_img))
#     # etf_info.loc[etf_info['symbol'] == symbol, 'last_12_months_Open'] = (spark_img_url)

#     # # Add the sparkline for 12 month Close history data
#     # etf_info.loc[etf_info['symbol'] == symbol, 'last_12_months_Close'] = sparkline(history_df, 'Close')

    
# etf_info = etf_info.drop(columns=['index']   )
# st.write(etf_info)

# #########################################################
# IGNORE BELOW
# #########################################################
