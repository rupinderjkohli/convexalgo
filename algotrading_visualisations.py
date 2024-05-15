
import pandas as pd
import numpy as np

import yfinance as yf       #install
import datetime as dt

from datetime import datetime, timedelta, date, timezone
import time
import pytz

import streamlit as st      #install
# from streamlit_lightweight_charts import renderLightweightCharts

from lightweight_charts import Chart
from lightweight_charts.widgets import StreamlitChart

import matplotlib.pyplot as plt

from algotrading_helper import * #MovingAverageCrossStrategy, timeToTz, unix_timestamp


# ##########################################################  
# Purpose: 
# ##########################################################  
def show_trading_charts(known_options, 
                              selected_algos, 
                              selected_period, 
                              selected_interval):
  # st.write("show_trading_charts")
  # Selection for a specific time frame.
  
  # can we use stock_view_details here
  # Define the market pattern


  selected_etf_data = get_selected_stock_history(known_options,selected_period, selected_interval)
  # st.subheader('Select a Ticker')
  selected_ticker = st.selectbox("Select Ticker",options=known_options,
                                help = 'Select a ticker', 
                                key='visualise_ticker',
                                placeholder="Choose a Ticker",)
  
  # st.subheader('Select a Date Range')
  # st.write(selected_etf_data[selected_ticker])
  df = selected_etf_data[selected_ticker]
  
  col1, col2 = st.columns(2)
  dt = datetime.now()
  # st.write(dt)
  # # st.write(datetime.date(2019,1,2))

  # Extract the date part using the date() method
  date_only = dt.date()
  
  # # Define minimum and maximum dates
  min_date = (selected_etf_data[selected_ticker].index.min())
  max_date = (selected_etf_data[selected_ticker].index.max())
  
  
  # # Default value
  # default_date = datetime.date(2022, 5, 1)

  with col1:
      # st.write('Select a Start Date')
      # Create a datetime object
      
      start_date = st.date_input('Start Date',
                                 min_value= min_date,
                                 max_value= max_date,
                                 value=min_date,
                                 key="start_date",
                                 )

  with col2:    
      # st.write('Select an End Date')
      end_date = st.date_input('End Date',
                               min_value=min_date,
                               max_value=max_date,
                               value=max_date,
                               key="end_date",
                               )
  # st.write(pd.Timestamp(start_date), pd.Timestamp(end_date)+ timedelta(days=1), df.index.tz_localize(None))
  
  df_selected = df[(pd.Timestamp(start_date) <= df.index.tz_localize(None)) & 
                   (df.index.tz_localize(None) <= (pd.Timestamp(end_date) + timedelta(days=1)))]
  
  # st.write(df_selected.index.min(), 
  #          df_selected.index.max())
  st.write(df_selected.head())

  # if(start_date != None or end_date != None):
  #   if(start_date < end_date):
  #       df_select = df[start_date:end_date]
  #   else:
  #       st.warning("Invalid Date Range - Re-enter Dates")
  # for symbol in known_options:
  
  # fig = plot_stk_hist(df_selected)
  # st.plotly_chart(fig, theme="streamlit", use_container_width=True)
  
  ticker_charts_snapshot(selected_ticker,
                       df_selected,)
  
  st.write("")
  lw_charts_snapshot(selected_ticker,
                       df_selected, 
                      )
  
  st.write("")
  fig = draw_candle_stick_chart(selected_ticker,df_selected)
  st.plotly_chart(fig, theme="streamlit", use_container_width=True)
  
  # Visualize the data
  df_mkt_patterns = identify_market_patterns(df)
  # st.write(df_mkt_patterns)
  
  stock_status_data, status_ema_merged_df = asyncio.run (stock_status(st.session_state.user_watchlist, # known_options, 
                              st.session_state.selected_algos, 
                              st.session_state.period, 
                              st.session_state.interval))
  
  # st.write("1",type(stock_status_result))
  # print(stock_status_result.columns)
  df_selected_ticker = stock_status_data[selected_ticker] #etf_processed_signals_df.loc[selected_ticker]
  # df_selected_ticker_filtered = df_selected_ticker[(pd.Timestamp(start_date) <= df_selected_ticker.index.tz_localize(None)) & 
  #                  (df_selected_ticker.index.tz_localize(None) <= (pd.Timestamp(end_date) + timedelta(days=1)))]
  
  st.write(df_selected_ticker)
  # apdict = mpf.make_addplot(df[['Bullish Engulfing', 'Bearish Engulfing', 'Doji', 'Hammer', 'Shooting Star']]*100,
  #                           type='scatter', markersize=200, marker='^')

  # mpf.plot(df, type='candle', style='charles', addplot=apdict,
  #         title='Candlestick Patterns', ylabel='Price (USD)')

  # lw_charts_snapshot(selected_ticker,
  #                     df, 
  #                     algo_strategy,
  #                     selected_short_window,
  #                     selected_long_window,
  #                     display_table = False)
  
  # lw_charts_snapshot(selected_ticker,
  #                      df_selected, 
  #                      algo_strategy = 'EMA',
  #                      selected_short_window = 5,
  #                      selected_long_window = 8,
  #                      )
    
  return

# ##########################################################  
# Purpose: 
# ##########################################################  
def ticker_charts_snapshot(symbol,
                       df, 
                      #  algo_strategy,
                      #  selected_short_window,
                      #  selected_long_window,
                      #  display_table = False
                      ):

    df.rename(columns = {'Datetime':'time'}, inplace = True)
    
    chart = StreamlitChart(width=900, height=400)
    # chart = Chart()
    chart.set(df, render_drawings = True)
    # print(stock_df.info())
    
    # line = chart.create_line(short_window_col, color = 'blue', price_line = True, price_label = True)
    # short_algo_data = stock_df
    # line.set(short_algo_data)
    
    # line = chart.create_line(long_window_col, color = 'red', price_line = True, price_label = True)
    # long_algo_data = stock_df
    # line.set(long_algo_data)
    
    # chart styling
    chart.layout(background_color='#090008', text_color='#FFFFFF', font_size=16,
                 font_family='Helvetica')

    chart.candle_style(up_color='#00ff55', down_color='#ed4807',
                       border_up_color='#FFFFFF', border_down_color='#FFFFFF',
                       wick_up_color='#FFFFFF', wick_down_color='#FFFFFF')

    chart.crosshair(mode='normal', vert_color='#FFFFFF', vert_style='dotted',
                    horz_color='#FFFFFF', horz_style='dotted')

    chart.legend(visible=True, font_size=14)
    
    chart.load()
    # chart.show(block=False)
    return
    
# ##########################################################  
# Purpose: 
# ##########################################################  
def lw_charts_snapshot(symbol,
                       stock_df, 
                      #  algo_strategy,
                      #  selected_short_window,
                      #  selected_long_window,
                       ):

    
    # # column names for long and short moving average columns
    # short_window_col = str(selected_short_window) + '_' + algo_strategy
    # long_window_col = str(selected_long_window) + '_' + algo_strategy
    
    stock_df.rename(columns = {'Datetime':'time'}, inplace = True)
    
    chart = StreamlitChart(width=900, height=400)
    # chart = Chart()
    chart.set(stock_df, render_drawings = True)
    # print(stock_df.info())
    
    # line = chart.create_line(short_window_col, color = 'blue', price_line = True, price_label = True)
    # short_algo_data = stock_df
    # line.set(short_algo_data)
    
    # line = chart.create_line(long_window_col, color = 'red', price_line = True, price_label = True)
    # long_algo_data = stock_df
    # line.set(long_algo_data)
    
    # chart styling
    chart.layout(background_color='#090008', text_color='#FFFFFF', font_size=16,
                 font_family='Helvetica')

    chart.candle_style(up_color='#00ff55', down_color='#ed4807',
                       border_up_color='#FFFFFF', border_down_color='#FFFFFF',
                       wick_up_color='#FFFFFF', wick_down_color='#FFFFFF')

    chart.crosshair(mode='normal', vert_color='#FFFFFF', vert_style='dotted',
                    horz_color='#FFFFFF', horz_style='dotted')

    chart.legend(visible=True, font_size=14)
    
    chart.load()
    # chart.show(block=False)
    return


# def lw_charts_snapshot(symbol,
#                        stock_hist_df, 
#                        algo_strategy,
#                        selected_short_window,
#                        selected_long_window,
#                        display_table = False):
    
#     # Create a new chart instance
#     chart = LightweightCharts()

#     # Set chart options, including timezone for x-axis
#     options = {'timeScale': {'timeVisible': True, 'secondsVisible': False, 'timezone': 'Etc/UTC'}}

#     # Create a new chart with the specified options
#     chart.create_chart(options)
    
    
#     # column names for long and short moving average columns
#     short_window_col = str(selected_short_window) + '_' + algo_strategy
#     long_window_col = str(selected_long_window) + '_' + algo_strategy
    
#     stock_df, df_pos = MovingAverageCrossStrategy(symbol,
#                                         stock_hist_df,
#                                         selected_short_window,
#                                         selected_long_window,
#                                         algo_strategy,
#                                         False)
    
    
#     timeZone = "America/New_York"
    
#     stock_df = stock_hist_df.reset_index()
    
#     # Convert timestamps to milliseconds since epoch (required by Lightweight Charts)
#     stock_df['time'] = stock_df['Datetime'].astype(int) // 10**6
#     stock_df = stock_df.set_index('time')
    
#     # Add the data series to the chart
#     chart.add_line_series(stock_df['time'], stock_df['Close'])
    
#     return

def show_atr(data):
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(25, 8), sharex=True)  # Share x-axis

    # Stock price plot with ATR-based buy and sell signals
    ax1.plot(data['Close'], label='Close Price', alpha=0.5)
    ax1.scatter(data.index[buy_signal], data['Close'][buy_signal], label='Buy Signal (ATR)', marker='^', color='green', alpha=1)
    ax1.scatter(data.index[sell_signal], data['Close'][sell_signal], label='Sell Signal (ATR)', marker='v', color='red', alpha=1)
    for idx in data.index[buy_signal]:
        ax1.axvline(x=idx, color='green', linestyle='--', alpha=0.5)
    for idx in data.index[sell_signal]:
        ax1.axvline(x=idx, color='red', linestyle='--', alpha=0.5)
    ax1.set_title(f'{ticker} Stock Price with ATR-Based Signals')
    ax1.set_ylabel('Price')
    ax1.legend()

    # ATR subplot with buy and sell signals
    ax2.plot(data['ATR'], label='Average True Range', color='purple')
    ax2.plot(data['ATR_MA'], label='14-day MA of ATR', color='orange', alpha=0.6)
    # ax2.scatter(data.index[buy_signal], data['ATR'][buy_signal], label='Buy Signal (ATR)', marker='^', color='green')
    # ax2.scatter(data.index[sell_signal], data['ATR'][sell_signal], label='Sell Signal (ATR)', marker='v', color='red')
    # for idx in data.index[buy_signal]:
    #     ax2.axvline(x=idx, color='green', linestyle='--', alpha=0.5)
    # for idx in data.index[sell_signal]:
    #     ax2.axvline(x=idx, color='red', linestyle='--', alpha=0.5)
    ax2.set_title(f'{ticker} Average True Range (ATR) with Signals')
    ax2.set_ylabel('ATR')
    ax2.legend()

    plt.tight_layout()
    plt.show()
    st.plotly_chart(fig)
    return


# ##########################################################  
# Purpose: 
# ##########################################################
def draw_candle_stick_triggers(symbol, df, short_window, long_window, algo_strategy):
  
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
      width=800, height=900,
      # title= symbol #  "NVDA, Today - Dec 2023",
      yaxis_title= symbol #'NVDA Stock'
  )
  
  fig.update_xaxes(rangebreaks=[
        dict(bounds=[16, 9.30], pattern="hour"), #hide hours outside of 9am-5pm
        dict(bounds=["sat", "mon"])
    ])
  
  # fig.show()
  return fig

# ##########################################################  
# Purpose: 
# ##########################################################
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

# ##########################################################  
# Purpose: 
# ##########################################################
def plot_stk_hist(df):
  # print("plotting Data and Histogram")
  fig = plt.figure(figsize=(12, 5))
  plt.plot(df.Close, color='green')
  plt.plot(df.Open, color='red')
  plt.ylabel("Counts")
  plt.xlabel("Date")

  return fig

# ##########################################################  
# Purpose: 
# ##########################################################
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
    return fig #png_base64 #png_base64 #('<img src="data:image/png;base64,{}"/>'.format(decoded_data))
    # return ('<img src="data:/png;pybase64,{}"/>'.format(png_base64))

# ##########################################################  
# Purpose: 
# ##########################################################
def draw_candle_stick_chart(symbol, df):
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
  fig.update_xaxes(rangebreaks=[
        dict(bounds=[16, 9.30], pattern="hour"), #hide hours outside of 9am-5pm
        dict(bounds=["sat", "mon"])
    ])
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
  
  fig.update_xaxes(rangebreaks=[
        dict(bounds=[16, 9.30], pattern="hour"), #hide hours outside of 9am-5pm
        dict(bounds=["sat", "mon"])
    ])
  # fig.show()
  return fig

