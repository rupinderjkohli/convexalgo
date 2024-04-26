
import pandas as pd
import numpy as np

import yfinance as yf       #install
# import datetime
from datetime import datetime
from datetime import timezone
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
def lw_charts_snapshot(symbol,
                       stock_df, 
                       algo_strategy,
                       selected_short_window,
                       selected_long_window,
                       display_table = False):

    # stock_df, df_pos, previous_triggers = MovingAverageCrossStrategy(symbol,
    #                                     stock_hist_df,
    #                                     selected_short_window,
    #                                     selected_long_window,
    #                                     algo_strategy,
    #                                     False)
    
    
    # column names for long and short moving average columns
    short_window_col = str(selected_short_window) + '_' + algo_strategy
    long_window_col = str(selected_long_window) + '_' + algo_strategy
    
    stock_df.rename(columns = {'Datetime':'time'}, inplace = True)
    
    # stock_df = stock_df.reset_index()
    # Convert timestamp strings to datetime objects
    # stock_df['timestamp'] = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') for ts in stock_df['Datetime']]
    
    # Convert timezone to desired timezone (e.g., from UTC to US/Eastern)
    # desired_timezone = pytz.timezone(timeZone)
    
    # stock_df['timestamp'] = stock_df['Datetime'].dt.tz_convert(desired_timezone)
    # stock_df.set_index('timestamp')
    # st.write(stock_df.info())
    # stock_df['time'] = stock_df['Datetime'].astype(int) // 10**6
    # st.write(stock_df.info())
    # stock_df = stock_df.set_index('time')
    # st.write(stock_df.info()) 
    # st.write(stock_df.index.name)
    
    # print("stock_df.tail(90)")
    # print(stock_df.info())
    
    # # Convert timestamps to UTC
    # timezone = pytz.timezone('America/New_York')  # Replace 'America/New_York' with your timezone
    # # stock_df['utc_timestamps'] = [timezone.localize(timestamp).astimezone(pytz.utc) for timestamp in stock_df['Datetime']]


    # # Assuming you have a datetime object with timezone information
    # dt_with_tz = datetime.now(pytz.timezone('America/New_York'))

    # # Convert the datetime object to UTC
    # dt_utc = dt_with_tz.astimezone(pytz.utc)

    # # If you want to localize a naive datetime to a specific timezone
    # naive_dt = datetime.now()
    # timezone = pytz.timezone('America/New_York')
    # localized_dt = timezone.localize(naive_dt)

    # # Now you can convert the localized datetime to UTC if needed
    # localized_dt_utc = localized_dt.astimezone(pytz.utc)
    
    # stock_df['utc_timestamps'] = [timezone.localize(timestamp).astimezone(pytz.utc) for timestamp in stock_df['Datetime']]

    # st.write(dt_with_tz,
    #          naive_dt,
    #          timezone,
    #          localized_dt,
    #          localized_dt_utc)
    # Convert timestamps to milliseconds since epoch (required by Lightweight Charts)
    # stock_df['time'] = stock_df['Datetime'].astype(int) // 10**6
    # stock_df['utc_timestamps'] = time_to_tz(stock_df['time'][1], desired_timezone)
    # st.write(time_to_tz(stock_df['time'][1], desired_timezone))
    # st.write(stock_df['utc_timestamps'][:10])
    
    # stock_df['last_time'] = pd.to_datetime(stock_df['Datetime']).dt.tz_convert('UTC')
    
    # stock_df['last_time_utc'] = pd.to_datetime(stock_df['Datetime']).dt.tz_convert(desired_timezone)
    #                         #  .dt.tz_localize('Europe/Paris') \
    
    # stock_df['last_time_utc'] = pytz.utc.localize(stock_df['Datetime'])                         


    # stock_df['timestamp'] = [ts.replace(tzinfo=pytz.utc).astimezone(desired_timezone) for ts in stock_df['Datetime']]
    # st.write("stock_df.tail(90)")
    # st.write(stock_df.tail(90))
     
    # st.write(stock_df.info())
    
    # print("stock_df.info()")
    # print(stock_df.info())
    
    
    
    # st.write(stock_hist_df.info())
    
    # # Get the local time zone
    # local_tz = pytz.timezone(pytz.country_timezones['US'][0])  # Adjust the country code as needed
    # st.write(local_tz)
    # st.write(stock_df.info()) 
    
    
    # stock_df['last_time_utc'] = stock_df['last_time_utc'].astype(str)
    # print(stock_df.info())
    # stock_df = stock_df.set_index('last_time')


    chart = StreamlitChart(width=900, height=400)
    # chart = Chart()
    chart.set(stock_df, render_drawings = True)
    # print(stock_df.info())
    
    line = chart.create_line(short_window_col, color = 'blue', price_line = True, price_label = True)
    short_algo_data = stock_df
    line.set(short_algo_data)
    
    line = chart.create_line(long_window_col, color = 'red', price_line = True, price_label = True)
    long_algo_data = stock_df
    line.set(long_algo_data)
    
    # chart styling
    chart.layout(background_color='#090008', text_color='#FFFFFF', font_size=16,
                 font_family='Helvetica')

    chart.candle_style(up_color='#00ff55', down_color='#ed4807',
                       border_up_color='#FFFFFF', border_down_color='#FFFFFF',
                       wick_up_color='#FFFFFF', wick_down_color='#FFFFFF')

    # chart.volume_config(up_color='#00ff55', down_color='#ed4807')

    # chart.watermark('1D', color='rgba(180, 180, 240, 0.7)')

    chart.crosshair(mode='normal', vert_color='#FFFFFF', vert_style='dotted',
                    horz_color='#FFFFFF', horz_style='dotted')

    chart.legend(visible=True, font_size=14)
    
    chart.load()
    # chart.show(block=False)
    return


def time_to_tz(original_time, time_zone):
    # print(original_time)
    # zoned_date = tz.localize(datetime(2011, 2, 11, 20), is_dst=None)
    #(original_time * 1000).toLocaleString('en-US', { timeZone })
    # return zoned_date.getTime() / 1000
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
  print(df.info())
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
        dict(bounds=[16, 9], pattern="hour"), #hide hours outside of 9am-5pm
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
  plt.figure(figsize=(12, 5))
  plt.plot(df.Close, color='green')
  plt.plot(df.Open, color='red')
  plt.ylabel("Counts")
  plt.xlabel("Date")

  return

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
    return png_base64 #png_base64 #('<img src="data:image/png;base64,{}"/>'.format(decoded_data))
    # return ('<img src="data:/png;pybase64,{}"/>'.format(png_base64))

# ##########################################################  
# Purpose: 
# ##########################################################
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
  fig.update_xaxes(rangebreaks=[
        dict(bounds=[16, 9], pattern="hour"), #hide hours outside of 9am-5pm
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
        dict(bounds=[16, 9], pattern="hour"), #hide hours outside of 9am-5pm
        dict(bounds=["sat", "mon"])
    ])
  # fig.show()
  return fig

