
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
                       stock_hist_df, 
                       algo_strategy,
                       selected_short_window,
                       selected_long_window,
                       display_table = False):

    stock_df, df_pos, previous_triggers, buy_short, sell_long = MovingAverageCrossStrategy(symbol,
                                        stock_hist_df,
                                        selected_short_window,
                                        selected_long_window,
                                        algo_strategy,
                                        False)
    
    
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