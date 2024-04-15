
import pandas as pd
import numpy as np

import yfinance as yf       #install
import datetime
from datetime import datetime
import time

import streamlit as st      #install
# from streamlit_lightweight_charts import renderLightweightCharts

from lightweight_charts import Chart
from lightweight_charts.widgets import StreamlitChart

from algotrading_helper import MovingAverageCrossStrategy

# ##########################################################  
# Purpose: 
# ##########################################################  
def lw_charts_snapshot(symbol,
                       stock_hist_df, 
                       algo_strategy,
                       selected_short_window,
                       selected_long_window,
                       display_table = False):
    COLOR_BULL = 'rgba(38,166,154,0.9)' # #26a69a
    COLOR_BEAR = 'rgba(239,83,80,0.9)'  # #ef5350
    
    # df = df.reset_index()
    
    print(stock_hist_df.info())
     
    # st.subheader("Multipane Chart")

    chart = StreamlitChart(width=900, height=600)
    chart.set(stock_hist_df)

    # chart.load()
    
    stock_df, df_pos = MovingAverageCrossStrategy(symbol,
                                        stock_hist_df,
                                        selected_short_window,
                                        selected_long_window,
                                        algo_strategy,
                                        False)
    
    # column names for long and short moving average columns
    short_window_col = str(selected_short_window) + '_' + algo_strategy
    long_window_col = str(selected_long_window) + '_' + algo_strategy
    
    print(stock_hist_df.info())
    # st.write(stock_df.head())
    line = chart.create_line(short_window_col)
    short_algo_data = stock_df
    line.set(short_algo_data)
    
    line = chart.create_line(long_window_col)
    long_algo_data = stock_df
    line.set(long_algo_data)
    
    chart.load()
    
    return
