import streamlit as st 
from algotrading_helper import *
from algotrading_visualisations import *
from algotrading_class import *

from pathlib import Path

pd.options.display.float_format = '${:,.2f}'.format

import streamlit.components.v1 as components

def show_trading_charts(known_options, 
                              selected_algos, 
                              period, 
                              interval,):
    st.write("show_trading_charts")
    lw_charts_snapshot(symbol,
                       stock_df, 
                       algo_strategy,
                       selected_short_window,
                       selected_long_window,
                       display_table = False)

def lw_charts_snapshot(symbol,
                       stock_df, 
                       algo_strategy,
                       selected_short_window,
                       selected_long_window,
                       display_table = False):

    
    # column names for long and short moving average columns
    short_window_col = str(selected_short_window) + '_' + algo_strategy
    long_window_col = str(selected_long_window) + '_' + algo_strategy
    
    stock_df.rename(columns = {'Datetime':'time'}, inplace = True)
    
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
    