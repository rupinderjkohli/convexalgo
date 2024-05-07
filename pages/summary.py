import streamlit as st 
from algotrading_helper import *
from algotrading_visualisations import *
from algotrading_class import *

from pathlib import Path

pd.options.display.float_format = '${:,.2f}'.format

import streamlit.components.v1 as components

def sma():
    for symbol in known_options:
        st.subheader(symbol)
        stock_name =  symbol
        yf_data = yf.Ticker(symbol) #initiate the ticker
        stock_hist_df = get_hist_info(yf_data, selected_period, selected_interval)
    