import streamlit as st 
from algotrading_helper import *
from algotrading_visualisations import *
from algotrading_class import *

from pathlib import Path

pd.options.display.float_format = '${:,.2f}'.format

import streamlit.components.v1 as components

# def show_trading_charts(known_options, 
#                               selected_algos, 
#                               period, 
#                               interval,):
#     st.write("show_trading_charts")
    