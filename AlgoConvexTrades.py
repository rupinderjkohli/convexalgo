from algotrading_helper import *
from algotrading_visualisations import *
from algotrading_class import *

from pathlib import Path

pd.options.display.float_format = '${:,.2f}'.format

import streamlit.components.v1 as components

def main():
  st.set_page_config(
    page_title="Convex Trades Dashboard",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://convextrades.com/',
        # 'Report a bug': "mailto:rupinder.johar.kohli@gmail.com",
        'About': "#An *extremely* cool app displaying your GoTo Trading Dashboard!"
    }
  )    
  # """### Select Stock and Time interval"""
  # https://github.com/smudali/stocks-analysis/blob/main/dasboard/01Home.py
  # new_ticker_list = []
  
  # new_ticker = add_ticker()
  
  symbol_list = load_config()
  
  # print(type(symbol_list))
  
  symbol_list = np.sort(symbol_list)
  # print(symbol_list)
  
  # base_symbol_list = ["MSFT","PLTR","TSLA","NVDA","AMZN", "NFLX","BA","GS","SPY","QQQ","IWM","SMH","RSP"]
  # symbol_list = base_symbol_list # new_ticker_list
  
  # NSE: TATAPOWER: Tata Power Company Ltd
  # NSE: TATAINVEST: Tata Investment Corporation Ltd

  ma_list = ["SMA", "EMA"]
  algo_list = ["3-Candle Reversal"]
  
  # user selected list of tickers
  # load_user_selected_options()
  user_sel_list = []
  
  # load_user_selected_options()
  
  user_sel_list = load_user_selected_options()
  print(user_sel_list)
  
  # ticker selection
  st.sidebar.header("Choose your Stock filter: ")
  ticker = st.sidebar.multiselect('Choose Ticker', options=symbol_list,
                                help = 'Select a ticker', 
                                key='ticker_list',
                                max_selections=8,
                                default= user_sel_list, #["TSLA"],
                                placeholder="Choose an option",
                                # on_change=update_selection(),
                                )
  print(ticker)
  print(st.session_state)
  known_options = ticker
  save_user_selected_options(ticker)
  
  # period selection
  selected_period = st.sidebar.selectbox(
      'Select Period', options=['1d','5d','1mo','3mo', '6mo', 'YTD', '1y', 'all'], index=1)
  
  # interval selection
  selected_interval = st.sidebar.selectbox(
      'Select Intervals', options=['1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo'], index=2)
  
  # trading strategy selection
  algo_strategy = st.sidebar.selectbox(
      'Select Algo Strategy', options=['SMA', 'EMA', "3-Candle Reversal"], index=2)
  selected_short_window =  st.sidebar.number_input(":gray[Short Window]", step = 1, value=5)  
  selected_long_window =  st.sidebar.number_input(":gray[Long Window]", step = 1, value=8)   

  #         Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
  # Either Use period parameter or use start and end
  #     interval : str
  #         Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo

  ema_period1 = selected_short_window
  ema_period2 = selected_long_window
  if "shared" not in st.session_state:
   st.session_state["shared"] = True

  if len(known_options) == 0:
    st.write ("Please select a ticker in the sidebar")
    return
  else:
      st.write("home page")