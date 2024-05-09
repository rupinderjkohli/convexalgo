from algotrading_helper import *
from algotrading_visualisations import *
from algotrading_algos import *
from algotrading_class import *


from pathlib import Path

pd.options.display.float_format = '{:,.2f}'.format

from streamlit_option_menu import option_menu
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
  
  # # Initialize SessionState
  # session_state = SessionState(selected_algos="")
  
  # load the default ticker list
  refresh = False
  symbol_list, period, interval, stop_loss, take_profit,trading_strategy_ma,trading_strategy_trend = load_config(refresh)
  
  symbol_list = np.sort(symbol_list)
  st.session_state.period = period[0]
  st.session_state.interval = interval[0]
  st.session_state.stop_loss_factor = float(stop_loss[0])
  st.session_state.take_profit_factor = float(take_profit[0])
  
  st.session_state.moving_average = trading_strategy_ma
  st.session_state.trend_based = trading_strategy_trend 
  
  ma_list = trading_strategy_ma #["SMA", "EMA","EMA 1-2 candle price continuation"]
  algo_list = trading_strategy_trend #["4-3-1 candle price reversal"]
  convex_trade_algos_list = ma_list + algo_list
  selected_algos = convex_trade_algos_list
  st.session_state.selected_algos = convex_trade_algos_list
  
  
  # List of algo functions
  algo_functions = [strategy_sma, strategy_ema, strategy_ema_continual, strategy_431_reversal]
  
  algo_functions_map = (((convex_trade_algos_list, algo_functions)))
  st.session_state.algo_functions_map = algo_functions_map
  
  print(convex_trade_algos_list)
  
  process_time = {}
  process_time_df = pd.DataFrame()
  
  # user selected list of tickers
  user_sel_list = []
  
  # load_user_selected_options()
  user_sel_list = load_user_selected_options()
  
  print(user_sel_list)
  
  
  # if "shared" not in st.session_state:
  #  st.session_state["shared"] = True

  # if len(known_options) == 0:
  #   st.write ("Please select a ticker in the sidebar")
  #   return
  # else:
  #     st.write("home page")
  
  # st.write("# Welcome to Streamlit! 👋")
  
  # 5. Add on_change callback
  # st.write(st.session_state)
  if('main_menu' not in st.session_state):
    st.session_state['main_menu'] = st.session_state.get('main_menu', 0)
  # st.write(st.session_state)  
  
  def on_change(key):
      selection = st.session_state[key]
      # st.write(f"Selection changed to {selection}")
      
  with st.sidebar:
    choose = option_menu("Convex Algos", ["Setup Day", "---" ,"Signals", "Status", "Trading Charts", "Change Logs"],
                         icons=['house', 'camera fill', 'list-columns-reverse', 'bar-chart-line','person lines fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
                          "container": {"padding": "5!important", "background-color": "#fafafa"},
                          "icon": {"color": "orange", "font-size": "25px"}, 
                          "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                          "nav-link-selected": {"background-color": "#02ab21"},
                      },
                         key='main_menu',
                         on_change=on_change
    )
    manual_select = "Setup Day"
    # st.write(choose)
    
    # Initialize session state
    if 'main_menu' not in st.session_state:
        st.session_state.main_menu = 0 
        manual_select = st.session_state['main_menu']
    
    if st.session_state.get('main_menu', 0):
        # st.session_state['main_menu'] = st.session_state.get('main_menu', 0)#+ 1) % 5
        manual_select = st.session_state['main_menu']
        st.write(manual_select)
    else:
        manual_select = st.session_state.get('main_menu', 0) #None
        
  # st.sidebar.success("Setup your trading day")

  if (manual_select == "Setup Day" ):
    process_name = "Setup Day"
    start_time = time.time()
    known_options, selected_algos = setup_day(user_sel_list, 
                                              st.session_state.period, 
                                              st.session_state.interval, 
                                              symbol_list, 
                                              algo_functions_map)
    end_time = time.time()
    execution_time = end_time - start_time
    
    # time check begin
    for variable in ["process_name","execution_time", "start_time", "end_time"]:
        process_time[variable] = eval(variable)
    x = pd.DataFrame([process_time])
    process_time_df = pd.concat([x, process_time_df], ignore_index=True)
    # time check end
    # print(known_options)
    
    print("Setup Day st.session_state")
    print(st.session_state) 
    print("")
    print("")
    
    if(selected_algos not in st.session_state):
      st.session_state['selected_algos'] = selected_algos #st.session_state.get(selected_algos) #, selected_algos)
    print("###########################")
    print("")
    print(st.session_state)  
    print("")
    print("")
    # Store the selected option in session state
    # else: st.session_state.selected_algos = selected_algos
    
  elif (manual_select == "Signals" ):
    st.header("Trading Signals View")
    
    process_name = "Signals"
    start_time = time.time()
    
    known_options = display_watchlist()
    # print("known_options")
    
    
    # Initialize session state if user coming directly to signals
    if(selected_algos not in st.session_state):
      st.session_state['selected_algos'] = selected_algos #st.session_state.get(selected_algos) #, selected_algos)
    
    # if(stop_loss_factor not in st.session_state):
    #   st.session_state['stop_loss_factor'] = float(stop_loss[0])
    # if(take_profit_factor not in st.session_state):
    #   st.session_state['take_profit_factor'] = float(take_profit[0])
      
    print("signals known_options, st.session_state")
    print(known_options, st.session_state)
    print("")
    print("")
    asyncio.run (signals_view(st.session_state.user_watchlist, # known_options, 
                              st.session_state.selected_algos, 
                              st.session_state.period, 
                              st.session_state.interval ))
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # if st.button("Generate Trading Chart"):
    #     show_trading_charts()
    
    # Hyperlink to generate trading chart
    if st.markdown("[Generate Trading Charts](show_trading_charts())"):
      st.write("show the charts")
        # show_trading_charts(st.session_state.user_watchlist, #known_options, 
        #                       st.session_state.selected_algos, 
        #                       st.session_state.period, 
        #                       st.session_state.interval) # known_options)
    
    # time check begin
    for variable in ["process_name","execution_time", "start_time", "end_time"]:
        process_time[variable] = eval(variable)
    x = pd.DataFrame([process_time])
    process_time_df = pd.concat([x, process_time_df], ignore_index=True)
    
  elif (manual_select == "Status" ):
    # Initialize session state if user coming directly to signals
    if(selected_algos not in st.session_state):
      st.session_state[selected_algos] = selected_algos
    # if(stop_loss_factor not in st.session_state):
    #   st.session_state[stop_loss_factor] = st.session_state.get(float(stop_loss[0]))
    # if(take_profit_factor not in st.session_state):
    #   st.session_state[take_profit_factor] = st.session_state.get(float(take_profit[0]))
      
    st.write("Shows the current status of all strategies against all stocks")
    process_name = "Status"
    start_time = time.time()
    
    asyncio.run (stock_status(st.session_state.user_watchlist, # known_options, 
                              st.session_state.selected_algos, 
                              st.session_state.period, 
                              st.session_state.interval))
    # stock_status()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # time check begin
    for variable in ["process_name","execution_time", "start_time", "end_time"]:
        process_time[variable] = eval(variable)
    x = pd.DataFrame([process_time])
    process_time_df = pd.concat([x, process_time_df], ignore_index=True)
    
  elif (manual_select == "Trading Charts" ):
    st.write("Display all charts with their candlestick charts")
    st.write("Display specific chart with their candlestick charts")
    
    process_name = "Trading Charts"
    start_time = time.time()
    
    known_options = display_watchlist()
    
    show_trading_charts(st.session_state.user_watchlist, #known_options, 
                              st.session_state.selected_algos, 
                              st.session_state.period, 
                              st.session_state.interval) # known_options
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # time check begin
    for variable in ["process_name","execution_time", "start_time", "end_time"]:
        process_time[variable] = eval(variable)
    x = pd.DataFrame([process_time])
    process_time_df = pd.concat([x, process_time_df], ignore_index=True)
  
  elif (manual_select == "Change Logs" ):
    # st.write("Change Logs")
    process_name = "Change Logs"
    start_time = time.time()
    
    show_change_logs()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # time check begin
    for variable in ["process_name","execution_time", "start_time", "end_time"]:
        process_time[variable] = eval(variable)
    x = pd.DataFrame([process_time])
    process_time_df = pd.concat([x, process_time_df], ignore_index=True)
  
  print (process_time_df)
  return

if __name__ == '__main__':
  main()