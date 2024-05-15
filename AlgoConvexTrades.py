from algotrading_helper import *
from algotrading_visualisations import *
from algotrading_algos import *
from algotrading_class import *

from st_social_media_links import SocialMediaIcons


from pathlib import Path

pd.options.display.float_format = '{:,.2f}'.format

from streamlit_option_menu import option_menu
import streamlit.components.v1 as components

def main():
  st.set_page_config(
    page_title="Convex Algos Dashboard",
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
  social_media_links = [
    "https://www.twitter.com/convextrades",
    "https://www.instagram.com/convex.trades",
    "https://www.facebook.com/convextrades",
   
    ]

  social_media_icons = SocialMediaIcons(social_media_links)

  
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
  st.session_state['user_watchlist'] = user_sel_list
  
  print(user_sel_list)
  
  if('main_menu' not in st.session_state):
    st.session_state['main_menu'] = st.session_state.get('main_menu', 0)
  if('selected_menu' not in st.session_state):  
    st.session_state['selected_menu'] = "Signals"
    
    
    
  # st.write(st.session_state)  
  
  def on_change(key):
      selection = st.session_state[key]
      # st.write(f"Selection changed to {selection}")
      st.session_state['main_menu'] = selection
      st.session_state['selected_menu'] = selection
      return
      
      
  with st.sidebar:
    choose = option_menu("Convex Algos", ["Signals", "Status", "Trading Charts", "Change Logs", "---" ,"Setup Day",],
                         icons=['camera fill', 'list-columns-reverse', 'bar-chart-line','person lines fill','house', ],
                         menu_icon="app-indicator", 
                         default_index=0,
                        #  default_index=["Signals", "Status", "Trading Charts", "Change Logs", "---" ,"Setup Day",].index(st.session_state.selected_menu),
                         styles={
                          "container": {"padding": "5!important", "background-color": "#fafafa"},
                          "icon": {"color": "orange", "font-size": "25px"}, 
                          "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                          "nav-link-selected": {"background-color": "#02ab21"},
                      },
                         key='main_menu',
                         on_change=on_change
    )
    manual_select = "Signals"
    # st.write(choose)
    # Update session state based on selection
    st.session_state.selected_menu = choose
    
    # # Initialize session state
    # if 'main_menu' not in st.session_state:
    #     st.session_state.main_menu = 0 
    #     manual_select = st.session_state['main_menu']
    
    if st.session_state.get('main_menu', 0):
        # st.session_state['main_menu'] = st.session_state.get('main_menu', 0)#+ 1) % 5
        manual_select = st.session_state['main_menu']
        # st.write(manual_select)
    else:
        manual_select = st.session_state.get('main_menu', 0) #None
  
  social_media_icons.render(sidebar=True, justify_content="space-evenly")
  
  to_twitter("post")    
  
  # st.sidebar.success("Setup your trading day")

  if (st.session_state.selected_menu == "Setup Day" ):
    process_name = "Setup Day"
    start_time = time.time()
    if('main_menu' not in st.session_state):
      st.session_state['main_menu'] = 0
    if('selected_menu' not in st.session_state):  
      st.session_state['selected_menu'] = "Setup Day"
      
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
    
  elif (st.session_state.selected_menu == "Signals" ):
    st.header("Trading Signals View")
    # if(main_menu not in st.session_state):
    #   st.session_state['main_menu'] = 1
    
    process_name = "Signals"
    start_time = time.time()
    
    known_options = display_watchlist()
    # print("known_options")
    
    if('selected_menu' not in st.session_state):  
      st.session_state['selected_menu'] = "Signals"
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
      #                       st.session_state.interval)
    
    # time check begin
    for variable in ["process_name","execution_time", "start_time", "end_time"]:
        process_time[variable] = eval(variable)
    x = pd.DataFrame([process_time])
    process_time_df = pd.concat([x, process_time_df], ignore_index=True)
    
  elif (st.session_state.selected_menu == "Status" ):
    st.header("Ticker Status View")
    st.caption("Shows the status of the implemented strategies for all tickers")
    
    # if(main_menu not in st.session_state):
    #   st.session_state['main_menu'] = 2
    
    # Initialize session state if user coming directly to signals
    # Initialize session state if user coming directly to signals
    if(selected_algos not in st.session_state):
      st.session_state['selected_algos'] = selected_algos #st.session_state.get(selected_algos) #, selected_algos)
    
    # if(stop_loss_factor not in st.session_state):
    #   st.session_state[stop_loss_factor] = st.session_state.get(float(stop_loss[0]))
    # if(take_profit_factor not in st.session_state):
    #   st.session_state[take_profit_factor] = st.session_state.get(float(take_profit[0]))
      
    process_name = "Status"
    start_time = time.time()
    
    stock_status_data, status_ema_merged_df = asyncio.run (stock_status(st.session_state.user_watchlist, # known_options, 
                              st.session_state.selected_algos, 
                              st.session_state.period, 
                              st.session_state.interval))
    
    # st.write("###############################")
    # st.write(status_ema_merged_df.sort_index(ascending=False))
    # st.write("###############################")
    
    # st.write("###############################")
    # st.write(type(stock_status_data))
    # st.write(stock_status_data.keys())
    
    # # Convert to DataFrame by flattening the dictionary
    for symbol, symbol_data in stock_status_data.items():
      st.write("fetching status for ticker: ", symbol)
      st.write(pd.DataFrame(symbol_data).sort_index(ascending=False))
      
    
    # Hyperlink to generate trading chart
    if st.markdown("[Generate Trading Charts](show_trading_charts())"):
      st.write("show the charts")
      # show_trading_charts(st.session_state.user_watchlist, #known_options, 
      #                         st.session_state.selected_algos, 
      #                         st.session_state.period, 
      #                         st.session_state.interval)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # time check begin
    for variable in ["process_name","execution_time", "start_time", "end_time"]:
        process_time[variable] = eval(variable)
    x = pd.DataFrame([process_time])
    process_time_df = pd.concat([x, process_time_df], ignore_index=True)
    
  elif (st.session_state.selected_menu == "Trading Charts" ):
    st.header("Ticker Trading Charts")
    st.caption("Presents the trading view of the tickers filtered on Ticker and Date")
    # if(main_menu not in st.session_state):
    #   st.session_state['main_menu'] = 3
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
  
  elif (st.session_state.selected_menu == "Change Logs" ):
    st.header("Change Logs")
    st.caption("Lists the change log since the last release")
    # if(main_menu not in st.session_state):
    #   st.session_state['main_menu'] = 4
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