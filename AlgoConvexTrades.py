from algotrading_helper import *
from algotrading_visualisations import *
from algotrading_algos import *
from algotrading_login import *
from algotrading_playground import *
from convexAlgos_standalone import *


# from streamlit_option_menu import option_menu
# import streamlit.components.v1 as components
# from st_social_media_links import SocialMediaIcons


from pathlib import Path

pd.options.display.float_format = '{:,.2f}'.format

def main():
  # st.set_page_config(
  #   page_title="Convex Algos Dashboard",
  #   page_icon="🧊",
  #   layout="wide",
  #   initial_sidebar_state="expanded",
  #   menu_items={
  #       'Get Help': 'https://convextrades.com/',
  #       # 'Report a bug': "mailto:rupinder.johar.kohli@gmail.com",
  #       'About': "#An *extremely* cool app displaying your GoTo Trading Dashboard!"
  #   }
  # )    
  # """### Select Stock and Time interval"""
  # https://github.com/smudali/stocks-analysis/blob/main/dasboard/01Home.py
  
  # # Initialize SessionState
  # session_state = SessionState(selected_algos="")
  social_media_links = [
    "https://www.twitter.com/convextrades",
    "https://www.instagram.com/convex.trades",
    "https://www.facebook.com/convextrades",
   
    ]

  # social_media_icons = SocialMediaIcons(social_media_links)
  
  # # # ***USER LOGIN***
  # user_type = user_login_process()
  # # st.write("user_type",user_type)
  # if user_type not in (['GU','RU']):
  #   return
        
  # # ***USER LOGIN DONE***
  
  # load the config file and the user specified default ticker list 
  refresh = False
  symbol_list, period, interval, stop_loss, take_profit,trading_strategy_ma,trading_strategy_trend = load_config(refresh)
  
  symbol_list = np.sort(symbol_list)
  period = period[0]
  interval = interval[0]
  stop_loss_factor = float(stop_loss[0])
  take_profit_factor = float(take_profit[0])
  
  ma_list = trading_strategy_ma #["SMA", "EMA","EMA 1-2 candle price continuation"]
  algo_list = trading_strategy_trend #["4-3-1 candle price reversal"]
  convex_trade_algos_list = ma_list + algo_list
  selected_algos = convex_trade_algos_list
  
  # List of algo functions
  algo_functions = [strategy_sma, strategy_ema, strategy_ema_continual, strategy_431_reversal]
  
  # TODO
  algo_functions_args = []
  
  algo_functions_map = (((convex_trade_algos_list, algo_functions, algo_functions_args)))
  # if('algo_functions_map' not in st.session_state):
  #   st.session_state['algo_functions_map'] = algo_functions_map
  
  print(convex_trade_algos_list)
  
  process_time = {}
  process_time_df = pd.DataFrame()
  
  # user selected list of tickers
  # NEED DB CONNECT HERE TO SAVE THE USER 
  user_sel_list = []
  
  # load_user_selected_options()
  user_sel_list = load_user_selected_options("demo")
  # if('user_watchlist' not in st.session_state):
  #   st.session_state['user_watchlist'] = user_sel_list
  
  print(user_sel_list)
    
    
  # load_signals_view(process_time,process_time_df)
  # if (st.session_state.main_menu == 0):
  #   # st.write(st.session_state.interval)
  #   (load_signals_view(process_time,process_time_df))
  load_signals_view(user_watchlist, # known_options, 
                            selected_algos, 
                            period, 
                            interval)
  
  # social_media_icons.render(sidebar=True, justify_content="space-evenly")
  
  # to_twitter("post")    
  
  # st.sidebar.success("Setup your trading day")
  # print(st.session_state)


  # ***************
  # Trading DAY SETUP
  # ***************
  # if (st.session_state.selected_menu == "Setup Day" ):
  #   process_name = "Setup Day"
  #   start_time = time.time()
  #   if('main_menu' not in st.session_state):
  #     st.session_state['main_menu'] = 1
  #   if('selected_menu' not in st.session_state):  
  #     st.session_state['selected_menu'] = "Setup Day"
      
  known_options, selected_algos = setup_day("demo",
                                            user_sel_list, 
                                            period, 
                                            interval, 
                                            symbol_list, 
                                            algo_functions_map)
  # end_time = time.time()
  # execution_time = end_time - start_time
    
  # # time check begin
  # for variable in ["process_name","execution_time", "start_time", "end_time"]:
  #     process_time[variable] = eval(variable)
  # x = pd.DataFrame([process_time])
  # process_time_df = pd.concat([x, process_time_df], ignore_index=True)
  # time check end
  # print(known_options)
  
  print("Setup Day st.session_state")
  print(known_options, selected_algos) 
  print("")
  print("")
  
  # if(selected_algos not in st.session_state):
  #   st.session_state['selected_algos'] = selected_algos #st.session_state.get(selected_algos) #, selected_algos)
  # print("###########################")
  # print("")
  # print(st.session_state)  
  # print("")
  print("")
  # Store the selected option in session state
  # else: st.session_state.selected_algos = selected_algos
  
  # ***************
  # Trading SIGNALS
  # ***************
  # elif (st.session_state.selected_menu == "Signals" ):
  (load_signals_view(user_watchlist, # known_options, 
                            selected_algos, 
                            period, 
                            interval))
    
  # # ***************
  # # Trading STATUS
  # # ***************
  # elif (st.session_state.selected_menu == "Status" ):
  #   st.header("Ticker Status View")
  #   st.caption("Shows the status of the implemented strategies for all tickers")
    
  #   # Initialize session state if user coming directly to signals
  #   if('selected_algos' not in st.session_state):
  #     st.session_state['selected_algos'] = selected_algos #st.session_state.get(selected_algos) #, selected_algos)
       
  #   process_name = "Status"
  #   start_time = time.time()
    
  #   run_count = 0
  #   stock_status_data, status_ema_merged_df, run_count = asyncio.run (stock_status(st.session_state.user_watchlist, # known_options, 
  #                             st.session_state.selected_algos, 
  #                             st.session_state.period, 
  #                             st.session_state.interval,
  #                             run_count))
  #   # await asyncio.sleep(1)
    
  #   # Convert to DataFrame by flattening the dictionary
  #   for symbol, symbol_data in stock_status_data.items():
  #     st.write("fetching status for ticker: ", symbol)
  #     st.write(pd.DataFrame(symbol_data).sort_index(ascending=False))
      
    
  #   # Hyperlink to generate trading chart
  #   if st.markdown("[Generate Trading Charts](show_trading_charts())"):
  #     st.write("show the charts")
  #     # show_trading_charts(st.session_state.user_watchlist, #known_options, 
  #     #                         st.session_state.selected_algos, 
  #     #                         st.session_state.period, 
  #     #                         st.session_state.interval)
    
  #   end_time = time.time()
  #   execution_time = end_time - start_time
    
  #   # time check begin
  #   for variable in ["process_name","execution_time", "start_time", "end_time"]:
  #       process_time[variable] = eval(variable)
  #   x = pd.DataFrame([process_time])
  #   process_time_df = pd.concat([x, process_time_df], ignore_index=True)
    
  # # ***************
  # # Trading CHARTS
  # # ***************
  # elif (st.session_state.selected_menu == "Trading Charts" ):
  #   st.header("Ticker Trading Charts")
  #   st.caption("Presents the trading view of the tickers filtered on Ticker and Date")
  #   # if(main_menu not in st.session_state):
  #   #   st.session_state['main_menu'] = 3
  #   process_name = "Trading Charts"
  #   start_time = time.time()
    
  #   known_options = display_watchlist()
    
  #   show_trading_charts(st.session_state.user_watchlist, #known_options, 
  #                             st.session_state.selected_algos, 
  #                             st.session_state.period, 
  #                             st.session_state.interval) # known_options
    
  #   end_time = time.time()
  #   execution_time = end_time - start_time
    
  #   # time check begin
  #   for variable in ["process_name","execution_time", "start_time", "end_time"]:
  #       process_time[variable] = eval(variable)
  #   x = pd.DataFrame([process_time])
  #   process_time_df = pd.concat([x, process_time_df], ignore_index=True)
  
  # # ***************
  # # CHANGE LOGS
  # # ***************
  # elif (st.session_state.selected_menu == "Change Logs" ):
  #   st.header("Change Logs")
  #   st.caption("Lists the change log since the last release")
  #   # if(main_menu not in st.session_state):
  #   #   st.session_state['main_menu'] = 4
  #   process_name = "Change Logs"
  #   start_time = time.time()
    
  #   show_change_logs()
    
  #   end_time = time.time()
  #   execution_time = end_time - start_time
    
  #   # time check begin
  #   for variable in ["process_name","execution_time", "start_time", "end_time"]:
  #       process_time[variable] = eval(variable)
  #   x = pd.DataFrame([process_time])
  #   process_time_df = pd.concat([x, process_time_df], ignore_index=True)
  
  # # ***************
  # # ALGO PLAYGROUND
  # # ***************
  # elif (st.session_state.selected_menu == "Algo Playground"):
  #   playground_ui(st.session_state.user_watchlist, #known_options, 
  #                             st.session_state.selected_algos, 
  #                             st.session_state.period, 
  #                             st.session_state.interval)
  #   # st.button("standalone_algos",type="primary")
  #   # if(st.button("standalone_algos")):
  #   #   # st.write("button clicked")
  #   #   convexalgos_standalone()
  #   # if(st.button("algo_playground")):
  #   #   asyncio.run (algo_playground())
    
  
  # print (process_time_df)
  return

def load_signals_view(user_watchlist, # known_options, 
                            selected_algos, 
                            period, 
                            interval):
  print("Trading Signals View")
  print("Lists the latest trade triggers")
  # if(main_menu not in st.session_state):
  #   st.session_state['main_menu'] = 1
  
  known_options = display_watchlist()
   
  print("signals known_options, st.session_state")
  print(known_options)
  print("")
  print("")
  (signals_view(user_watchlist, # known_options, 
                            selected_algos, 
                            period, 
                            interval ))
  
  # print("###################result_signal",result_signal)
  
  # if st.button("Generate Trading Chart"):
  #     show_trading_charts()
  
  # Hyperlink to generate trading chart
  # if st.markdown("[Generate Trading Charts](show_trading_charts())"):
  #   st.write("show the charts")
  #   # show_trading_charts(st.session_state.user_watchlist, #known_options, 
    #                       st.session_state.selected_algos, 
    #                       st.session_state.period, 
    #                       st.session_state.interval)
  
  # time check begin
  
  
  return
    
    
if __name__ == '__main__':
  main()
