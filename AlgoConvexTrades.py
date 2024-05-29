from algotrading_helper import *
# from algotrading_visualisations import *
from algotrading_algos import *
# from algotrading_login import *
# from algotrading_playground import *
# from convexAlgos_standalone import *


from pathlib import Path

pd.options.display.float_format = '{:,.2f}'.format


def main():
  
  social_media_links = [
    "https://www.twitter.com/convextrades",
    "https://www.instagram.com/convex.trades",
    "https://www.facebook.com/convextrades",
  ]
   
  
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
  
  print("convex_trade_algos_list:   ***** ",convex_trade_algos_list)
  
  process_time = {}
  process_time_df = pd.DataFrame()
  
  # user selected list of tickers
  # NEED DB CONNECT HERE TO SAVE THE USER 
  user_sel_list = []
  
  # load_user_selected_options()
  user_sel_list = load_user_selected_options("demo")
  
  print("user_sel_list:    ****** ",user_sel_list)
    
  # social_media_icons.render(sidebar=True, justify_content="space-evenly")
  
  # to_twitter("post")    
  
  

  # ***************
  # Trading DAY SETUP
  # ***************
  

  known_options = user_sel_list
  selected_algos = algo_functions_map
  
  print("Setup Day ***** ", known_options, selected_algos)
  
  print("")
  print("")
  
  
  # ***************
  # Trading SIGNALS
  # ***************
  # elif (st.session_state.selected_menu == "Signals" ):
  # load_signals_view(user_sel_list, # known_options, 
  #                           selected_algos, 
  #                           period, 
  #                           interval)
  known_options = display_watchlist()
   
  print("signals known_options    ***** ", known_options)
  print("")
  print("")
  
  signals_view(known_options, 
                            selected_algos, 
                            period, 
                            interval )
    
  # # ***************
  # # Trading STATUS
  # # ***************
  print("Ticker Status View")
  print("Shows the status of the implemented strategies for all tickers")
  
  run_count = 0
  # stock_status_data, status_ema_merged_df, run_count = asyncio.run (stock_status(known_options, 
  #                           selected_algos, 
  #                           period, 
  #                           interval,
  #                           run_count))
  # # await asyncio.sleep(1)
    
  # # Convert to DataFrame by flattening the dictionary
  # for symbol, symbol_data in stock_status_data.items():
  #   print("fetching status for ticker: ", symbol)
  #   print(pd.DataFrame(symbol_data).sort_index(ascending=False))
    
    
    
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

def load_signals_view(known_options, # known_options, 
                            selected_algos, 
                            period, 
                            interval):
  print("Trading Signals View")
  print("Lists the latest trade triggers")
  # if(main_menu not in st.session_state):
  #   st.session_state['main_menu'] = 1
  # await asyncio.sleep(1)
  known_options = display_watchlist()
   
  print("signals known_options    ***** ", known_options)
  print("")
  print("")
  
  signals_view(known_options, 
                            selected_algos, 
                            period, 
                            interval )
  
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
