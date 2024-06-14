import globals
from algotrading_helper import *
# from algotrading_visualisations import *
from algotrading_algos import *
# from algotrading_login import *
# from algotrading_playground import *
# from convexAlgos_standalone import *

from pathlib import Path

pd.options.display.float_format = '{:,.2f}'.format
      
def main():
  
  # load the config file and the user specified default ticker list 
  refresh = False
  print("###################################")
  print("LOAD CONFIG ")
  print("###################################")
  print(" ")
  load_config(refresh)
  # print(globals.SYMBOLS, globals.PERIOD, globals.INTERVAL, globals.STOP_LOSS, globals.TAKE_PROFIT, globals.MOVING_AVERAGE_BASED, globals.TREND_BASED)
  
  symbol_list = np.sort(globals.SYMBOLS)
  period = globals.PERIOD[0]
  interval = globals.INTERVAL[0]
  globals.stop_loss_factor = float(globals.STOP_LOSS[0])
  globals.take_profit_factor = float(globals.TAKE_PROFIT[0])
  
  ma_list = globals.MOVING_AVERAGE_BASED #["SMA", "EMA","EMA 1-2 candle price continuation"]
  algo_list = globals.TREND_BASED #["4-3-1 candle price reversal"]
  globals.convex_trade_algos_list = ma_list + algo_list
  
  algo_functions_map = (globals.convex_trade_algos_list, 
                        globals.algo_functions, 
                        globals.algo_functions_args)
  
  process_time = {}
  process_time_df = pd.DataFrame()
  
  # user selected list of tickers
  # NEED DB CONNECT HERE TO SAVE THE USER 
  user_sel_list = []
  
  # load_user_selected_options()
  user_sel_list = load_user_selected_options("demo")
  
  # ***************
  # Trading DAY SETUP
  # ***************
  known_options = user_sel_list
  # selected_algos = algo_functions_map
  
  # print("Setup Day ***** ", known_options, globals.convex_trade_algos_list)
  
  # print(" ")
  # print(" ")
  
  
  # ***************
  # Trading SIGNALS
  # ***************
  
  known_options = display_watchlist()
   
  print("###################################")
  print("Processing Trading Signals Summary ")
  print("###################################")
  print(" ")
  signals_view(known_options, 
                            globals.convex_trade_algos_list, 
                            period, 
                            interval )
    
  # # ***************
  # # Trading STATUS
  # # ***************
  # print("Ticker Status View")
  # print("Shows the status of the implemented strategies for all tickers")
  
  # data = {'symbol': 'RSP', 'stock_trigger_at': 2024-05-30 11:04:00-04:00, 'stock_trigger_state': 'Sell', 'stock_price_at_trigger': 163.13999938964844, 'stock_stop_loss_atr': 163.19886193956648, 'stock_take_profit_atr': 163.03909216121752, 'stock_atr_ma': 0.03363574281030771, 'stock_ema_p1': 163.18687467319677, 'stock_ema_p2': 163.19013138629703, 'algo_strategy': 'EMA', 'tweet_post': 'RSP: Sell; 2024-05-30 11:04; 163.14; SL: 163.20; PT: 163.04',} # 'stock_previous_triggers': ['2024/05/30 11:04', '2024/05/30 10:44', '2024/05/30 10:43', '2024/05/30 10:35', '2024/05/30 10:27', '2024/05/30 10:20']} 
  # display(pd.DataFrame(data))

  # run_count = 0
  print("###################################")
  print("Processing Trading Signals Status")
  print("###################################")
  print(" ")
  stock_status_data, status_ema_merged_df = stock_status(known_options, 
                            globals.convex_trade_algos_list, 
                            period, 
                            interval,
                            )
  # await asyncio.sleep(1)
  print(status_ema_merged_df[:20]) 
  # Convert to DataFrame by flattening the dictionary
  for symbol, symbol_data in stock_status_data.items():
    print("fetching status for ticker: ", symbol)
    print(pd.DataFrame(symbol_data[:10]).sort_index(ascending=False))
    
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

# def load_signals_view(known_options, # known_options, 
#                             selected_algos, 
#                             period, 
#                             interval):
#   print("Trading Signals View")
#   print("Lists the latest trade triggers")
#   # if(main_menu not in st.session_state):
#   #   st.session_state['main_menu'] = 1
#   # await asyncio.sleep(1)
#   known_options = display_watchlist()
   
#   print("signals known_options    ***** ", known_options)
#   print("")
#   print("")
  
#   signals_view(known_options, 
#                             selected_algos, 
#                             period, 
#                             interval )
  
#   # print("###################result_signal",result_signal)
  
#   # if st.button("Generate Trading Chart"):
#   #     show_trading_charts()
  
#   # Hyperlink to generate trading chart
#   # if st.markdown("[Generate Trading Charts](show_trading_charts())"):
#   #   st.write("show the charts")
#   #   # show_trading_charts(st.session_state.user_watchlist, #known_options, 
#     #                       st.session_state.selected_algos, 
#     #                       st.session_state.period, 
#     #                       st.session_state.interval)
  
#   # time check begin
  
  
#   return
    
    
if __name__ == '__main__':
  main()
