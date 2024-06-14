import globals
from algotrading_helper import *
from algotrading_algos import *
from algotrading_playground import *
# from algotrading_login import *
# from algotrading_visualisations import *
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

    
if __name__ == '__main__':
  main()
