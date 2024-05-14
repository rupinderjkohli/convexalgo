
import pandas as pd
import numpy as np

import yfinance as yf       #install
import datetime
from datetime import datetime
import time
import pytz

import csv

import streamlit as st      #install
from streamlit_js_eval import streamlit_js_eval

# from lightweight_charts import Chart
import time
import asyncio
import nest_asyncio


from millify import millify # shortens values (10_000 ---> 10k)

# To read external property file
from jproperties import Properties


# import tracemalloc

# # Enable tracemalloc
# tracemalloc.start()

async def strategy_sma(symbol,
                 stock_hist_df,
                 selected_period, 
                 selected_interval,
                 algo_strategy,
                 selected_short_window,
                 selected_long_window,
                 is_summary,
                 ):
    
    print("processing strategy_sma ", symbol)
    await asyncio.sleep(1)
    #   print("Function B is done")
    
    stock_df, df_pos, previous_triggers, short_window_col, long_window_col = MovingAverageCrossStrategy(symbol,
                                                                    stock_hist_df,
                                                                    selected_short_window,
                                                                    selected_long_window,
                                                                    algo_strategy,)
    trading_snapshot_sma = await trading_signals_view(symbol,
                                                 algo_strategy,
                                                 stock_df,
                                                 df_pos,
                                                 previous_triggers,
                                                 selected_period,
                                                 selected_interval,
                                                 short_window_col, 
                                                 long_window_col,
                                                 is_summary,)
    
    
    if (is_summary):
        return_snapshot = trading_snapshot_sma
    else: return_snapshot = stock_df
    return return_snapshot
    # return trading_snapshot_sma
  
  
async def strategy_ema(symbol,
                 stock_hist_df,
                 selected_period, 
                 selected_interval,
                 algo_strategy,
                 selected_short_window,
                 selected_long_window,
                 is_summary,
                 ):
    await asyncio.sleep(1)
    print("processing strategy_ema ", symbol)
    # await asyncio.sleep(1)
    stock_df, df_pos, previous_triggers, short_window_col, long_window_col = MovingAverageCrossStrategy(symbol,
                                                                    stock_hist_df,
                                                                    selected_short_window,
                                                                    selected_long_window,
                                                                    algo_strategy,
                                                                    True)
    trading_snapshot_ema = await trading_signals_view(symbol,
                                                 algo_strategy,
                                                 stock_df,
                                                 df_pos,
                                                 previous_triggers,
                                                 selected_period,
                                                 selected_interval,
                                                 short_window_col,
                                                 long_window_col,
                                                 is_summary,)
    
    
    if (is_summary):
        return_snapshot = trading_snapshot_ema
    else: return_snapshot = stock_df
    # st.write(">>>>>>>>>>>strategy ema >>>>>>>>>> return_snapshot",type(return_snapshot))
    return return_snapshot
    
    # return trading_snapshot_ema
    
  
async def strategy_ema_continual(symbol,
                                 stock_hist_df,
                                 selected_period, 
                                 selected_interval,
                                 algo_strategy,
                                 selected_short_window,
                                 selected_long_window,
                                 is_summary,
                                 ):
    await asyncio.sleep(1)
    print("processing strategy_ema_continual ", symbol)
    # await asyncio.sleep(1)
    stock_df, df_pos, previous_triggers, short_window_col, long_window_col = MovingAverageCrossStrategy(symbol,
                                                                    stock_hist_df,
                                                                    selected_short_window,
                                                                    selected_long_window,
                                                                    algo_strategy,
                                                                    True)
    # st.write(">>>>strategy_ema_continual - stock_df",stock_df)
    stock_ema_continual_df = await ema_continual(symbol,
                             stock_df,
                             selected_short_window,
                             selected_long_window,
                             algo_strategy,
                                    #   selected_short_window,
                                    #   selected_long_window
                             )
     
    trading_snapshot_ema_continual = await trading_signals_view(symbol,
                                                 algo_strategy,
                                                 stock_ema_continual_df,
                                                 df_pos,
                                                 previous_triggers,
                                                 selected_period,
                                                 selected_interval,
                                                 short_window_col,
                                                 long_window_col,
                                                 is_summary,)
    
    if (is_summary):
        return_snapshot = trading_snapshot_ema_continual
    else: return_snapshot = stock_ema_continual_df
    # st.write(">>>>>>>>>>>strategy ema continual >>>>>>>>>> return_snapshot",type(return_snapshot))
    return return_snapshot
    
    
  
async def strategy_431_reversal(symbol,
                                stock_hist_df,
                                selected_period, 
                                selected_interval,
                                is_summary,
                                algo_strategy = "4-3-1 candle price reversal",
                                ):
    print("processing strategy_431_reversal ", symbol)
    await asyncio.sleep(1)
    # Collate high level stats on the data
    quick_explore = {}

    quick_explore_df = pd.DataFrame() 
    
    df_four_three_one_soldiers = await strategy_four_three_one_soldiers(symbol,
                                    stock_hist_df,
                                    selected_period, 
                                    selected_interval,
                                    algo_strategy = "4-3-1 candle price reversal",)
    
    # st.write("df_four_three_one_soldiers",type(df_four_three_one_soldiers),df_four_three_one_soldiers)

    # st.write("df_four_three_one_soldiers")
    # Get the index of the row where the column value is "Buy" or "Sell"
    # buy_sell_index = (df_four_three_one_soldiers == "Buy") | (df_four_three_one_soldiers == "Sell")
    # index_of_buy_sell = buy_sell_index.idxmax(axis=1)
    # st.write(index_of_buy_sell)
    stock_trigger_at = df_four_three_one_soldiers.index.max()
    # stock_trigger_at = df_four_three_one_soldiers.index.index_of_buy_sell()
    
    stock_trigger_state = df_four_three_one_soldiers.loc[df_four_three_one_soldiers.index ==  stock_trigger_at, "position"].to_list()[0]
    stock_price_at_trigger = df_four_three_one_soldiers.loc[df_four_three_one_soldiers.index ==  stock_trigger_at, "Close"].to_list()[0]
    
    # df_four_three_one_soldiers, buy_short, sell_long = calculate_atr_buy_sell(df_four_three_one_soldiers)
    
    # st.write("df_four_three_one_soldiers.columns")
    # print(df_four_three_one_soldiers.columns)
    # st.write(stock_trigger_state)
    
    if (stock_trigger_state == "Buy"):
        stock_stop_loss_atr = stock_price_at_trigger - st.session_state.stop_loss_factor * (df_four_three_one_soldiers.loc[(df_four_three_one_soldiers.index == df_four_three_one_soldiers.index.max()), "atr_ma"]).to_list()[0]
        stock_take_profit_atr = (stock_price_at_trigger + st.session_state.take_profit_factor * (df_four_three_one_soldiers.loc[(df_four_three_one_soldiers.index == df_four_three_one_soldiers.index.max()), "atr_ma"])).to_list()[0]

        # df_four_three_one_soldiers['stop_loss_atr'] = df_four_three_one_soldiers.Close - st.session_state.stop_loss_factor * df_four_three_one_soldiers.atr_ma
        # df_four_three_one_soldiers['take_profit_atr'] = df_four_three_one_soldiers.Close + st.session_state.take_profit_factor * df_four_three_one_soldiers.atr_ma
    
    elif (stock_trigger_state == "Sell"):
        stock_stop_loss_atr = (stock_price_at_trigger + st.session_state.stop_loss_factor * (df_four_three_one_soldiers.loc[(df_four_three_one_soldiers.index == df_four_three_one_soldiers.index.max()), "atr_ma"])).to_list()[0]
        stock_take_profit_atr = (stock_price_at_trigger - st.session_state.take_profit_factor * (df_four_three_one_soldiers.loc[(df_four_three_one_soldiers.index == df_four_three_one_soldiers.index.max()), "atr_ma"])).to_list()[0]

        # df_four_three_one_soldiers['stop_loss_atr'] = df_four_three_one_soldiers.Close + st.session_state.stop_loss_factor * df_four_three_one_soldiers.atr_ma
        # df_four_three_one_soldiers['take_profit_atr'] = df_four_three_one_soldiers.Close - st.session_state.take_profit_factor * df_four_three_one_soldiers.atr_ma
          
            
    # stock_ema_p1 = df_pos.loc[df_pos.index == df_pos.index.max(), short_window_col].to_list()[0]
    # stock_ema_p2 = df_pos.loc[df_pos.index == df_pos.index.max(), long_window_col].to_list()[0]

    # stock_atr_ma = df_pos.loc[(df_pos.index == df_pos.index.max()), "atr_ma"].to_list()[0]

    # stock_view_details = etf_data[symbol]
    # print("processing strategy_431_reversal ", df_four_three_one_soldiers.columns)
    
    previous_triggers = df_four_three_one_soldiers[['position']][-6:]
    previous_triggers_list = previous_triggers.index.strftime('%Y/%m/%d %H:%M')
    previous_triggers_list = np.sort(previous_triggers_list)[::-1]
    stock_previous_triggers = previous_triggers_list 
    
    tweet_post =  symbol + ': ' + stock_trigger_state + '; ' + str(stock_trigger_at.strftime("%Y-%m-%d %H:%M")) +'; ' + str("{:.2f}".format(stock_price_at_trigger)) +'; SL: ' + str("{:.2f}".format(stock_stop_loss_atr)) +'; PT: ' + str("{:.2f}".format(stock_take_profit_atr))
        
    for variable in ["symbol",
                                "stock_trigger_at",
                                "stock_trigger_state",
                                "stock_price_at_trigger",
                                "stock_stop_loss_atr",
                                "stock_take_profit_atr",
                                "algo_strategy",
                                "stock_previous_triggers",
                                "tweet_post"
                                ]:
        quick_explore[variable] = eval(variable)
                  
    # st.write(quick_explore)
    
    if (is_summary):
        return_snapshot = quick_explore
    else: return_snapshot = df_four_three_one_soldiers
    # st.write(">>>>>>>>>>>strategy 4-3-1 >>>>>>>>>> return_snapshot",type(return_snapshot))
    return return_snapshot
      

async def trading_signals_view(symbol, 
                          algo_strategy,
                          stock_df, 
                          df_pos, 
                          previous_triggers,
                          selected_period, 
                          selected_interval,
                          short_window_col, 
                          long_window_col,
                          is_summary,
                          ):
    # Collate high level stats on the data
    quick_explore = {}

    quick_explore_df = pd.DataFrame() 
    etf_info = pd.DataFrame()
    etf_data = {} # dictionary
    await asyncio.sleep(1)
    
    if (~df_pos.empty & ~previous_triggers.empty):  
        # put this to a new tab on click of the button
        # ###################################################
        # st.write((stock_df.sort_index(ascending=False)[:10])) 
        
        etf_data[symbol] = stock_df
        # previous_triggers = previous_triggers.sort_values(by='index', ascending=False)
        previous_triggers_list = previous_triggers.index.strftime('%Y/%m/%d %H:%M')
        previous_triggers_list = np.sort(previous_triggers_list)[::-1]
        # print(previous_triggers_list)
        
        # stock_day_close = get_current_price(symbol, selected_period, selected_interval)
        stock_price_at_trigger = df_pos.loc[df_pos.index == df_pos.index.max(), "Close"].to_list()[0]
        stock_trigger_at = df_pos.index.max()
        stock_trigger_state = df_pos.loc[df_pos.index == df_pos.index.max(), "Position"].to_list()[0]
        
        # (buy order) profit order is + if trigger is Buy; loss order is - if trigger is Buy 
        # (sell order) profit order is - if trigger is Sell; loss order is + if trigger is Buy 
        
        if (stock_trigger_state == "Buy"):
            stock_stop_loss_atr = stock_price_at_trigger - st.session_state.stop_loss_factor * (df_pos.loc[(df_pos.index == df_pos.index.max()), "atr_ma"]).to_list()[0]
            stock_take_profit_atr = (stock_price_at_trigger + st.session_state.take_profit_factor * (df_pos.loc[(df_pos.index == df_pos.index.max()), "atr_ma"])).to_list()[0]
            
            etf_data[symbol]['stop_loss_atr'] = stock_df.Close - st.session_state.stop_loss_factor * stock_df.atr_ma
            etf_data[symbol]['take_profit_atr'] = stock_df.Close + st.session_state.take_profit_factor * stock_df.atr_ma
        
        elif (stock_trigger_state == "Sell"):
            stock_stop_loss_atr = (stock_price_at_trigger + st.session_state.stop_loss_factor * (df_pos.loc[(df_pos.index == df_pos.index.max()), "atr_ma"])).to_list()[0]
            stock_take_profit_atr = (stock_price_at_trigger - st.session_state.take_profit_factor * (df_pos.loc[(df_pos.index == df_pos.index.max()), "atr_ma"])).to_list()[0]
            
            etf_data[symbol]['stop_loss_atr'] = stock_df.Close + st.session_state.stop_loss_factor * stock_df.atr_ma
            etf_data[symbol]['take_profit_atr'] = stock_df.Close - st.session_state.take_profit_factor * stock_df.atr_ma
        
        
        
        stock_ema_p1 = df_pos.loc[df_pos.index == df_pos.index.max(), short_window_col].to_list()[0]
        stock_ema_p2 = df_pos.loc[df_pos.index == df_pos.index.max(), long_window_col].to_list()[0]
        
        stock_atr_ma = df_pos.loc[(df_pos.index == df_pos.index.max()), "atr_ma"].to_list()[0]
        
        stock_view_details = etf_data[symbol]
        stock_previous_triggers = previous_triggers_list 
        
        #generating tweet for TSLA with trigger state of Buy at 2024-05-10 09:50 at a price of 171.72
        # symbol: buy: trigger time triggr price: SL:stoploss PT: profit target
        tweet_post =  symbol + ': ' + stock_trigger_state + '; ' + str(stock_trigger_at.strftime("%Y-%m-%d %H:%M")) +'; ' + str("{:.2f}".format(stock_price_at_trigger)) +'; SL: ' + str("{:.2f}".format(stock_stop_loss_atr)) +'; PT: ' + str("{:.2f}".format(stock_take_profit_atr))
        
        # st.write(stock_previous_triggers)
        for variable in ["symbol",
                         "stock_trigger_at",
                        "stock_trigger_state",
                        "stock_price_at_trigger",
                        "stock_stop_loss_atr",
                        "stock_take_profit_atr",
                        "stock_atr_ma",
                        "stock_ema_p1",
                        "stock_ema_p2",
                        "algo_strategy",
                        "tweet_post",
                        "stock_previous_triggers"
                        ]:
            quick_explore[variable] = eval(variable)
            x = pd.DataFrame([quick_explore])
        
        quick_explore_df = pd.concat([x, quick_explore_df], ignore_index=True)
    quick_explore_df = quick_explore_df.sort_values(by = ['stock_trigger_at'], ascending=False)
    # quick_explore_df = quick_explore_df.set_index(['symbol'])
    # print(quick_explore_df)
    
    await asyncio.sleep(1)
    # st.subheader("stock_view_details")
    # st.write(stock_view_details.sort_index(ascending=False))
    
    if (is_summary):
        return_snapshot = quick_explore
    else: return_snapshot = stock_view_details
    
    return return_snapshot
    # return quick_explore #quick_explore_df

# ##########################################################  
# Purpose: 
# ##########################################################
def MovingAverageCrossStrategy(symbol, 
                               stock_df,
                               short_window,
                               long_window, 
                               moving_avg, 
                               display_table = True):
    
    '''
    The function takes the stock symbol, time-duration of analysis, 
    look-back periods and the moving-average type(SMA or EMA) as input 
    and returns the respective MA Crossover chart along with the buy/sell signals for the given period.
    '''
    # stock_symbol - (str)stock ticker as on Yahoo finance. Eg: 'ULTRACEMCO.NS' 
    # start_date - (str)start analysis from this date (format: 'YYYY-MM-DD') Eg: '2018-01-01'
    # end_date - (str)end analysis on this date (format: 'YYYY-MM-DD') Eg: '2020-01-01'
    # short_window - (int)lookback period for short-term moving average. Eg: 5, 10, 20 
    # long_window - (int)lookback period for long-term moving average. Eg: 50, 100, 200
    # moving_avg - (str)the type of moving average to use ('SMA' or 'EMA')
    # display_table - (bool)whether to display the date and price table at buy/sell positions(True/False)

    df_pos = pd.DataFrame()
    previous_triggers = pd.DataFrame()
    
    if moving_avg == 'SMA':
        # column names for long and short moving average columns
        short_window_col = str(short_window) + '_' + moving_avg
        long_window_col = str(long_window) + '_' + moving_avg
        
        # ema_period1
          
        # Create a short simple moving average column
        stock_df[short_window_col] = stock_df['Close'].rolling(window = short_window, min_periods = 1).mean()

        # Create a long simple moving average column
        stock_df[long_window_col] = stock_df['Close'].rolling(window = long_window, min_periods = 1).mean()
        
        # create a new column 'Signal' such that if faster moving average is greater than slower moving average 
        # then set Signal as 1 else 0.
        stock_df['Signal'] = 0.0  
        stock_df['Signal'] = np.where(stock_df[short_window_col] > stock_df[long_window_col], 1.0, 0.0) 

        # create a new column 'Position' which is a day-to-day difference of the 'Signal' column. 
        stock_df['Position'] = stock_df['Signal'].diff()

    elif (moving_avg == 'EMA' or moving_avg == 'EMA 1-2 candle price'):
        
        # column names for long and short moving average columns
        short_window_col = str(short_window) + '_' + moving_avg
        long_window_col = str(long_window) + '_' + moving_avg
        
        # Create short exponential moving average column
        stock_df[short_window_col] = stock_df['Close'].ewm(span = short_window, adjust = True).mean()
        
        # Create a long exponential moving average column
        stock_df[long_window_col] = stock_df['Close'].ewm(span = long_window, adjust = True).mean()
        
        # calculate the stop loss / stop profit
        # Determine Stop-Loss Order
        # A stop-loss order is a request to a broker to sell stocks at a certain price. 
        # These orders aid in minimizing an investor’s loss in a security position.
        
        # create a new column 'Signal' such that if faster moving average is greater than slower moving average 
        # then set Signal as 1 else 0.
        stock_df['Signal'] = 0.0  
        stock_df['Signal'] = np.where(stock_df[short_window_col] > stock_df[long_window_col], 1.0, 0.0) 

        # create a new column 'Position' which is a day-to-day difference of the 'Signal' column. 
        stock_df['Position'] = stock_df['Signal'].diff()
        
    
    # ########################################
    # plot close price, short-term and long-term moving averages
    # https://towardsdatascience.com/making-a-trade-call-using-simple-moving-average-sma-crossover-strategy-python-implementation-29963326da7a
    # ########################################
    df_pos = pd.DataFrame()
    previous_triggers = pd.DataFrame()
    
    stock_df, buy_short, sell_long = calculate_atr_buy_sell(stock_df)
    #   RK_0509: corrected the 'Position parameter of the trading signal 
    stock_df['Position_c'] = np.where(stock_df['Position']==1, 'Buy', 
                                     np.where(stock_df['Position']==-1, 'Sell', None))
  
    
    if display_table == True:
        df_pos = stock_df[(stock_df['Position'] == 1) | (stock_df['Position'] == -1)]
        df_pos['Position'] = df_pos['Position'].apply(lambda x: 'Buy' if x == 1 else 'Sell')
        
        previous_triggers = df_pos[['Position']][-6:]
        
    
    # # #########################
    # # BEGIN: DEBUG_INFO
    # st.write(symbol)
    # st.write("base data")
    # # stock_df = stock_df.reset_index()
    # # stock_df.Datetime = stock_df.Datetime.dt.strftime('%Y/%m/%d %H:%M')
    # # stock_df.index = stock_df.index.strftime('%Y/%m/%d %H:%M')
    # st.write(stock_df.sort_index(ascending=False)[:10])
    # # stock_df = stock_df.set_index('Datetime')
    
    # # END: DEBUG_INFO
    # # #########################
    return stock_df, df_pos, previous_triggers, short_window_col, long_window_col

# ##########################################################  
# Purpose: Function to calculate the Average True Range (ATR)
# For each time period (bar), the true range is simply the greatest of the three price differences:
# High - Low
# High - Previous close
# Previous close - Low
# ########################################################## 
def calculate_atr(data, window=14):
    high_low = data['High'] - data['Low']
    high_close = abs(data['High'] - data['Close'].shift())
    low_close = abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=window).mean()
    
    return atr
    
# ##########################################################  
# Purpose:  Calculate Average True Range (ATR) and its moving average
# ##########################################################  
def calculate_atr_buy_sell(data):
  data['atr'] = calculate_atr(data)
  data['atr_ma'] = data['atr'].rolling(window=14).mean()  # 14-day moving average of ATR

  # NOT IN USE
  # Define buy and sell signals
  buy_signal = (data['atr'] > data['atr_ma']) & (data['atr'].shift(1) <= data['atr_ma'].shift(1))
  sell_signal = (data['atr'] < data['atr_ma']) & (data['atr'].shift(1) >= data['atr_ma'].shift(1))
  
  buy_long_idx = data.index[buy_signal]
  sell_short_idx = data.index[sell_signal]
  
  buy_long = buy_signal.loc[buy_signal==True]
  sell_short = sell_signal.loc[sell_signal==True]
  
  return data, buy_long, sell_short

# ##########################################################  
# Purpose: 
# ##########################################################
def get_current_price(symbol, selected_period, selected_interval):
    try:
      ticker = yf.Ticker(symbol)
      todays_data = ticker.history(period = selected_period, interval = selected_interval)
      
    except:
      print("unable to load the ticker current price") 
      return 
    # getting error: IndexError: single positional indexer is out-of-bounds
    return todays_data['Close'].iloc[-1]

async def ema_continual(symbol, 
                  stock_df,
                  short_window,
                  long_window, 
                  moving_avg, 
                  # display_table = True
                  ):
    
    # column names for long and short moving average columns
    short_window_col = str(short_window) + '_' + moving_avg
    long_window_col = str(long_window) + '_' + moving_avg
        
    # #########################
    # BEGIN: DEBUG_INFO
    # st.write(symbol)
    
    await asyncio.sleep(1)
    
    ema_continual_df = stock_df
    ema_continual_df['ema_5above8'] = (ema_continual_df[short_window_col] > ema_continual_df[long_window_col])
    
    ema_continual_df['t0_close_aboveema5'] = ((ema_continual_df['Close'].shift(1) > ema_continual_df[short_window_col]) &
                                       ((ema_continual_df['Low'].shift(1) < ema_continual_df[short_window_col]) |
                                       (ema_continual_df['Low'].shift(1) < ema_continual_df[long_window_col])) &
                                       (ema_continual_df['Close'].shift(1) <  ema_continual_df['High'].shift(1)) & 
                                       (ema_continual_df['Close'].shift(1) <  ema_continual_df['High'].shift(2)))
    
    ema_continual_df['t0_low_belowema5'] = (((ema_continual_df['Low'].shift(2) < ema_continual_df[short_window_col]) |
                                       (ema_continual_df['Low'].shift(2) < ema_continual_df[long_window_col])) &
                                       (ema_continual_df['High'].shift(2) > ema_continual_df[short_window_col]))
    
    ema_continual_df['ema_continual_long'] = ((ema_continual_df[short_window_col] > ema_continual_df[long_window_col]) & #Ema 5 is above Ema  8
                                      # Last candle (C0) closes above ema5 with low below ema 5 or ema 8 (green candle) and 
                                      # close of C0 candle is less than high of the last two candles
                                      ((ema_continual_df['Close'].shift(1) > ema_continual_df[short_window_col]) &
                                       ((ema_continual_df['Low'].shift(1) < ema_continual_df[short_window_col]) |
                                       (ema_continual_df['Low'].shift(1) < ema_continual_df[long_window_col])) &
                                       (ema_continual_df['Close'].shift(1) <  ema_continual_df['High'].shift(1)) & 
                                       (ema_continual_df['Close'].shift(1) <  ema_continual_df['High'].shift(2))) &
                                      # Low of Candle before C0 (C1) < ema 5 or <  ema 8 
                                      # with high above ema 5(red candle) 
                                      (((ema_continual_df['Low'].shift(2) < ema_continual_df[short_window_col]) |
                                       (ema_continual_df['Low'].shift(2) < ema_continual_df[long_window_col])) &
                                       (ema_continual_df['High'].shift(2) > ema_continual_df[short_window_col]))
                                      )
    
    ema_continual_df['ema_5below8'] = (ema_continual_df[short_window_col] < ema_continual_df[long_window_col]) 
                               
                                      # Last candle (C0) closes above ema5 with low below ema 5 or ema 8 (green candle) and 
                                      # close of C0 candle is less than high of the last two candles
    ema_continual_df['t0_close_belowema5'] = ((ema_continual_df['Close'].shift(1) < ema_continual_df[short_window_col]) &
                                       ((ema_continual_df['High'].shift(1) > ema_continual_df[short_window_col]) |
                                       (ema_continual_df['High'].shift(1) > ema_continual_df[long_window_col])) &
                                       (ema_continual_df['Close'].shift(1) >  ema_continual_df['Low'].shift(1)) & 
                                       (ema_continual_df['Close'].shift(1) >  ema_continual_df['Low'].shift(2)))
    
                                      # Low of Candle before C0 (C1) < ema 5 or <  ema 8 
                                      # with high above ema 5(red candle) 
    ema_continual_df['t0_low_aboveema5'] =  (((ema_continual_df['High'].shift(2) > ema_continual_df[short_window_col]) |
                                       (ema_continual_df['High'].shift(2) > ema_continual_df[long_window_col])) &
                                       (ema_continual_df['Low'].shift(2) < ema_continual_df[short_window_col]))
                                      
    
    ema_continual_df['ema_continual_short'] = ((ema_continual_df[short_window_col] < ema_continual_df[long_window_col]) & #Ema 5 is above Ema  8
                                      # Last candle (C0) closes above ema5 with low below ema 5 or ema 8 (green candle) and 
                                      # close of C0 candle is less than high of the last two candles
                                      ((ema_continual_df['Close'].shift(1) < ema_continual_df[short_window_col]) &
                                       ((ema_continual_df['High'].shift(1) > ema_continual_df[short_window_col]) |
                                       (ema_continual_df['High'].shift(1) > ema_continual_df[long_window_col])) &
                                       (ema_continual_df['Close'].shift(1) >  ema_continual_df['Low'].shift(1)) & 
                                       (ema_continual_df['Close'].shift(1) >  ema_continual_df['Low'].shift(2))) &
                                      # Low of Candle before C0 (C1) < ema 5 or <  ema 8 
                                      # with high above ema 5(red candle) 
                                      (((ema_continual_df['High'].shift(2) > ema_continual_df[short_window_col]) |
                                       (ema_continual_df['High'].shift(2) > ema_continual_df[long_window_col])) &
                                       (ema_continual_df['Low'].shift(2) < ema_continual_df[short_window_col]))
                                      )
    
    #   RK_0509: corrected the 'Position parameter of the trading signal 
    ema_continual_df['position'] = np.where(ema_continual_df['ema_continual_long'], 'Buy', 
                                     np.where(ema_continual_df['ema_continual_short'], 'Sell', None))
    # st.write("EMA 1-2 candle price continuation - LONG")
    # st.write(stock_df.sort_index(ascending=False)[['High', 'Low', 'Close', 
    #    '5_EMA 1-2 candle price continuation',
    #    '8_EMA 1-2 candle price continuation', 'Signal', 'Position', 
    #    'ema_5above8','t0_close_aboveema5','t0_low_belowema5','ema_continual_long']])
    
    # st.write("EMA 1-2 candle price continuation - SHORT")
    # st.write(stock_df.sort_index(ascending=False)[['High', 'Low', 'Close', 
    #    '5_EMA 1-2 candle price continuation',
    #    '8_EMA 1-2 candle price continuation', 'Signal', 'Position', 
    #    'ema_5below8','t0_close_belowema5','t0_low_aboveema5','ema_continual_short']])
    
    # ((stock_df['Close'].shift(4) > stock_df['Close'].shift(3)) &
    #           (stock_df['Close'].shift(3) > stock_df['Close'].shift(2)) &
    #           (stock_df['Close'].shift(2) < stock_df['Close'].shift(1))
    #           )
    
    # END: DEBUG_INFO
    # #########################
    return ema_continual_df

def candle_four_three_one_soldiers(df, is_sorted) -> pd.Series:
  """
  
  # for long
  # close of 4th greater than close of 3rd
  # close of 3rd greater than close of 2nd -
  # close of 2nd less than close of 1st
  
  # for short
  # close of 4th less than close of 3rd
  # close of 3rd less than close of 2nd -
  # close of 2nd higher than close of 1st

  """
#   if(~is_sorted):
#     df = df.sort_index(ascending = False)
#   st.write(df.head())
  # Fill NaN values with 0
  df = df.fillna(0)
  # print(df.head())
  df_evaluate = df[['Open','Close', 'High', 'Low']]
  df_evaluate['t3'] = df_evaluate['Close'].shift(4)
  df_evaluate['t2'] = df_evaluate['Close'].shift(3)
  df_evaluate['t1'] = df_evaluate['Close'].shift(2)
  df_evaluate['t0'] = df_evaluate['Close'].shift(1)
  
  df_evaluate = df_evaluate.fillna(0)
        
  # for long
  # close of 4th greater than close of 3rd
  # close of 3rd greater than close of 2nd -
  # close of 2nd less than close of 1st
  df_evaluate['strategy_431_long_c1'] = (df['Close'].shift(4) > df['Close'].shift(3))
  df_evaluate['strategy_431_long_c2'] = (df['Close'].shift(3) > df['Close'].shift(2))
  df_evaluate['strategy_431_long_c3'] = (df['Close'].shift(2) < df['Close'].shift(1))
  
  df_evaluate['strategy_431_long'] = ((df['Close'].shift(4) > df['Close'].shift(3)) &
              (df['Close'].shift(3) > df['Close'].shift(2)) &
              (df['Close'].shift(2) < df['Close'].shift(1))
              )
  
  # for short
  # close of 4th less than close of 3rd
  # close of 3rd less than close of 2nd -
  # close of 2nd higher than close of 1st

  df_evaluate['strategy_431_short_c1'] = (df['Close'].shift(4) < df['Close'].shift(3)) 
  df_evaluate['strategy_431_short_c2'] = (df['Close'].shift(3) < df['Close'].shift(2))
  df_evaluate['strategy_431_short_c3'] = (df['Close'].shift(2) > df['Close'].shift(1))
              
  df_evaluate['strategy_431_short'] = ((df['Close'].shift(4) < df['Close'].shift(3)) &
              (df['Close'].shift(3) < df['Close'].shift(2)) &
              (df['Close'].shift(2) > df['Close'].shift(1))
              )
  
#   RK_0509: corrected the 'Position parameter of the trading signal 
  df_evaluate['position'] = np.where(df_evaluate['strategy_431_long'], 'Buy', 
                                     np.where(df_evaluate['strategy_431_short'], 'Sell', None))
  
#   st.write("df_evaluate['corrected_position']")
#   st.write("---")
#   st.write(df_evaluate[['Close', 't3', 't2', 't1', 't0',
#        'strategy_431_long_c1', 'strategy_431_long_c2', 'strategy_431_long_c3',
#        'strategy_431_long', 'strategy_431_short_c1', 'strategy_431_short_c2',
#        'strategy_431_short_c3', 'strategy_431_short', 
#        'position',
       
#        ]].sort_index(ascending=False))
#   st.write("---")
#   st.write("---")
  
  df_evaluate, buy_short, sell_long = calculate_atr_buy_sell(df_evaluate)
  
  if any(df_evaluate['position'] == "Buy"):
      df_evaluate['stop_loss_atr'] = df_evaluate.Close - st.session_state.stop_loss_factor * df_evaluate.atr_ma
      df_evaluate['take_profit_atr'] = df_evaluate.Close + st.session_state.take_profit_factor * df_evaluate.atr_ma
    
  elif any(df_evaluate['position'] == "Sell"):
      df_evaluate['stop_loss_atr'] = df_evaluate.Close + st.session_state.stop_loss_factor * df_evaluate.atr_ma
      df_evaluate['take_profit_atr'] = df_evaluate.Close - st.session_state.take_profit_factor * df_evaluate.atr_ma
  
#   st.write("df_evaluate['position']")
#   st.write("---")
#   st.write(df_evaluate.sort_index(ascending=False))
#   st.write("---")
#   st.write("---")
  return df_evaluate
    
async def strategy_four_three_one_soldiers(symbol,
                                 df,
                                 selected_period, 
                                 selected_interval,
                                 algo_strategy = "4-3-1 candle price reversal",):
    # st.write(df.head())
    
    df_strategy_431 = candle_four_three_one_soldiers(df, False)
    # df_strategy_431 = df
    
    await asyncio.sleep(1)
    # st.write("filtered data - strategy_431_long")
    df_strategy_431_long = (df_strategy_431[df_strategy_431["strategy_431_long"] == True])
    # st.write(df_strategy_431_long.sort_index(ascending=False))
    
    # st.write("filtered data - strategy_431_short")
    df_strategy_431_short = (df_strategy_431[df_strategy_431["strategy_431_short"] == True])
    # st.write(df_strategy_431_short.sort_index(ascending=False))
    
    # stock_price_at_trigger = df_strategy_431_long.loc[df_strategy_431_long.index.max(), "Close"].to_list()[0]
    # stock_trigger_at = df_pos.index.max()
    # stock_trigger_state = df_pos.loc[df_pos.index == df_pos.index.max(), "Position"].to_list()[0]
    
    # # lets send all triggers instead of the top one
    # df_summary = df_strategy_431_long[df_strategy_431_long.index == df_strategy_431_long.index.max()]
    # df_summary_short = df_strategy_431_short[df_strategy_431_short.index == df_strategy_431_short.index.max()]
    # df_summary = pd.concat([df_summary, df_summary_short], ignore_index=False)
    
    df_summary = df_strategy_431_long #[df_strategy_431_long.index == df_strategy_431_long.index.max()]
    df_summary_short = df_strategy_431_short #[df_strategy_431_short.index == df_strategy_431_short.index.max()]
    df_summary = pd.concat([df_summary, df_summary_short], ignore_index=False)
    
    # st.write("df_summary",df_summary)
    # return df_strategy_431 #df_summary
    return df_summary #returns the filtered data only
    
 
