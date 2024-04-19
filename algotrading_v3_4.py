# -*- coding: utf-8 -*-
"""AlgoTrading_v3.2_panel.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18MuK4_G2Nf8oow21NW_a3pHg_35JgVSI
"""
from algotrading_helper import *
from algotrading_visualisations import *
from pathlib import Path

pd.options.display.float_format = '${:,.2f}'.format

def main():
     
  # """### Select Stock and Time interval"""
  # https://github.com/smudali/stocks-analysis/blob/main/dasboard/01Home.py
  new_ticker_list = []
  
  base_symbol_list = ["PLTR","TSLA","NVDA","AMZN", "NFLX","BA","GS","SPY","QQQ","IWM","SMH","RSP"]
  # if (ticker_list.isna()):
  #   ticker_list = []
  
  # using set() to remove duplicated from list
  symbol_list = base_symbol_list + new_ticker_list
  
  
  # symbol_list = list(set(symbol_list))
  
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
      'Select Moving Average Strategy', options=['SMA', 'EMA'], index=1)
  selected_short_window =  st.sidebar.number_input(":gray[Short Window]", step = 1, value=5)  
  selected_long_window =  st.sidebar.number_input(":gray[Long Window]", step = 1, value=8)   

  #         Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
  # Either Use period parameter or use start and end
  #     interval : str
  #         Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo

  ema_period1 = selected_short_window
  ema_period2 = selected_long_window

  
  
  
  # all_symbols  = " ".join(known_options)
  # print(known_options)
  # print(all_symbols)
  # show_snapshot(all_symbols)
  
  
  # print(show_snapshot(known_options))
  
  if len(known_options) == 0:
    st.write ("Please select a ticker in the sidebar")
    return
  else:
    tab = st.tabs(["Summary","🗃 List View","📈 Visualisations", "🗃 Details", "Customization", "Release Notes"])
    # ###################################################
    # Summary: 
    # # of stocks being watched; 
    # Algo being used
    # # of winining vs losing trades
    # # best stocks
    # ###################################################
    with tab[0]:    
      # creates the container for page title
      dash_1 = st.container()

      with dash_1:
          st.markdown("<h3 style='text-align: center;'>You are watching</h3>", unsafe_allow_html=True)
          st.write("")
          
          # get basics
          selected_stocks = len(known_options) 

          col1, col2, col3, col4 = st.columns(4)
          # create column span
          col1.metric(label="No. Stocks Watch", value= selected_stocks , delta=None)
          col2.metric(label="Period", value= selected_period , delta=None)
          col3.metric(label="Interval", value= selected_interval , delta=None)
          # trading_strategy = str(algo_strategy) + '_' + str(selected_short_window) + '_' + str(selected_long_window) + '_crossover'
          col4.metric(label="Trading Strategy", value= algo_strategy , delta=None)
        
      st.divider()
       
      dash_2 = st.container()
      with dash_2:
        
          title = "Moving Average Strategy: " + algo_strategy
          st.subheader(title)
          st.divider()
          
          # Collate high level stats on the data
          quick_explore = {}
          
          quick_explore_df = pd.DataFrame() 
          etf_info = pd.DataFrame()
          etf_data = {} # dictionary
                   
          for symbol in known_options:
              # st.subheader(symbol)
              stock_name =  symbol
              yf_data = yf.Ticker(symbol) #initiate the ticker
              stock_hist_df = get_hist_info(yf_data, selected_period, selected_interval)
              stock_df, df_pos, previous_triggers, buy_short, sell_long = MovingAverageCrossStrategy(symbol,
                                        stock_hist_df,
                                        selected_short_window,
                                        selected_long_window,
                                        algo_strategy,
                                        True)
              
              # print("stock_df, df_pos")
              # print(stock_df, df_pos)
              # st.subheader(symbol)
              # st.subheader("Stock history")
              # st.caption(df_pos.index.max())
              
              # put this to a new tab on click of the button
              # ###################################################
              # st.write((stock_df.sort_index(ascending=False)[:10])) 
              
              etf_data[symbol] = stock_df
              
              # previous_triggers_list = previous_triggers
              previous_triggers = previous_triggers.reset_index()
              
              # st.write(previous_triggers)
              # etf_info.loc[etf_info['symbol'] == symbol, 'last_12_months_Open'] = (spark_img_url)
              # etf_data.loc[etf_data['symbol'] == symbol, 'last_12_months_Open'] = (spark_img_url)
   
              # etf_info = etf_info.drop(columns=['index']   )
              # etf_info_df = pd.DataFrame.from_dict(etf_info)
              
              # st.write(etf_info_df) #.to_html(render_links=True))
              # ###################################################
              
              # df_atr, buy_signal, sell_signal = calculate_atr_buy_sell(stock_df)
              
              # st.write(buy_signal.max())
              # st.write(df_pos.index.max())
              
              # st.write (sell_signal.max())
              # st.write(df_pos.index.max())
              
              # st.write(df[df_pos.index.max().isin(buy_signal.max())])
              
              # st.write(df.loc[np.isin(df_pos.index, buy_signal)])
              
              stock_day_close = get_current_price(symbol, selected_period, selected_interval)
              stock_price_at_trigger = df_pos.loc[df_pos.index == df_pos.index.max(), "Close"].to_list()[0]
              stock_trigger_at = df_pos.index.max()
              stock_trigger_state = df_pos.loc[df_pos.index == df_pos.index.max(), "Position"].to_list()[0]
              stock_stop_loss = df_pos.loc[df_pos.index == df_pos.index.max(), "stop_loss"].to_list()[0]
              stock_take_profit = df_pos.loc[df_pos.index == df_pos.index.max(), "stop_profit"].to_list()[0]
              stock_atr = df_pos.loc[(df_pos.index == df_pos.index.max()), "atr_ma"].to_list()[0]
              stock_view_details = etf_data[symbol]
              stock_previous_triggers = previous_triggers.Datetime.astype(str).to_list() #df_pos.Position[:6]#.to_list()
              
              # # print(df_pos.info())
              # print("buy_signal.index")
              # print((buy_signal.sort_index(ascending=False)))
              # # print("sell_signal.index")
              # print(sell_signal.sort_index(ascending=False))
              # print("df_pos.index")
              # # print(df_pos.index.name)
              # st.write(df_pos.sort_index(ascending=False))
              
              # print(df_atr.loc[df_atr.index == df_pos.index.max()])
              # # print(df_atr.loc[df_atr.index.isin(buy_signal)])
              # print(buy_signal.loc[buy_signal==True])
              
              # print(df_atr[[df_atr.index == df_pos.index.max() ]])
              # print()
              
              # stock_take_profit_atr = df_pos.loc[df_pos.index ==  df_pos.index.max(), "atr_ma"].to_list()[0]
              # print(stock_stop_loss_atr)
              # print(stock_take_profit_atr)
              
              
              # st.write(stock_previous_triggers)
              for variable in ["symbol",
                              "stock_take_profit",
                              "stock_stop_loss",
                              # "stock_take_profit_atr",
                              # "stock_stop_loss_atr",
                              "stock_atr",
                              "stock_price_at_trigger",
                              "stock_trigger_state",
                              "stock_trigger_at",
                              "stock_previous_triggers"
                              # "stock_view_details"
                              ]:
                quick_explore[variable] = eval(variable)
              # print(quick_explore)  
              #x = pd.DataFrame.from_dict(quick_explore, orient = 'index')
              x = pd.DataFrame([quick_explore])
              #print("x\n", x)
                
              # quick_explore_df = quick_explore_df.append(x)
              quick_explore_df = pd.concat([x, quick_explore_df], ignore_index=True)
          quick_explore_df = quick_explore_df.sort_values(by = ['stock_trigger_at'], ascending=False)
          
          st.data_editor(
          quick_explore_df,
          column_config={"stock_take_profit": st.column_config.NumberColumn(
              "Take-Profit Order",
              format="%.3f",
          ),
                         "stock_stop_loss": st.column_config.NumberColumn(
              "Stop-Loss Order",
              format="%.3f",
          ),
                         "stock_atr": st.column_config.NumberColumn(
              "Order (ATR)",
              format="%.3f",
          ),
                         "stock_price_at_trigger": st.column_config.NumberColumn(
              "Stock Price at Trigger",
              format="%.3f",
          ),
                         "stock_previous_triggers": st.column_config.ListColumn(
              "Previous Triggers at",
              width=None,
          ),
              # "stock_view_details": st.column_config.LinkColumn
              # (
              #     "Stock Details",
              #     help="The top trending Streamlit apps",
              #     max_chars=100,
              #     display_text="view table",
              #     # default=add_container(etf_data[symbol], quick_explore_df[symbol])
              # ),
              
          },
          hide_index=True,
          )
  
          st.divider()
          
        
    # ###################################################
    # List View: 
    # # of all stocks; 
    # ###################################################
    with tab[1]:
      
      # get stock metrics
        
      for symbol in known_options:
        st.subheader(symbol)
        try:
          yf_data = yf.Ticker(symbol) #initiate the ticker
          etf_summary_info = get_all_stock_info(yf_data)
          stock_caption = ("exchange: "+etf_summary_info.exchange[0]
                      + "; currency: "+etf_summary_info.currency[0])
                  
          st.caption(stock_caption)
          
          stock_hist_df = get_hist_info(yf_data, selected_period, selected_interval)
        
          stock_hist_df, df_pos, previous_triggers, buy_short, sell_long = MovingAverageCrossStrategy(symbol,
                                      stock_hist_df,
                                      selected_short_window,
                                      selected_long_window,
                                      algo_strategy,
                                      True)  
          
          df_pos.reset_index(inplace=True)
          df_pos = df_pos.sort_index(ascending=False)
          
          buy_trigger = len(df_pos[df_pos['Position']=='Buy'])
          sell_trigger = len(df_pos[df_pos['Position']=='Sell'])
          # print(buy_trigger, sell_trigger)
         
          col1, col2, col3, col4 = st.columns(4)
          
          with st.container(): # chart_container(chart_data):
            toast_message = (":red["
                              +"Fetching information for " 
                              + etf_summary_info.shortName[0] 
                              + " "+ symbol
                              +"]"
                      )
            st.toast(toast_message, icon='🏃')  
            # time.sleep(1)            
            col1.metric(label="Close", value= etf_summary_info.previousClose , delta=None)
            col1.metric(label="Open", value= etf_summary_info.open , delta=None) 
            col2.metric(label="Day Low", value= etf_summary_info.dayLow)
            col2.metric(label="Day High", value= etf_summary_info.dayHigh)
            col3.metric(label="52 week Low", value= etf_summary_info.fiftyTwoWeekLow)
            col3.metric(label="52 week High", value= etf_summary_info.fiftyTwoWeekHigh)
            col4.metric(label="Buy (period)", value= buy_trigger)
            col4.metric(label="Sell (period)", value= sell_trigger)
            
            expander = st.expander("Ticker trading prompts")
            # # expander.write(\"\"\"
            # #     The chart above shows some numbers I picked for you.
            # #     I rolled actual dice for these, so they're *guaranteed* to
            # #     be random.
            # # \"\"\")
            # # expander.image("https://static.streamlit.io/examples/dice.jpg")
            # expander.write("the last 10 records")
            days=1  
            # print(df_pos.columns)  
            cutoff_date = df_pos['Datetime'].iloc[0] - pd.Timedelta(days=days)
            # print ("cutoff_date")
            # print (cutoff_date)
          
            df1 = df_pos[df_pos['Datetime'] > cutoff_date]
            # print ("state till cutoff_date")
            expander.write(df1[['Datetime','Close', 'Position']]) 
        except:
          print('Error loading stock data for ' + symbol)
          return None 
   
        
    # ###################################################
    # Charts: 
    # # of stocks being watched; 
    # ###################################################
    with tab[2]:    
      for symbol in known_options:
        yf_data = yf.Ticker(symbol) #initiate the ticker
        
        st.session_state.page_subheader = '{0} ({1})'.format(yf_data.info['shortName'], yf_data.info['symbol'])
        st.subheader(st.session_state.page_subheader)
        # st.write(yf_data)
        # st.write(symbol)
        st.write("Historical data per period (Showing EMA-5day period vs EMA-10day period)")
        
        # st.write("(Showing EMA-5day period vs EMA-10day period)")
        stock_hist_df = get_hist_info(yf_data, selected_period, selected_interval)
        
        # use streamlit light weight charts
        lw_charts_snapshot(symbol, 
                           stock_hist_df, 
                           algo_strategy,
                           selected_short_window,
                           selected_long_window,
                           False)
        
       
        st.divider()

    # ###################################################
    # Details: 
    # Details of all stocks individually being watched; 
    # ###################################################
    with tab[3]:    
      st.subheader("News on the selected stocks")
      # for symbol in known_options:
      #   st.session_state.page_subheader = '{0} ({1})'.format(yf_data.info['shortName'], yf_data.info['symbol'])
      #   st.subheader(st.session_state.page_subheader)
      #   yf_data = yf.Ticker(symbol) #initiate the ticker
      #   stock_news_df = get_stk_news(yf_data)
      #   # st.write(stock_news_df)
      #   st.data_editor(
      #       stock_news_df,
      #       column_config={
      #           "link": st.column_config.LinkColumn(
      #               "News Link", #display_text="Open profile"
      #           ),
      #       },
      #       hide_index=True,
      #   )
      # st.write("News")
      # st.write(stock_news_df.to_html(escape=False, index=True), unsafe_allow_html=True)
      # st.divider()
      
    # ###################################################
    # Volatility Indicators
    # ###################################################
    with tab[4]:    
      st.subheader("Customise Stocks list")
      # new_element = st.text_input("Add a new symbol:", "")
      # symbol_list.append(str(new_element))
      # print(symbol_list)
      
  
      # ticker_list = ""
      # ticker_list = st.text_area(":red[enter the ticker list seperated with commas]"
      #     )

      # st.write(ticker_list)
      # print((type(ticker_list)))
      # st.write(list(ticker_list.split(",")))
      # symbol_list = base_symbol_list + ticker_list
      
      # for symbol in known_options:
      #   yf_data = yf.Ticker(symbol) #initiate the ticker
      #   st.write(symbol)
      #   stock_hist_df = get_hist_info(yf_data, selected_period, selected_interval)
      #   # st.write(stock_hist_df.tail())
      #   df, buy_signal, sell_signal = calculate_atr_buy_sell(stock_hist_df)
      #   st.write(df)
        
        # show_atr(df)
        
        
    

    with tab[5]:
      st.subheader("Change Log")
      st.write("Ability to add more stocks to the existing watchlist from the universe of all stocks allowed by the app.")

  return

if __name__ == '__main__':
  main()

# # https://www.quantstart.com/articles/candlestick-subplots-with-plotly-and-the-alphavantage-api/
