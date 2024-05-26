from algotrading_helper import *
from convexAlgos_standalone import *
from algotrading_algos import *
from algotrading_login import *
import globals

# convexalgos_standalone()
# @st.experimental_fragment(run_every='1m')
def playground_ui(known_options, 
                              selected_algos, 
                              selected_period, 
                              selected_interval):
    
    st.write(st.session_state.last_run)
    selected_etf_data = get_selected_stock_history(known_options,selected_period, selected_interval)
    # st.subheader('Select a Ticker')
    selected_ticker = st.selectbox("Select Ticker",options=known_options,
                                    help = 'Select a ticker', 
                                    key='visualise_ticker',
                                    placeholder="Choose a Ticker",)
    
    # st.subheader('Select a Date Range')
    # st.write(selected_etf_data[selected_ticker])
    df = selected_etf_data[selected_ticker][['Open','Close','High','Low']]
    
    new_algo_strategy = st.selectbox(
    "Select ConvexAlgo strategy to test",
    ["Candle Properties", "Trends","Strategy - Hammer"],
    # captions = ["Candle Properties","Hammer", "Inverted Hammer", "Trends"],
    index = None,)


    if new_algo_strategy == "Strategy - Hammer":
        st.write("Showing results for ***Hammer***")
        df_hammer = strategy_hammer(df)
        st.write(df_hammer.sort_index(ascending=False))
        
    # elif new_algo_strategy == "Inverted Hammer":
    #     st.write("Showing results for ***Inverted Hammer***")
    #     df_inverted_hammer = strategy_hammer(df)

    elif new_algo_strategy == "Trends":
        st.write("Showing results for ***Trends***")
        df_trends = df.copy()
        df_trends['down_trend'] = candles_downtrend(df_trends)
        df_trends['up_trend'] = candles_uptrend(df_trends)
        st.write(df_trends.head(50).sort_index(ascending=False))
        
    elif new_algo_strategy == "Candle Properties":
        app_refresh(selected_interval, "candle_properties")
        st.write("Showing Candle Properties (app_refresh)")
        df_prop = df
        df_prop = candle_properties(df)
        st.write(df_prop.sort_index(ascending=False))
    
        
    
