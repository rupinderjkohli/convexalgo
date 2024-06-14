emaFastLength = 5
emaSlowLength = 9
cciLength = 20

SYMBOLS = ['MSFT']

# 4. Trading Period & Interval
PERIOD = '5d'
INTERVAL = '1m'

# 11. Stop_loss_price / Take_profit_price parameters
STOP_LOSS = 1.75
stop_loss_factor = 1
TAKE_PROFIT = 3
take_profit_factor = 1

# 15. Trading Strategy List
MOVING_AVERAGE_BASED = ['5/8 EMA','5/8 EMA 1-2 candle price']
TREND_BASED = ['4-3-1 candle price reversal']
convex_trade_algos_list = ['algo_list']

MOVING_AVERAGE_WINDOW_SHORT = 5
MOVING_AVERAGE_WINDOW_LONG = 8

social_media_links = [
    "https://www.twitter.com/convextrades",
    "https://www.instagram.com/convex.trades",
    "https://www.facebook.com/convextrades",
  ]

# List of algo functions
algo_functions = ['strategy_sma', 'strategy_ema', 'strategy_ema_continual', 'strategy_431_reversal']

# TODO
algo_functions_args = []
  