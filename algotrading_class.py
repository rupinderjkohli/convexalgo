import pandas as pd
import numpy as np

# Bullish Candle — Green / Bull / Long CandleStick
# This candle is formed when price opens at let’s say 10 INR and closes above 10 INR (let’s say at 15), 
# this implies during this time frame the bulls (or Buyers) were active and take the price of financial 
# instrument UP.

# Open Price < Close Price
# Lower Wick or shadow - difference(open price, low price)
# Upper Wick or shadow - difference(High price, close price)

# Bearish Candle — Red / Bear / Short CandleStick
# This candle is formed when price opens at let’s say 10 INR and closes below 10 INR (let’s say at 5), 
# this implies during this time frame the bears (or Buyers) were active and bring the price of financial 
# instrument Down.

# Open Price > Close Price
# Lower Wick or shadow -  difference(close price, low price)
# Upper Wick or shadow - difference(High price, open price)

class CandleStick:
    def __init__(self, data: dict):
        self.open = data["Open"]
        self.close = data["Close"]
        self.high = data["High"]
        self.low = data["Low"]
        self.volume = data["Volume"]
        self.date = data["Date"]
        # Some extra details

        self.length = self.high - self.low
        self.bodyLength = abs(self.open - self.close)
        self.lowerWick = self.__get_lower_wick_length()
        self.upperWick = self.__get_upper_wick_length()

    def __repr__(self):
        return (f"CandleStick(open={self.open}, close={self.close},"
                f" high={self.high}, low={self.low}, volume={self.volume}")

    # Bullish Candle — Green / Bull / Long CandleStick
    def is_bullish(self):
        return self.open < self.close
  
    # Bearish Candle — Red / Bear / Short CandleStick
    def is_bearish(self):
        return self.open > self.close

    def __get_lower_wick_length(self):
        """Calculate and return the length of lower shadow or wick."""
        return (self.open if self.open <= self.close else self.close) - self.low

    def __get_upper_wick_length(self):
        """Calculate and return the length of upper shadow or wick."""
        return self.high - (self.open if self.open >= self.close else self.close)
      

import pandas as pd
from pandas.api.types import is_numeric_dtype

# https://github.com/SpiralDevelopment/candlestick-patterns/blob/master/candlestick/patterns/candlestick_finder.py#L5

class CandlestickFinder(object):
    def __init__(self, name, required_count, target=None):
        self.name = name
        self.required_count = required_count
        self.close_column = 'close'
        self.open_column = 'open'
        self.low_column = 'low'
        self.high_column = 'high'
        self.data = None
        self.is_data_prepared = False
        self.multi_coeff = -1

        if target:
            self.target = target
        else:
            self.target = self.name

    def get_class_name(self):
        return self.__class__.__name__

    def logic(self, row_idx):
        raise Exception('Implement the logic of ' + self.get_class_name())

    def has_pattern(self,
                    candles_df,
                    ohlc,
                    is_reversed):
        self.prepare_data(candles_df,
                          ohlc)

        if self.is_data_prepared:
            results = []
            rows_len = len(candles_df)
            idxs = candles_df.index.values

            if is_reversed:
                self.multi_coeff = 1

                for row_idx in range(rows_len - 1, -1, -1):

                    if row_idx <= rows_len - self.required_count:
                        results.append([idxs[row_idx], self.logic(row_idx)])
                    else:
                        results.append([idxs[row_idx], None])

            else:
                self.multi_coeff = -1

                for row in range(0, rows_len, 1):

                    if row >= self.required_count - 1:
                        results.append([idxs[row], self.logic(row)])
                    else:
                        results.append([idxs[row], None])

            candles_df = candles_df.join(pd.DataFrame(results, columns=['row', self.target]).set_index('row'),
                                         how='outer')

            return candles_df
        else:
            raise Exception('Data is not prepared to detect patterns')

    def prepare_data(self, candles_df, ohlc):

        if isinstance(candles_df, pd.DataFrame):

            if len(candles_df) >= self.required_count:
                if ohlc and len(ohlc) == 4:
                    if not set(ohlc).issubset(candles_df.columns):
                        raise Exception('Provided columns does not exist in given data frame')

                    self.open_column = ohlc[0]
                    self.high_column = ohlc[1]
                    self.low_column = ohlc[2]
                    self.close_column = ohlc[3]
                else:
                    raise Exception('Provide list of four elements indicating columns in strings. '
                                    'Default: [open, high, low, close]')

                self.data = candles_df.copy()

                if not is_numeric_dtype(self.data[self.close_column]):
                    self.data[self.close_column] = pd.to_numeric(self.data[self.close_column])

                if not is_numeric_dtype(self.data[self.open_column]):
                    self.data[self.open_column] = pd.to_numeric(self.data[self.open_column])

                if not is_numeric_dtype(self.data[self.low_column]):
                    self.data[self.low_column] = pd.to_numeric(self.data[self.low_column])

                if not is_numeric_dtype(self.data[self.high_column]):
                    self.data[self.high_column] = pd.to_numeric(candles_df[self.high_column])

                self.is_data_prepared = True
            else:
                raise Exception('{0} requires at least {1} data'.format(self.name,
                                                                        self.required_count))
        else:
            raise Exception('Candles must be in Panda data frame type')
        
        
class BearishEngulfing(CandlestickFinder):
    def __init__(self, target=None):
        super().__init__(self.get_class_name(), 2, target=target)

    def logic(self, idx):
        candle = self.data.iloc[idx]
        prev_candle = self.data.iloc[idx + 1 * self.multi_coeff]

        close = candle[self.close_column]
        open = candle[self.open_column]
        high = candle[self.high_column]
        low = candle[self.low_column]

        prev_close = prev_candle[self.close_column]
        prev_open = prev_candle[self.open_column]
        prev_high = prev_candle[self.high_column]
        prev_low = prev_candle[self.low_column]
        
        return (open >= prev_close > prev_open and
                open > close and
                prev_open >= close and 
                open - close > prev_close - prev_open)
        
        # return (prev_close > prev_open and
        #         0.3 > abs(prev_close - prev_open) / (prev_high - prev_low) >= 0.1 and
        #         close < open and
        #         abs(close - open) / (high - low) >= 0.7 and
        #         prev_high < open and
        #         prev_low > close)        