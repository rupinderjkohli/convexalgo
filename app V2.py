# https://www.fiverr.com/inbox/debachat

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)

from ibapi.client import *
from ibapi.wrapper import *
from ibapi.contract import Contract

import csv

#Shekhu026
#Cheatcode@26

from indicators import *
import globals

import time
import pandas as pd
import numpy as np

import threading

accountID = ""

class IBapi(EWrapper, EClient):
	def __init__(self):
		EClient.__init__(self, self)
		self.data = []

	def historicalData(self, reqId, bar):
		self.data.append([bar.date, bar.close, bar.high, bar.low, bar.volume, bar.open])

	def historicalDataEnd(self, reqId: int, start: str, end: str):
		super().historicalDataEnd(reqId, start, end)
		#print("HistoricalDataEnd. ReqId:", reqId, "from", start, "to", end)


 
def create_contract(symbol, sec_type, exchange, currency):
    contract = Contract()
    contract.symbol = symbol
    contract.secType = sec_type
    contract.exchange = exchange
    contract.currency = currency
    return contract

def create_order(action, quantity, order_type):
    order = Order()
    order.action = action
    order.totalQuantity = quantity
    order.orderType = order_type
    order.eTradeOnly = False
    order.firmQuoteOnly = False
    #if you have subAccounts, you need that
    if accountID != "":
        order.account  = accountID
    return order

nextID = 1

def placeOrder(contract, order):
    global nextID
    app.placeOrder(nextID, contract, order)
    nextID += 1

def buyStock(symbol, quantity):
    contract = create_contract(symbol, "STK", "SMART", "USD")
    order = create_order("BUY", quantity, "MKT")
    placeOrder(contract, order)

def sellStock(symbol, quantity):
    contract = create_contract(symbol, "STK", "SMART", "USD")
    order = create_order("SELL", quantity, "MKT")
    placeOrder(contract, order)


def run_loop():
	app.run()

app = IBapi()
app.connect('127.0.0.1', 7497, 1)

api_thread = threading.Thread(target=run_loop, daemon=True)
api_thread.start()

file_path = "symbolList.py" 

symbols = []

with open(file_path, 'r') as file:
    for line in file:
        line = line.strip()
        lineSplit = line.split(",")
        symbols.append({'sym': lineSplit[0].strip(), 'qty': lineSplit[1].strip()})


if __name__ == "__main__":
    while True:

        for sym in symbols:
            
            try:
                stock = sym['sym']
                qty = float(sym['qty'])

                print(f'currently doing {stock} with the amount of {qty}')

                contract = Contract()
                contract.symbol = stock
                contract.secType = 'STK'
                contract.exchange = "SMART"
                contract.currency = 'USD'

                # https://interactivebrokers.github.io/tws-api/historical_bars.html#hd_barsize
                app.reqHistoricalData(1, contract, '', "1 M", "5 mins", 'TRADES', 1, 1, False, [])
                time.sleep(1.5) #sleep to allow enough time for data to be returned
                if len(app.data) == 0:
                    continue

                df = pd.DataFrame(app.data, columns=['DateTime', 'Close', 'High', 'Low', 'Volume', 'Open'])

                currClose = df.at[len(df) - 1, 'Close']
                currOpen = df.at[len(df) - 1, 'Open']

                df['cci'] = CCI(df['Close'], globals.cciLength)

                currCCI = df.at[len(df) - 1, 'cci']
                cciOverSold = True if currCCI >= -100 and currCCI <= 0 else False
                cciOverBought = True if currCCI <= 100 and currCCI >= 0 else False

                cciUpTrending = True if currCCI >= 50 else False
                cciDownTrending = True if currCCI <= 50 else False

                df['cciSlope'] = Linreg(df['Close'], 1)
                #print(df['cciSlope'])
                currSlope = df.at[len(df) - 1, 'cciSlope']
                cciSlopeUp = True if currSlope > 0 else False
                cciSlopeDown = True if currSlope < 0 else False

                # EMA with 5/8 cross over
                df['emaFast'] = ema(df['Close'], globals.emaFastLength)
                df['emaSlow'] = ema(df['Close'], globals.emaSlowLength)

                currFast = df.at[len(df) - 1, 'emaFast']
                currSlow = df.at[len(df) - 1, 'emaSlow']
                emaFastCrossedUp = True if currClose > currFast else False
                emaFastCrossedDown = True if currClose < currFast else False

                emaSlowCrossedUp = True if currClose > currSlow else False
                emaSlowCrossedDown = True if currClose < currSlow else False

                # 431
                threeCandlesUp = True if (df.at[len(df) - 2, 'Close'] > df.at[len(df) - 3, 'Close']) and (df.at[len(df) - 3, 'Close'] > df.at[len(df) - 4, 'Close']) else False
                currentCandleDown = True if currClose < (df.at[len(df) - 2, 'Close'] - ((df.at[len(df) - 2, 'Close'] > df.at[len(df) - 2, 'Open']) / 2)) else False    

                threeCandlesDown = True if (df.at[len(df) - 2, 'Close'] < df.at[len(df) - 3, 'Close']) and (df.at[len(df) - 3, 'Close'] < df.at[len(df) - 4, 'Close']) else False
                currentCandleUp = True if currClose > (df.at[len(df) - 2, 'Close'] + ((df.at[len(df) - 2, 'Close'] > df.at[len(df) - 2, 'Open']) / 10)) else False    

                currentCandleRed = True if currClose < currOpen else False
                currentCandleGreen = True if currClose > currOpen else False

                twoCandlesUp = True if df.at[len(df) - 2, 'Close'] >= df.at[len(df) - 3, 'Close'] else False
                currentCandleBearishEngulfing = True if df.at[len(df) - 2, 'Close'] > df.at[len(df) - 2, 'Open'] and currClose < df.at[len(df) - 2, 'Low'] else False

                twoCandlesDown = True if df.at[len(df) - 2, 'Close'] <= df.at[len(df) - 3, 'Close'] else False
                currentCandleBullishEngulfing = True if df.at[len(df) - 2, 'Close'] < df.at[len(df) - 2, 'Open'] and currClose > df.at[len(df) - 2, 'High'] else False

                short3CandleHarami = True if threeCandlesUp and currentCandleDown and cciOverBought and currentCandleRed and emaSlowCrossedDown else False
                long3CandleHarami = True if threeCandlesDown and currentCandleUp and cciOverSold and currentCandleGreen and emaSlowCrossedUp else False

                short2CandleEngulfing = True if twoCandlesUp and currentCandleBearishEngulfing and cciOverBought and currentCandleRed and emaFastCrossedDown else False
                long2CandleEngulfing = True if twoCandlesDown and currentCandleBearishEngulfing and cciOverSold and currentCandleGreen and emaFastCrossedUp else False

                longDip = True if (currClose < currOpen and currClose < df.at[len(df) - 2, 'Close'] and df.at[len(df) - 1, 'Low'] < currFast and currClose >= currSlow and currFast > currSlow and currentCandleBearishEngulfing == False) else False
                shortDip = True if (currClose > currOpen and currClose > df.at[len(df) - 2, 'Close'] and df.at[len(df) - 1, 'High'] > currFast and currClose <= currSlow and currFast < currSlow and currentCandleBullishEngulfing == False) else False

                if short3CandleHarami or short2CandleEngulfing:
                    #sellStock(stock, qty)
                    print("Short")

                if long3CandleHarami or long2CandleEngulfing:
                    #buyStock(stock, qty)
                    print("Long")

                # CCI
                cciCrossUp0 = True if (df.at[len(df) - 1, 'CCI'] > 0) and (df.at[len(df) - 2, 'CCI'] < 0) else False
                cciCrossDown0 = True if (df.at[len(df) - 1, 'CCI'] < 0) and (df.at[len(df) - 2, 'CCI'] > 0) else False
                cciCrossUpMinus100 = True if (df.at[len(df) - 2, 'CCI'] < -100) and (df.at[len(df) - 1, 'CCI'] > -100) else False
                cciCrossDownPlus100 = True if (df.at[len(df) - 2, 'CCI'] > 100) and (df.at[len(df) - 1, 'CCI'] < 100) else False

                cciEmaShort = True if cciCrossDown0 or cciCrossDownPlus100 and emaSlowCrossedDown else False
                cciEmaLong = True if cciCrossUp0 or cciCrossUpMinus100 and emaSlowCrossedUp else False

                if cciEmaShort:
                    sellStock(stock, qty)
                    print("Short")

                if cciEmaLong:
                    buyStock(stock, qty)
                    print("Long")

            except Exception as err:
                print(err)
                print(stock)

            time.sleep(1)