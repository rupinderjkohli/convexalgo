import pandas as pd
import numpy as np

def CCI(df, ndays): 
    sma = df.rolling(ndays).mean()
    mad = df.rolling(ndays).apply(lambda x: pd.Series(x).mad())
    cci = (df - sma) / (0.015 * mad) 
    return cci

def ema(df, len):
    ema1 = df.ewm(span=len).mean()
    return ema1

def Linreg(source, length):
    size = len(source)
    linear = np.zeros(size)

    for i in range(length, size):

        sumX = 0.0
        sumY = 0.0
        sumXSqr = 0.0
        sumXY = 0.0

        for z in range(length):
            val = source.iloc[i - z]
            per = z + 1.0
            sumX += per
            sumY += val
            sumXSqr += per * per
            sumXY += val * per

        slope = (length * sumXY - sumX * sumY) / (length * sumXSqr - sumX * sumX)
        average = sumY / length
        intercept = average - slope * sumX / length + slope

        linear[i] = intercept

    df = pd.DataFrame(linear, columns=['linreq'])
    return df


def LinregNP(source: np.ndarray, length: int):
    size = len(source)
    linear = np.zeros(size)

    for i in range(length, size):

        sumX = 0.0
        sumY = 0.0
        sumXSqr = 0.0
        sumXY = 0.0

        for z in range(length):
            val = source[i-z]
            per = z + 1.0
            sumX += per
            sumY += val
            sumXSqr += per * per
            sumXY += val * per

        slope = (length * sumXY - sumX * sumY) / (length * sumXSqr - sumX * sumX)
        average = sumY / length
        intercept = average - slope * sumX / length + slope

        linear[i] = intercept

    df = pd.DataFrame(linear, columns = ['linreq'])
    return df
