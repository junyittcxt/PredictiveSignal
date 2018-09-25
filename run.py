# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 00:05:30 2018

@author: Workstation
"""

import numpy as np
import talib
import pandas as pd
import mf


df = pd.read_csv("C:/Users/Workstation/Desktop/82_ETF_FOREX_1_DAY.csv").fillna(method = "ffill")

par = dict(sma_period = 5, signal_shift = 1)
par2 = dict(sma_period = 5, signal_shift = 2)

signal_lambda = lambda x: talib.SMA(x, par["sma_period"]) > x

def PredAna(df, par, signal_lambda):
    signal_df = df.set_index("Date").apply(signal_lambda)
    
    return_df = df.set_index("Date").pct_change().fillna(0)
    signal_df = signal_df.shift(par["signal_shift"]).fillna(False)
    
    full_result = []
    returns_dict = dict()
    for rcol in return_df:
        for scol in signal_df:
            signal = signal_df[scol].values
            returns = return_df[rcol].values
            filtered_returns = signal*returns
            metric_result = mf.compute_metrics(filtered_returns)
            result_name = scol + "_" + rcol
            metric_result["name"] = result_name
            full_result.append(metric_result)
            returns_dict[result_name] = filtered_returns
    return full_result, returns_dict

full_result, returns_dict = PredAna(df, par, signal_lambda)
full_result2, returns_dict2 = PredAna(df, par2, signal_lambda)



result_df = pd.DataFrame(full_result).sort_values("R Squared", ascending = False)
result_df2 = pd.DataFrame(full_result2).sort_values("R Squared", ascending = False)

result_df[result_df["name"] == "LQD_DIA"]
result_df2[result_df2["name"] == "LQD_DIA"]

result_df["Sharpe"].hist(bins = 30)


ref = "FEZ"
main = "XLP"
a1 = df[["Date", ref]].set_index("Date").apply(signal_lambda)
a2 = df[["Date", main]].set_index("Date").pct_change().shift(-1).fillna(0)
pd.DataFrame(mf.cumulative_returns((a1[ref]*a2[main]).values)).plot()
(a1[ref]*a2[main]).reset_index().set_index("Date").apply(lambda x: mf.cumulative_returns(x)).plot()
