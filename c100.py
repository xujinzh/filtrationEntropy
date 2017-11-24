# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 19:29:29 2017

@author: xujin
"""

import pandas as pd
import numpy as np
import ComputeEntropyFunction

df = pd.read_csv('c100.csv')
df = df.iloc[:, :]

x = [0] * len(df)
for i in np.arange(len(df)):
    x[i] = tuple(df.iloc[i, :])

ComputeEntropyFunction.ComputeEntropy(x, 0.0)