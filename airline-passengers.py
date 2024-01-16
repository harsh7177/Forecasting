#!/usr/bin/env python
# coding: utf-8

# **1. Importing the Libraries**

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# **2. Load the Dataset**

# In[4]:


df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv", error_bad_lines = False)


# In[5]:


df.head(5)


# In[6]:


df['Month'] = pd.to_datetime(df['Month'])


# In[7]:


df.set_index("Month", inplace = True)


# In[16]:


# Plotting the number of Passengers for every month of the year
plt.figure(figsize=(10, 6))
plt.plot(df.index, df.Passengers, '--', marker = "*", )
plt.grid()
plt.title("Airline Passengers")
plt.xlabel('Year')
plt.ylabel('Passengers')
plt.show()


#  Handling the Missing Values

# In[8]:


df.isnull().sum()


# In[39]:


# Fill the NULL values: Interpolation - Backward Fill
df_copy_LIB = df_copy.interpolate(method = "linear", limit_direction='backward')


# In[40]:


df_copy_LIB.isnull().sum()


# In[41]:


# Resampling
# Fill the NULL values: Downsample to quaterly data points (3 Months)
df_quaterly = df.resample('3M').mean()


# In[42]:


df_quaterly


# In[43]:


df_quaterly.size


# In[44]:


df_quaterly.isnull().any()


# In[45]:


plt.figure(figsize = (10, 6))
plt.plot(df_quaterly.index, df_quaterly.Passengers, '-', marker='*')
plt.grid()
plt.xlabel('Year')
plt.title("Resampled datapoints(Downsampled to 3M)")
plt.ylabel('Passengers')
plt.show()


# **3.2** Time Series **E**xploratory **D**ata **A**nalysis

# In[12]:


import statsmodels.api as sm
from pylab import rcParams


# In[13]:


rcParams['figure.figsize'] = 12, 8
# A object with seasonal, trend, and resid attributes
# Y[t] = T[t] + S[t] + e[t]
decompose_series = sm.tsa.seasonal_decompose(df['Passengers'], model = 'additive')
decompose_series.plot()
plt.show()


# **3.3** Convert time series into stationary by removing trend and seasonality

# In[92]:


df_log = np.log(df)


# In[94]:


df_log


# In[95]:


# Differencing
df_diff = df_log.diff(periods = 1)


# In[96]:


# Calculating the difference from its previous value, periods=1 (default) is the difference from its previous value
df_diff


# In[97]:


df_diff.rolling(12)


# In[98]:


df_diff.rolling(12).mean()


# In[99]:


plt.figure(figsize=(12, 8))
plt.plot(df_diff.index, df_diff.Passengers, '-')
plt.title("Differenced Data")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.plot(df_diff.rolling(12).mean(), color='red')
plt.show()


# **4. Time Series Forecasting using Stochastic Models**

# In[101]:


from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[103]:


df_diff['Passengers']


# In[112]:


from statsmodels.tsa.arima_model import ARIMA
AR_model = ARIMA(df_diff, order=(2, 0, 0))
AR_model_results = AR_model.fit()
plt.plot(df_diff)
plt.title("Autoregressive")
plt.plot(AR_model_results.fittedvalues, color = 'red')


# In[113]:


# Moving Average
MA_model = ARIMA(df_diff, order = (0, 0, 2))
MA_model_results = MA_model.fit()
plt.plot(df_diff)
plt.title("Moving Average")
plt.plot(MA_model_results.fittedvalues, color='red')


# In[114]:


# ARIMA model
ARIMA_model = ARIMA(df_diff, order=(2, 0, 1))
ARIMA_results = ARIMA_model.fit()
plt.plot(df_diff)
plt.title("ARIMA model")
plt.plot(ARIMA_results.fittedvalues, color="red")

