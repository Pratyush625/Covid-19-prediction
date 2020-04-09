#!/usr/bin/env python
# coding: utf-8

# In[52]:


# Disable warnings
import warnings
warnings.filterwarnings('ignore')
#Importing the required libraries
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from plotly.offline import download_plotlyjs, init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
import plotly.express as px
import plotly.graph_objects as go


# In[53]:


#Importing Dataset from Github(Source_link: https://www.kaggle.com/imdevskp/corona-virus-report#covid_19_clean_complete.csv) updated till 6th April
dw=pd.read_excel('COVID19_jan-22-apr-6.xlsx',parse_dates=['Date'])
dw.tail()


# In[54]:


#Preprocessing
dw.groupby('Date').sum().tail()


# In[55]:


confirmed1=dw.groupby('Date').sum()['Confirmed'].reset_index()
deaths1=dw.groupby('Date').sum()['Deaths'].reset_index()
recovered1=dw.groupby('Date').sum()['Recovered'].reset_index()


# In[56]:


#Data Visualization using Plotly
fig=go.Figure()
fig.add_trace(go.Scatter(x=confirmed1['Date'],y=confirmed1['Confirmed'],mode='lines+markers',name='confirmed'))
fig.add_trace(go.Scatter(x=deaths1['Date'],y=deaths1['Deaths'],mode='lines+markers',name='deaths'))
fig.add_trace(go.Scatter(x=recovered1['Date'],y=recovered1['Recovered'],mode='lines+markers',name='recovered'))

fig.update_layout(title='Covid-19 Cases across world',yaxis=dict(title='No of cases'),xaxis=dict(title='Date'))


# In[57]:


#Importing Prophet for Prediction
from fbprophet import Prophet

confirmed1=dw.groupby('Date').sum()['Confirmed'].reset_index()
deaths1=dw.groupby('Date').sum()['Deaths'].reset_index()
recovered1=dw.groupby('Date').sum()['Recovered'].reset_index()


# In[58]:


#Confirmed_case
confirmed1.rename(columns={'Date':'ds', 'Confirmed':'y'})


# In[59]:


confirmed1.columns=['ds','y']
confirmed1['ds']=pd.to_datetime(confirmed1['ds'])


# In[60]:


m=Prophet(interval_width=0.95)
m.fit(confirmed1)
future=m.make_future_dataframe(periods=7)
future.tail()


# In[61]:


forecast=m.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()


# In[62]:


confirmed_case=m.plot(forecast)


# In[63]:


confirmed_case_trend=m.plot_components(forecast)


# In[64]:


#Death cases
deaths1.groupby('Date').sum().tail()


# In[65]:


#Deaths_case
deaths1.rename(columns={'Date':'ds', 'Deaths':'y'})


# In[66]:


deaths1.columns=['ds','y']
deaths1['ds']=pd.to_datetime(deaths1['ds'])


# In[67]:


m=Prophet(interval_width=0.95)
m.fit(deaths1)
future_deaths=m.make_future_dataframe(periods=7)
future_deaths.tail()


# In[68]:


forecast_deaths=m.predict(future_deaths)
forecast_deaths[['ds','yhat','yhat_lower','yhat_upper']].tail()


# In[69]:


deaths_case=m.plot(forecast_deaths)


# In[70]:


deaths_case_trend=m.plot_components(forecast_deaths)


# In[71]:


#Recovered_cases
recovered1.groupby('Date').sum().tail()


# In[72]:


recovered1.rename(columns={'Date':'ds', 'Deaths':'y'})


# In[73]:


recovered1.columns=['ds','y']
recovered1['ds']=pd.to_datetime(recovered1['ds'])


# In[74]:


m=Prophet(interval_width=0.95)
m.fit(recovered1)
future_recovered=m.make_future_dataframe(periods=7)
future_recovered.tail()


# In[75]:


forecast_recovered=m.predict(future_recovered)
forecast_recovered[['ds','yhat','yhat_lower','yhat_upper']].tail()


# In[76]:


recovered_case=m.plot(forecast_recovered)


# In[77]:


recovered_case_trend=m.plot_components(forecast_recovered)

