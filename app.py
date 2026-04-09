import streamlit as st
import yfinance as yf
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from arch import arch_model

st.title("Quant Risk Analyser")


#Let user tyoe stock ticker"

ticker=st.text_input("Enter Stock Ticker", value="AAPL")

# Let user choos how many years of data

period=st.selectbox("Time Period",["1y","2y","5y","10y"])

# download data 

if st.button("Analyze"):
	st.write(f"Fetching data for {ticker}...")
	data=yf.download(ticker,period=period)
	data.columns=data.columns.get_level_values(0) 
	data['Returns']=data['Close'].pct_change()
	data=data.dropna()
	st.write(f"Got {len(data)} trading days of data")
	returns=data["Returns"]

## Descriptive Stats

	st.header("Descriptitve Statistics")
	mean=np.mean(returns)
	sigma=np.std(returns,ddof=1)
	col1,col2,col3=st.columns(3)
	col1.metric("Annual Return",f"{round(mean*100*252,2)}%")
	col2.metric("Annual Volatility",f"{round(sigma*100*np.sqrt(252),2)}%")
	col3.metric("Sharpe Ratio",round((mean*252)/(sigma*np.sqrt(252)),2))
	
	col4,col5,col6=st.columns(3)
	col4.metric("Skewness",round(stats.skew(returns),2))
	col5.metric("Kurtosis",round(stats.kurtosis(returns),2))
	col6.metric("Trading days",len(returns))

## Distribution chart

	st.header("Distribution Analysis")
	x=np.linspace(np.min(returns),np.max(returns),100)
	mu=mean
	normal_curve=stats.norm.pdf(x,mu,sigma)

	nu,t_mu,t_sigma=stats.t.fit(returns)
	t_curve=stats.t.pdf(x,nu,t_mu,t_sigma)

	fig,ax=plt.subplots(figsize=(10,6))
	ax.hist(returns,bins=50,density=True,alpha=0.7,color="steelblue",label='Actual Daily Returns')
	ax.plot(x,normal_curve,"r--",linewidth=2, label="Normal Distribution")
	ax.plot(x,t_curve,"g--",linewidth=2, label="t Distribution")
	ax.set_title(f"{ticker} Daily Returns Normal vs t-distribution")
	ax.set_xlabel("Daily Return")
	ax.set_ylabel("Density")
	ax.legend()
	st.pyplot(fig)


## VaR

	st.header("Value At Risk for $1,000,000 Portfolio")
	portfolio=1000000
	var_normal=-stats.norm.ppf(0.01,mu,sigma)*portfolio
	var_t=-stats.t.ppf(0.01,nu,t_mu,t_sigma)*portfolio
	
	col7,col8=st.columns(2)
	col7.metric("99% 1-day VaR(Normal):",round(var_normal,0),"$")
	col8.metric("99% 1-day VaR(t-distribution):",round(var_t,0),"$")

	st.warning(f" The Normal distribution underestimates risk by ${round(var_t-var_normal,0):,.0f}")


## GARCH volatility
	st.header("GARCH Volatility analysis")
	model=arch_model(returns*100,vol="GARCH",p=1,q=1)
	result=model.fit(disp="off")
	vol=result.conditional_volatility
	
	col9,col10,col11=st.columns(3)
	col9.metric("Current GARCH vol",f'{round(vol.iloc[-1],2)}%')
	col10.metric("Fixed Historical vol",f'{round(np.std(returns)*100,2)}%')
	col11.metric("Alpha+Beta",round(result.params['alpha[1]']+result.params['beta[1]'],4))

	fig2,ax2=plt.subplots(figsize=(12,4))
	ax2.plot(data.index,vol,color='red')
	ax2.set_title(f'{ticker} GARCH (1,1) Daily Volatility Over time')
	ax2.set_ylabel('Volatility(%)')
	st.pyplot(fig2)