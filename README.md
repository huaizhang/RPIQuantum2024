# RPI Quantum 2024

## Multi Period Asset Allocation Problem

We use simulations to estimate the potential return and risk of a portfolio with specific asset allocation over the next 10 years. We break down this problem into several steps as following. Some sample Python codes are in the [sample file](./src/__pycache__/Python_sample_code.py). 

### 1. Asset classes performance data
A typical investment portfolio has three asset classes :
- US Equities : we will use the *S&P 500 index* as the proxy.
- International Equities : we will use the *MSCI ACWI ex US index* as the proxy.
- Global fixed income : we will use the *Bloomberg Global Aggregated Index*.

The historic monthly returns of above indices in the past 20 yeears are in this [data file](./data/historic_data.xlsx)[^1].

---

### 2. Calculate the historic portfolio performance
Assuming the portfolio invested in 30% in US equities, 30% in international equities, and 40% in global fixed income at the beginning of April 2004, and the portfolio is rebalanced monthly[^2].

---

### 3. Calculated the historic risk measures, such as covariance, correlation, volatility
Calculate the historic covariance, correlation, and volatility based on logarithmic returns[^3].

---

### 4. Performance simulations
In the simulation, we will NOT use the historic returns, instead we will give the expected returns for three asset classes. We will keep the historic covariance since it is much more stable.

Generate 10 years of simulated performances based expected returns and historic covariance matrix. The simulation is done with assumption that the logarithmic returns is normally distributed.
   
<span style='color : red;'>The generation of simulated performances potentiall can be sped up by the use for quantum computer for more than 3 asset classes.</span>
   
The simulations can be done 10000 times and obtain the distribution of annual portfolio returns and volatilities. <span style='color : red;'>This is why we need to speed up the simulation.</span>

---

### 5. Change rebalance rules
- Rebalance quartery, annually, or other fixed rebalance frequencies.
- Reablance dynamically based on some rules.


### 6. Some statistics
- Sharpe Ratio is the ratio of annualized return and annualized volatility. In principle, we should subtract so-call risk-free rate from the annulaized return. We can set it to zero for now.
- Maximum Drawdown is the maximum peak to trough loss of a portfolio.
- Calmar Ratio is the ratio of annualized return over maximum drawdown.

---
[^1]: We use the first day of the month to indicate the month. The performance is for 3/1/2024 is the total return from 3/1/2024 to 3/31/2024.
[^2]: "Rebalance" means to reset the weights after the movement of markets. For example, the 30%-30%-40% will deviate from the original values after the three asset classes perform differently.
[^3]: Logarithmic return is calculated as log(1+r) where r is the return. The logarithmic returns can be assumed to be normally distributed.
