# RPI Quantum 2024

Some sample Python code is in the [sample file](./Python_sample_code.py)

### Asset classes performance data
A typical investment portfolio has three asset classes :
- US Equities : we will use the *S&P 500 index* as the proxy.
- International Equities : we will use the *MSCI ACWI ex US index* as the proxy.
- Global fixed income : we will use the *Bloomberg Global Aggregated Index*.
  
The historic monthly performances of above indices in the past 20 yeears are in this [data file](./data/historic_data.xlsx)[^1]. The following Python code can 

### Calculate the historic portfolio performance
Assuming the portfolio invested in 30% in US equities, 30% in international equities, and 40% in global fixed income at the beginning of April 2004, and the portfolio is rebalanced monthly[^2].

### Covariance, correlation, volatility
Calculate the covariance, correlation, and volatility based on logarithmic performances.

### Performance simulations
Generate 10 years of simulated performances based expected returns and historic covariance matrix. The simulation is done with assumption that the logarithmic returns is normally distributed.
   
<span style='color : red;'>The generation of simulated performances potentiall can be sped up by the use for quantum computer for more than 3 asset classes.</span>
   
The simulations can be done 10000 times and obtain the distribution of annual portfolio returns and volatilities. <span style='color : red'>This is why we need to speed up the simulation.</span>

### Change rebalance rules
- Rebalance quartery, annually, or other fixed rebalance frequencies.
- Reablance dynamically based on some rules.

[^1]: We use the first day of the month to indicate the month. The performance is for 3/1/2024 is the total return from 3/1/2024 to 3/31/2024.
[^2]: "Rebalance" means to reset the weights after the movement of markets. For example, the 30%-30%-40% will deviate from the original values after the three asset classes perform differently.