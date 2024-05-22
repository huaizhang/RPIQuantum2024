import pandas as pd
import numpy as np

# read data from ./data/historical_data.xlsx
historic_data = pd.read_excel("../data/historic_data.xlsx")
historic_data = historic_data.sort_values("Date")
historic_data = historic_data.reset_index(drop=True)

# calculate the portfolio performance
weights = np.array([0.3, 0.3, 0.4])
historic_data["portfolio"] = historic_data.iloc[:, 1:4].dot(weights)

# convert the percentage performances to logaritmic returns
log_returns = np.log(1 + historic_data.iloc[:, 1:4])
# calculate the coveriance matrix
cov_matrix = log_returns.cov()
print(cov_matrix)
# caluclate the percentage volatility of each asset
volatilities = np.exp(np.sqrt(np.diag(cov_matrix))) - 1
# calculate the correlation matrix
correlation_matrix = log_returns.corr()

# simulated performance based on expected returns and historic covariance matrix
annual_expected_returns = np.array([0.1, 0.1, 0.06])
# convert the annualized expected returns to monthly logarithmic returns
monthly_expected_log_returns = np.log(1 + annual_expected_returns) / 12


simulated_log_returns = np.random.multivariate_normal(
    monthly_expected_log_returns, cov_matrix, 360
)
simulated_log_returns = pd.DataFrame(
    simulated_log_returns,
    columns=["US Equities", "International Equities", "Global Fixed Income"],
)

# convert the simulated logarithmic returns to percentage returns
simulated_returns = np.exp(simulated_log_returns) - 1
# calculate the portfolio performance
simulated_returns["portfolio"] = simulated_returns.dot(weights)
# calculate the annual portfolio return and annual volatility
annual_portfolio_return = (1 + simulated_returns["portfolio"]).prod() ** (
    12 / simulated_log_returns.shape[0]
) - 1
annual_portfolio_volatility = np.std(simulated_returns["portfolio"]) * np.sqrt(12)
# calculate the Sharpe ratio
risk_free_rate = 0.00
sharpe_ratio = (annual_portfolio_return - risk_free_rate) / annual_portfolio_volatility
# calculate the maximum drawdown
cumulative_returns = (1 + simulated_returns["portfolio"]).cumprod()
max_drawdown = np.min(
    cumulative_returns / np.maximum.accumulate(cumulative_returns) - 1
)
# calculate the Calmar ratio
calmar_ratio = annual_portfolio_return / max_drawdown

# Rebalance the portfolio quarterly based on the historical data
# Similar approach can be used to calculate the portfolio performance for other rebalancing frequencies
# such as monthly, semi-annually, and annually
# The approach can also be used to calculate the portfolio performance for simulated data.
quarterly_returns = historic_data.iloc[:, 1:4]
quarterly_returns = (1 + quarterly_returns).cumprod()
dates = historic_data["Date"]
months = pd.DatetimeIndex(dates).month
quarter_ends = np.where(months % 3 == 0)[0]
quarterly_returns = quarterly_returns.iloc[quarter_ends]
quarterly_returns.loc[-1] = 1
quarterly_returns = quarterly_returns.sort_index()
quarterly_returns = quarterly_returns.pct_change()
quarterly_returns = quarterly_returns.dropna()
quarterly_returns = quarterly_returns.reset_index(drop=True)
# calculate the portfolio performance
quarterly_returns["portfolio"] = quarterly_returns.dot(weights)
annual_portfolio_return = (1 + quarterly_returns["portfolio"]).prod() ** (
    4 / quarterly_returns.shape[0]
) - 1
annual_portfolio_volatility = np.std(simulated_returns["portfolio"]) * np.sqrt(4)

# Dynamic rebalancing based on the historical data
