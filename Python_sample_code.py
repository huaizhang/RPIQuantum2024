import pandas as pd
import numpy as np

# read data from ./data/historical_data.xlsx
historic_data = pd.read_excel("./data/historic_data.xlsx")

# calculate the portfolio performance
weights = np.array([0.3, 0.3, 0.4])
historic_data["portfolio"] = historic_data.iloc[:, 1:4].dot(weights)

# convert the percentage performances to logaritmic returns
log_returns = np.log(1 + historic_data.iloc[:, 1:4])
# calculate the coveriance matrix
cov_matrix = log_returns.cov()
# caluclate the percentage volatility of each asset
volatilities = np.exp(np.sqrt(np.diag(cov_matrix))) - 1
# calculate the correlation matrix
correlation_matrix = log_returns.corr()

# simulated performance based on expected returns and covariance matrix
annual_expected_returns = np.array([0.1, 0.1, 0.06])
# convert the annualized expected returns to monthly logarithmic returns
monthly_expected_log_returns = np.log(1 + annual_expected_returns) / 12

simulated_log_returns = np.random.multivariate_normal(
    monthly_expected_log_returns, cov_matrix, 360
)
simulated_log_returns = pd.DataFrame(
    simulated_log_returns,
    columns=["US Equities", "International Equiteis", "Global Fixed Income"],
)
# convert the simulated logarithmic returns to percentage returns
simulated_returns = np.exp(1 + simulated_log_returns) - 1
# calculate the portfolio performance
simulated_returns["portfolio"] = simulated_returns.dot(weights)
# calculate the annual portfolio return and annual volatility
annual_portfolio_return = (1 + simulated_returns["portfolio"]).prod() ^ (1 / 12) - 1
annual_portfolio_volatility = np.std(simulated_returns["portfolio"]) * np.sqrt(12)
