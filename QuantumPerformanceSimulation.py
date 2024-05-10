import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from qiskit_finance.data_providers import BaseDataProvider

class StockDataProcessor(BaseDataProvider):
    def __init__(self, file_path, tickers, start, end):
        """
        Initialize with the path to the Excel file, list of tickers, and date range.
        """
        self._file_path = file_path
        self._tickers = tickers
        self._start = pd.to_datetime(start)
        self._end = pd.to_datetime(end)
        self._data = pd.DataFrame()

    def load_data(self):
        """
        Loads data from an Excel file into a DataFrame.
        """
        # Read data from the provided Excel file
        df = pd.read_excel(self._file_path, index_col=0)
        # Optionally sort by index if the data is not in date order
        df.sort_index(inplace=True)
        # Filter the DataFrame to only include the specified date range
        df.index = pd.to_datetime(df.index)
        df = df[(df.index >= self._start) & (df.index <= self._end)]
        return df

    def calculate_log_returns(self, prices: pd.Series) -> pd.Series:
        """
        Calculates the log returns for a series of prices.

        Parameters:
        prices (pd.Series): A series of percentage returns or prices.

        Returns:
        pd.Series: A series containing log returns.
        """
        return np.log(1 + prices).dropna()

    def run(self) -> None:
        """
        Processes each ticker to calculate log returns from the Excel file.
        """
        # Load data from the Excel file
        df = self.load_data()
        
        # Create a DataFrame to store the log returns
        log_returns_data = {}

        # Process each ticker and compute log returns
        for ticker in self._tickers:
            if ticker in df.columns:
                # Calculate log returns for the current ticker
                log_returns = self.calculate_log_returns(df[ticker])
                log_returns_data[ticker] = log_returns
            else:
                # Handle the case where a ticker is not found in the DataFrame columns
                print(f"Warning: Ticker '{ticker}' not found in the data.")
                log_returns_data[ticker] = pd.Series(dtype=float)

        # Create a DataFrame with log returns
        self._data = pd.DataFrame(log_returns_data).dropna()

    def get_period_return_mean_vector(self) -> np.ndarray:
        """
        Computes the mean vector of period returns.

        Returns:
        np.ndarray: Mean vector of period returns.
        """
        if self._data.empty:
            raise ValueError("No data available. Ensure `run` has been executed successfully.")

        return self._data.mean(axis=0).to_numpy()

    def get_period_return_covariance_matrix(self) -> np.ndarray:
        """
        Computes the covariance matrix of period returns.

        Returns:
        np.ndarray: Covariance matrix of period returns.
        """
        if self._data.empty:
            raise ValueError("No data available. Ensure `run` has been executed successfully.")

        return self._data.cov().to_numpy()

# Example usage
data = StockDataProcessor(
    tickers=["^GSPC", "^ACWX", "^GLAB.L"],
    start=datetime.datetime(2004, 4, 30),
    end=datetime.datetime(2024, 3, 31),
    file_path="./data/historic_data.xlsx"
)

data.run()

# Plotting log returns for each ticker
for (cnt, ticker) in enumerate(data._tickers):
    plt.plot(data._data.iloc[:, cnt], label=ticker)
plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=3)
plt.xticks(rotation=90)
plt.savefig('foo2.png')

# Computing covariance matrix
sigma = data.get_period_return_covariance_matrix()
print(sigma)
