import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_finance.data_providers import BaseDataProvider
from qiskit.quantum_info import Statevector

class StockDataProcessor(BaseDataProvider):
    """
    StockDataProcessor is a child class of parent class BaseDataProvider from Qiskit Finance. 
    Storing data in this form will be beneficial as it allows usage of further Qiskit Finance functions. 

    """
    def __init__(self, file_path, start, end):
        self._file_path = file_path
        self._start = pd.to_datetime(start)
        self._end = pd.to_datetime(end)
        self._tickers = []
        self._data = pd.DataFrame()

        self._cov_matrix = np.ndarray
        self._mean_vec = np.ndarray
        self._correlation = np.ndarray
        self._stddev = np.ndarray
        self._volatility = np.ndarray

    def load_data(self) -> pd.DataFrame:
        try:
            # Read data from the provided Excel file
            df = pd.read_excel(self._file_path, index_col=0)
            df.index = pd.to_datetime(df.index)
            df.sort_values("Date")
            df.reset_index(drop=True)

            # Filter the DataFrame to only include the specified date range
            df = df[(df.index >= self._start) & (df.index <= self._end)]
            # Set the tickers to the column headers
            self._tickers = df.columns.tolist()
            return df
        except Exception as e:
            raise IOError(f"Error loading data from {self._file_path}: {e}")

    @staticmethod
    def calculate_log_returns(prices: pd.Series) -> pd.Series:
        return np.log(1 + prices).dropna()

    def run(self) -> None:
        # Load data from the Excel file
        df = self.load_data()
        # Calculate log returns for all tickers
        log_returns = df.apply(self.calculate_log_returns, axis=0)
        # Drop rows with NaN values (e.g., first row due to shift)
        self._data = log_returns.dropna()
        self.get_mean_vector()
        self.get_covariance_matrix()
        self.get_stddev()
        self.get_correlation()
        self.get_volatility()

    def get_mean_vector(self):
        if self._data.empty:
            raise ValueError("No data available. Ensure `run` has been executed successfully.")
        self._mean_vec = self._data.mean(axis=0).to_numpy()

    def get_covariance_matrix(self):
        if self._data.empty:
            raise ValueError("No data available. Ensure `run` has been executed successfully.")
        self._cov_matrix = self._data.cov().to_numpy()
        
    def get_stddev(self):
        if self._data.empty:
            raise ValueError("No data available. Ensure `run` has been executed successfully.")
        self._stddev = np.sqrt(np.diag(self._cov_matrix))

        return self._stddev
    
    def get_correlation(self):
        if self._data.empty:
            raise ValueError("No data available. Ensure `run` has been executed successfully.")
        self._correlation = self._cov_matrix / np.outer(self._stddev, self._stddev)
    
    def get_volatility(self):
        if self._data.empty:
            raise ValueError("No data available. Ensure `run` has been executed successfully.")

        self._volatility = np.exp(np.sqrt(np.diag(self._cov_matrix))) - 1


    def print_stats(self):
        print("-------------------------------------------------------")
        print("Correlation Matrix: ")
        print(self._correlation)
        print("Covariance Matrix: ")
        print(self._cov_matrix)
        print("Mean Vector: ", self._mean_vec)
        print("Std Devs: ", self._stddev)
        print("Volatility: ", self._volatility)
        print("-------------------------------------------------------")


# Example usage of class
data = StockDataProcessor(
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
plt.savefig('StockGraph.png')

data.run()
data.print_stats()
