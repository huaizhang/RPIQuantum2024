import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from qiskit_finance.data_providers import BaseDataProvider
from qiskit_finance.circuit.library import GaussianConditionalIndependenceModel


class StockDataProcessor(BaseDataProvider):
    """
    StockDataProcessor is a child class of parent class BaseDataProvider from Qiskit Finance. 
    Storing data in this form will be beneficial as it allows usage of further Qiskit Finance functions. 

    """
    def __init__(self, file_path, start, end):
        """
        Initialize with the path to the Excel file and date range.
        """
        self._file_path = file_path
        self._start = pd.to_datetime(start)
        self._end = pd.to_datetime(end)
        self._tickers = []
        self._data = pd.DataFrame()

    def load_data(self) -> pd.DataFrame:
        """
        Loads data from an Excel file into a DataFrame and sets the tickers.
        """
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
        
        # Calculate log returns for all tickers
        log_returns = df.apply(self.calculate_log_returns, axis=0)
        
        # Drop rows with NaN values (e.g., first row due to shift)
        self._data = log_returns.dropna()

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

#        return np.log(1 + prices).dropna()

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


mean_vector = data.get_period_return_mean_vector() 
cov_matrix = data.get_period_return_covariance_matrix() 
plt.imshow(cov_matrix)
plt.savefig('StockCovar.png')
volatility = np.exp(np.sqrt(np.diag(cov_matrix))) - 1

std_devs = np.sqrt(np.diag(cov_matrix))
# Calculate the correlation matrix
correlation_matrix = cov_matrix / np.outer(std_devs, std_devs)
flattened = list(set(correlation_matrix.flatten()))
correlation_matrix_adjusted = [ round(elem, 4) for elem in flattened ] # Sensitivities of default probabilities rounded to 4 decimal places
correlation_matrix_adjusted.remove(1.) #remove dummy data (correlation of 1 for same stock)
correlation_matrix_adjusted.remove(1.)
correlation_matrix_adjusted.remove(1.)

annual_expected_returns = np.array([0.1, 0.1, 0.06]) # Standard default probabilities
monthly_expected_log_returns = np.log(1 + annual_expected_returns) / 12

# Gaussian Conditional Independence Model parameters
n_normal = 3  # Number of qubits to represent the latent normal random variable Z
normal_max_value = 3  # Truncate the latent normal random variable Z between +/- this value

# Create Gaussian conditional independence model
print(monthly_expected_log_returns)
model = GaussianConditionalIndependenceModel(n_normal, normal_max_value, monthly_expected_log_returns, correlation_matrix_adjusted)
