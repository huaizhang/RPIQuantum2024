import pandas as pd
import numpy as np
from qiskit_finance.data_providers import BaseDataProvider
class StockDataProcessor(BaseDataProvider):
    """
    StockDataProcessor is a child class of parent class BaseDataProvider from Qiskit Finance.
    Storing data in this form will be beneficial as it allows usage of further Qiskit Finance functions.
    """
    def __init__(self, start, end, file_path=None, data=None):
        self._file_path = file_path
        self._start = pd.to_datetime(start)
        self._end = pd.to_datetime(end)
        self._tickers = []
        self._data = data if data is not None else pd.DataFrame()

        self._cov_matrix = np.ndarray
        self._mean_vec = np.ndarray
        self._correlation = np.ndarray
        self._stddev = np.ndarray
        self._volatility = np.ndarray

        if self._data.empty and self._file_path is None:
            raise ValueError("Either file_path or data must be provided")

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

    def rebalance_with_returns(self, target_weights, returns):
        new_values = {stock: target_weights[stock] * (1 + returns[stock]) for stock in target_weights}
        total_new_value = sum(new_values.values())

        new_weights = {stock: new_values[stock] / total_new_value for stock in new_values}
        adjustments = {stock: target_weights[stock] - new_weights[stock] for stock in target_weights}

        rebalanced_weights = {stock: new_weights[stock] + adjustments[stock] for stock in new_weights}
        return rebalanced_weights, adjustments
    

    def aggregate_returns(self, frequency):
        if self._data.empty:
            raise ValueError("No data available. Ensure `run` has been executed successfully.")
        
        df = self._data
        if frequency == 'monthly': 
            return df.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        elif frequency == 'quarterly':
            return df.resample('QE').apply(lambda x: (1 + x).prod() - 1)
        elif frequency == 'semi-annual':
            return df.resample('6ME').apply(lambda x: (1 + x).prod() - 1)
        elif frequency == 'annual':
            return df.resample('YE').apply(lambda x: (1 + x).prod() - 1)
        else:
            raise ValueError("Invalid frequency. Choose from 'monthly', 'quarterly', 'semi-annual', 'annual'.")

    def rebalance_portfolio_over_time(self, target_weights, frequency, printbool):
        aggregated_returns = self.aggregate_returns(frequency)
        
        portfolio_weights = target_weights.copy()
        for date, period_returns in aggregated_returns.iterrows():
            rebalanced_weights, adjustments = self.rebalance_with_returns(portfolio_weights, period_returns.to_dict())
            if printbool == 1:
                print(f"\nRebalancing on {date.date()}:")
                print(f"Portfolio Weights: {portfolio_weights}")
                print(f"Returns: {period_returns.to_dict()}")
                print(f"Adjustments to match Weights: {adjustments}")
            
            # Apply rebalanced weights to the appropriate date in the ._data
            if date in self._data.index:
                self._data.loc[date] = [rebalanced_weights[ticker] for ticker in self._tickers]
            portfolio_weights = rebalanced_weights
        
        return portfolio_weights