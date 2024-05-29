import pandas as pd
import numpy as np
from qiskit_finance.data_providers import BaseDataProvider
from decimal import Decimal, getcontext


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
        self._initial_monetary_value = 100000000  # 100 million
        self._current_monetary_value = 100000000
        self._monetary_data = pd.DataFrame()
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

    def run_nonlog(self) -> None:
        # Load data from the Excel file
        df = self.load_data()
        self._data = df

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

    # Set the precision.
    getcontext().prec = 6

    def rebalance_with_returns(self, target_weights, returns):
        # Calculate the new values after applying returns
        new_values = {stock: (Decimal(target_weights[stock]) * Decimal(self._current_monetary_value)) * (1 + Decimal(returns[stock])) for stock in target_weights}
        total_new_value = sum(new_values.values())
        # Calculate the desired values
        desired_values = {stock: Decimal(target_weights[stock]) * total_new_value for stock in target_weights}

        self._current_monetary_value = total_new_value

        # Calculate the rebalanced weights
        rebalanced_weights = {stock: desired_values[stock] / total_new_value for stock in desired_values}

        return desired_values, rebalanced_weights

    def aggregate_returns(self, frequency):
        if self._data.empty:
            raise ValueError("No data available. Ensure `run` has been executed successfully.")

        df = self._data
        if frequency == 'monthly':
            return df.resample('M').apply(lambda x: (1 + x).prod() - 1)
        elif frequency == 'quarterly':
            return df.resample('Q').apply(lambda x: (1 + x).prod() - 1)
        elif frequency == 'semi-annual':
            return df.resample('6M').apply(lambda x: (1 + x).prod() - 1)
        elif frequency == 'annual':
            return df.resample('A').apply(lambda x: (1 + x).prod() - 1)
        else:
            raise ValueError("Invalid frequency. Choose from 'monthly', 'quarterly', 'semi-annual', 'annual'.")

    def rebalance_portfolio_over_time(self, target_weights, frequency, printbool=False, output_file="./data/monetary.xlsx"):
        aggregated_returns = self.aggregate_returns(frequency)

        # Initialize portfolio weights to the target weights
        portfolio_weights = {k: Decimal(v) for k, v in target_weights.items()}

        # Create a DataFrame to store the monetary values and weights over time
        monetary_data = pd.DataFrame(index=aggregated_returns.index, columns=['Total Value'] + [f"{stock} Value" for stock in target_weights.keys()] + [f"{stock} % of Total" for stock in target_weights.keys()])

        for date, period_returns in aggregated_returns.iterrows():
            period_returns = period_returns.to_dict()
            period_returns = {k: Decimal(v) for k, v in period_returns.items()}

            rebalanced_values, rebalanced_weights = self.rebalance_with_returns(portfolio_weights, period_returns)
            rebalanced_values = {k: float(v) for k, v in rebalanced_values.items()}
            rebalanced_weights = {k: float(v) for k, v in rebalanced_weights.items()}

            # Record the current total value and the values and weights of each stock
            monetary_data.loc[date, 'Total Value'] = float(self._current_monetary_value)
            for stock in target_weights.keys():
                monetary_data.loc[date, f"{stock} Value"] = rebalanced_values[stock]
                monetary_data.loc[date, f"{stock} % of Total"] = rebalanced_weights[stock] * 100

            if printbool:
                print(f"\nRebalancing on {date.date()}:")
                print(f"Returns: {period_returns}")
                print(f"Rebalanced Values: {rebalanced_values}")
                print(f"Rebalanced Weights: {rebalanced_weights}")

            # Update portfolio weights based on rebalanced weights
            portfolio_weights = {k: Decimal(v) for k, v in rebalanced_weights.items()}

        if output_file:
            monetary_data.to_excel(output_file)

        return {k: Decimal(v) for k, v in portfolio_weights.items()}


# Example usage:
# processor = StockDataProcessor(start="2022-01-01", end="2022-12-31", file_path="path/to/data.xlsx")
# processor.run()
# target_weights = {'AAPL': 0.3, 'MSFT': 0.4, 'GOOG': 0.3}
# processor.rebalance_portfolio_over_time(target_weights, frequency='monthly', printbool=True, output_file="path/to/output.xlsx")
