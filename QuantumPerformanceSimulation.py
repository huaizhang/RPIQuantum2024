import pandas as pd
from qiskit.circuit.library import TwoLocal
from qiskit.result import QuasiDistribution
from qiskit_algorithms import NumPyMinimumEigensolver, QAOA, SamplingVQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_finance.data_providers import YahooDataProvider
from qiskit_finance.data_providers import BaseDataProvider
from qiskit_optimization.algorithms import MinimumEigenOptimizer
import numpy as np
import matplotlib.pyplot as plt
import datetime
from qiskit_ibm_runtime import QiskitRuntimeService
from typing import Optional, Union, List



class StockDataProcessor(BaseDataProvider):
    def __init__(self, file_path, tickers, start, end):
        """
        Initialize with the path to the Excel file, list of tickers, and date range.
        """
        self._file_path = file_path
        self._tickers = tickers
        self._start = pd.to_datetime(start)
        self._end = pd.to_datetime(end)
        self._data = []

    def load_data(self):
        """
        Loads data from an Excel file into a DataFrame.
        """
        # Read data from the provided Excel file
        df = pd.read_excel(self._file_path, index_col=0)
        #df = df.sort_values("Date")
        #df = df.reset_index(drop=True)

        # Filter the DataFrame to only include the specified date range
        #df = df.loc[self._start:self._end]
        #print(df)
        return df

    def run(self) -> None:
        """
        Fills each ticker with the corresponding stock data from the Excel file.
        """
        # Load data from the Excel file
        df = self.load_data()
        
        # Process each ticker
        for ticker in self._tickers:
            if ticker in df.columns:
                # Extract the column data for the current ticker
                ticker_data = df[ticker].tolist()
                # Append to the internal data storage
                self._data.append(ticker_data)
            else:
                # Handle the case where a ticker is not found in the DataFrame columns
                print(f"Warning: Ticker '{ticker}' not found in the data.")
                # Append an empty list or some default data
                self._data.append([])  # This could be modified to handle missing data differently
    

#4/30/2004
#3/31/2024

data = StockDataProcessor(
    tickers=["^GSPC", "^ACWX", "^GLAB.L"],
    start=datetime.datetime(2024, 2, 19),
    end=datetime.datetime(2024, 3, 31),
    file_path = "./data/historic_data.xlsx"
)
data.run()
for (cnt, s) in enumerate(data._tickers):
    plt.plot(data._data[cnt], label=s)
plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=3)
plt.xticks(rotation=90)
plt.show()


mu = data.get_period_return_mean_vector()
sigma = data.get_period_return_covariance_matrix()
print(sigma)
plt.imshow(sigma)
plt.show()