import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_finance.circuit.library.probability_distributions import NormalDistribution
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit.circuit.library import QFT
from qiskit_finance.data_providers import BaseDataProvider
from qiskit_finance.circuit.library import GaussianConditionalIndependenceModel
from qiskit_ibm_runtime import QiskitRuntimeService, Session




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
precision_matrix = np.linalg.inv(cov_matrix) # I think we need to use this for the GaussianConditionalIndependenceModel rather than the covariance matrix? not sure 
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

num_qubits = [3,3,3]
def normalize_probabilities(means, cov_matrix):
    """
    Normalize the mean and covariance matrix so that the integral over the
    probability density function equals 1.
    
    Args:
        means (np.ndarray): Mean vector of the distribution.
        cov_matrix (np.ndarray): Covariance matrix of the distribution.
    
    Returns:
        (np.ndarray, np.ndarray): Normalized mean vector and covariance matrix.
    """
    # Assuming a multivariate normal, calculate a simple scaling factor
    # This is a placeholder: actual normalization might depend on how these
    # are used to generate probabilities.
    scaling_factor = np.sqrt(np.linalg.det(2 * np.pi * cov_matrix))
    normalized_means = means / scaling_factor
    normalized_cov = cov_matrix / scaling_factor
    
    return normalized_means, normalized_cov

# Normalize your mean vector and covariance matrix
normalized_mean_vector, normalized_cov_matrix = normalize_probabilities(mean_vector, cov_matrix)

std_devs = np.sqrt(np.diag(cov_matrix))

bounds = [(monthly_expected_log_returns[i] - 3*std_devs[i], monthly_expected_log_returns[i] + 3*std_devs[i]) for i in range(len(monthly_expected_log_returns))]

print("Calculated Bounds:")
for i, b in enumerate(bounds):
    print(f"Dimension {i+1}: {b}")
# Use normalized outputs for your quantum operations
mvnd = NormalDistribution(num_qubits, normalized_mean_vector, normalized_cov_matrix, bounds=bounds)
qc = QuantumCircuit(sum(num_qubits))
qc.append(mvnd, range(sum(num_qubits)))
#qc.append(QFT(sum(num_qubits)), range(sum(num_qubits)))
qc.measure_all()

# Sample using the Sampler primitive
sampler = Sampler()
job = sampler.run([qc], shots=10000)
result = job.result()

# Assuming 'result' contains the output from the Sampler
counts = result.quasi_dists[0].nearest_probability_distribution().binary_probabilities()

# Convert counts to probabilities
total_shots = sum(counts.values())
probabilities = {state: count / total_shots for state, count in counts.items()}

# Sort states for coherent plotting (optional but helpful)
sorted_states = sorted(probabilities.items(), key=lambda x: int(x[0], 2))  # Sort by binary value

# Extract states and their probabilities for plotting
states = [int(state, 2) for state, _ in sorted_states]  # Convert binary states to integers
values = [prob for _, prob in sorted_states]

# Prepare to plot distributions for each stock
fig, axes = plt.subplots(1, len(data._tickers), figsize=(18, 6))

for i, ticker in enumerate(data._tickers):
    stock_states = {k: 0 for k in range(8)}  # Initialize all possible states with zero probability
    total_prob = 0  # Initialize total probability for normalization

    for binary_state, prob in probabilities.items():
        # Extract substate corresponding to current stock (i-th set of 3 qubits)
        substate = binary_state[3*i:3*(i+1)]
        decimal_state = int(substate, 2)
        stock_states[decimal_state] += prob
        total_prob += prob

    # Normalize probabilities if total_prob is not 1
    if total_prob != 0:
        for state in stock_states:
            stock_states[state] /= total_prob

    # Extract lower and upper bounds for current stock
    lower_bound, upper_bound = bounds[i]

    # Normalize states to their specific range
    state_keys = sorted(stock_states.keys())  # Ensuring states are sorted for plotting
    normalized_states = [(lower_bound + (upper_bound - lower_bound) * (state / max(state_keys))) for state in state_keys]
    values = [stock_states[state] for state in state_keys]

    # Plot for the i-th stock
    axes[i].bar(normalized_states, values, width=(upper_bound - lower_bound) / len(state_keys), align='edge', color='b')
    axes[i].set_title(f'{ticker} Normal Distribution')
    axes[i].set_xlabel('Normalized State (Scaled to Actual Bounds)')
    axes[i].set_ylabel('Probability')
    axes[i].grid(True)
    axes[i].set_xticks(normalized_states)  # Set x-ticks to match the state values adjusted to new bounds

plt.tight_layout()
plt.savefig("stock_distributions.png")

