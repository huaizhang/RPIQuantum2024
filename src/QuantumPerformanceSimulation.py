import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from qiskit_finance.circuit.library.probability_distributions import NormalDistribution
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
import util
from StockDataProcessor import StockDataProcessor

def calculate_monthly_returns(data):
    monthly_returns = data.iloc[:, 1:4]

    # Compute cumulative product of returns
    monthly_returns = (1 + monthly_returns).cumprod()

    # Extract dates and identify month ends
    first_column_name = data.columns[0]
    dates = data[first_column_name]
    dates = pd.to_datetime(dates, errors='coerce')

    # Drop rows with invalid dates (if any)
    valid_indices = dates.dropna().index
    monthly_returns = monthly_returns.loc[valid_indices]
    dates = dates.dropna()

    month_ends = dates.groupby([dates.dt.year, dates.dt.month]).transform('max')
    month_end_indices = np.where(dates == month_ends)[0]

    # Filter the returns to only include month ends
    monthly_returns = monthly_returns.iloc[month_end_indices]

    # Add an initial value of 1 for calculation purposes
    initial_row = pd.DataFrame([[1] * monthly_returns.shape[1]], columns=monthly_returns.columns)
    monthly_returns = pd.concat([initial_row, monthly_returns])

    # Reset the index before calculating percentage change
    monthly_returns = monthly_returns.reset_index(drop=True)

    # Calculate the percentage change to get monthly returns
    monthly_returns = monthly_returns.pct_change()

    # Drop any rows with NaN values resulting from the percentage change calculation
    monthly_returns = monthly_returns.dropna()

    # Reset the index for the final output
    monthly_returns = monthly_returns.reset_index(drop=True)

    return monthly_returns

def generate_quantum_normal_distribution(cov_matrix, monthly_expected_log_returns, num_qubits, stddev) -> QuantumCircuit:
    # Calculate bounds as +- 3 standard deviations around the mean
    bounds = [(monthly_expected_log_returns[i] - 3*stddev[i], monthly_expected_log_returns[i] + 3*stddev[i]) for i in range(len(monthly_expected_log_returns))]
    mvnd = NormalDistribution(num_qubits,monthly_expected_log_returns, cov_matrix, bounds=bounds )

    qc = QuantumCircuit(sum(num_qubits))
    qc.append(mvnd, range(sum(num_qubits)))
    qc.measure_all()
    return qc

# Example usage of class
data = StockDataProcessor(
    start=datetime.datetime(2004, 4, 30),
    end=datetime.datetime(2024, 3, 31),
    file_path="./data/historic_data.xlsx"
)
data.run()
print("[ORIGINAL DATA STATS]")
data.print_stats()

annual_expected_returns = np.array([0.1, 0.1, 0.06]) # Standard default probabilities
monthly_expected_log_returns = np.log(1 + annual_expected_returns) / 12
num_qubits = [3,3,3]

#util.run_numpy_simulated_returns(data._cov_matrix,monthly_expected_log_returns)
qc = generate_quantum_normal_distribution(data._cov_matrix,monthly_expected_log_returns,num_qubits, data._stddev)

# Sample using the Sampler primitive
sampler = Sampler()
job = sampler.run([qc], shots=240)
result = job.result()

# Extract quasi-probabilities and convert them to binary-encoded samples
counts = result.quasi_dists[0].nearest_probability_distribution().binary_probabilities()
binary_samples = [k for k, v in counts.items() for _ in range(int(v * 240))]

# Apply the conversion function to all samples
asset_samples = np.array([util.binary_to_asset_values(sample, num_qubits, monthly_expected_log_returns, data._cov_matrix) for sample in binary_samples])
#creating file for storing generated data
util.create_new_xlsx_monthly_dates(asset_samples,filename="data/output.xlsx")

#creating data object for the generated data
generated_Data = StockDataProcessor( 
    start=datetime.datetime(2024, 4, 30),
    end=datetime.datetime(2044, 11, 30),
    file_path="data/output.xlsx")
generated_Data.run()
print("[GENERATED DATA STATS]")
generated_Data.print_stats()

# Plot the sampled distribution
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, asset in enumerate(data._tickers):
    sns.histplot(asset_samples[:, i], bins=16, kde=False, ax=axes[i], color='blue')
    axes[i].set_xlabel(f'{asset} Returns')
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'{asset} Returns Distribution (120 Samples)')

fig.suptitle('Sample Distribution of Multivariate Normal Distribution (120 Samples)')
plt.savefig("graphs/gen_output.png")
simulated_percent_returns = np.array(np.exp(asset_samples) - 1)

util.create_new_xlsx_monthly_dates(simulated_percent_returns, filename="data/percentage_output.xlsx",secondTime=1)

# Load the generated percent data
generated_percent_data = StockDataProcessor(
    start=datetime.datetime(2024, 4, 30),
    end=datetime.datetime(2044, 11, 30),
    file_path="data/percentage_output.xlsx"
)
generated_percent_data.run_nonlog()
"""
portfolio_returns = generated_percent_data._data.dot(annual_expected_returns)

annual_portfolio_return = (1 + portfolio_returns).prod() ** (12 / generated_Data._data.shape[0]) - 1
annual_portfolio_volatility = np.std(portfolio_returns) * np.sqrt(12)
risk_free_rate = 0.00
sharpe_ratio = (annual_portfolio_return - risk_free_rate) / annual_portfolio_volatility
# calculate the maximum drawdown
cumulative_returns = (1 + portfolio_returns).cumprod()
max_drawdown = np.min(
    cumulative_returns / np.maximum.accumulate(cumulative_returns) - 1
)
# calculate the Calmar ratio
calmar_ratio = annual_portfolio_return / max_drawdown

print("annual_portfolio_return: ",annual_portfolio_return)
print("annual_portfolio_volatility: ",annual_portfolio_volatility)
print("sharpe_ratio: ",sharpe_ratio)
print("max_drawdown: ", max_drawdown)
print("calmar_ratio: ",calmar_ratio)
Plot the generated percent returns
"""
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, asset in enumerate(generated_percent_data._data.columns):
    sns.histplot(generated_percent_data._data[asset], bins=16, kde=False, ax=axes[i], color='green')
    axes[i].set_xlabel(f'{asset} Percent Returns')
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'{asset} Percent Returns Distribution')

fig.suptitle('Generated Percent Returns Distribution')
plt.savefig("graphs/gen_percent_output.png")
from decimal import Decimal

target_weights = {
    '^GSPC': Decimal('0.3000000'),
    '^ACWX': Decimal('0.3000000'),
    '^GLAB.L': Decimal('0.4000000')
}
frequency = 'quarterly'  # Choose from 'monthly', 'quarterly', 'semi-annual', 'annual'
generated_percent_data.rebalance_portfolio_over_time(target_weights, frequency='monthly', printbool=True)
