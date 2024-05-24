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
data.print_stats()

annual_expected_returns = np.array([0.1, 0.1, 0.06]) # Standard default probabilities
monthly_expected_log_returns = np.log(1 + annual_expected_returns) / 12
num_qubits = [3,3,3]

#util.run_numpy_simulated_returns(data._cov_matrix,monthly_expected_log_returns)
qc = generate_quantum_normal_distribution(data._cov_matrix,monthly_expected_log_returns,num_qubits, data._stddev)

# Sample using the Sampler primitive
sampler = Sampler()
job = sampler.run([qc], shots=2000)
result = job.result()

# Extract quasi-probabilities and convert them to binary-encoded samples
counts = result.quasi_dists[0].nearest_probability_distribution().binary_probabilities()
binary_samples = [k for k, v in counts.items() for _ in range(int(v * 2000))]

# Apply the conversion function to all samples
asset_samples = np.array([util.binary_to_asset_values(sample, num_qubits, monthly_expected_log_returns, data._cov_matrix) for sample in binary_samples])
#creating file for storing generated data
print("ASSET SAMPLES")
print(asset_samples)
util.create_new_xlsx_monthly_dates(asset_samples,filename="data/output.xlsx")
"""
I've edited the class to take in a dataframe and fill the self._data with that data instead or it can take a file (either/or)
The percentage_output excel file does not fill and I don't know why yet, it's probably a stupid issue 
I first want to turn our new percent returns back into log returns and compare it to the original log returns so then we can
verify that we are doing it properly
"""
#creating data object for the generated data
generated_Data = StockDataProcessor( 
    start=datetime.datetime(2004, 4, 30),
    end=datetime.datetime(2170, 11, 30),
    file_path="data/output.xlsx")
generated_Data.run()
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
# Convert generated data to percent returns
simulated_percent_returns = np.array(np.exp(generated_Data._data) - 1)
# Convert the simulated percent returns to a DataFrame
#simulated_percent_returns_df = pd.DataFrame(simulated_percent_returns)

#print(simulated_percent_returns_df)
# Save the simulated percent returns to Excel
print("COUNT IS \n\n\n\n",generated_Data._data.count())
util.create_new_xlsx_monthly_dates(simulated_percent_returns, filename="data/percentage_output.xlsx",secondTime=1)

# Load the generated percent data
generated_percent_data = StockDataProcessor(
    start=datetime.datetime(2004, 4, 30),
    end=datetime.datetime(2170, 11, 30),
    file_path="data/percentage_output.xlsx"
)
generated_percent_data.run()

# Ensure the loaded data has no empty rows and columns
#generated_percent_data._data.dropna(how='all', inplace=True)
#generated_percent_data._data.dropna(axis=1, how='all', inplace=True)

# Print the cleaned data to check
#print(generated_percent_data._data.head())

generated_percent_data.print_stats()

# Plot the generated percent returns
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, asset in enumerate(generated_percent_data._data.columns):
    sns.histplot(generated_percent_data._data[asset], bins=16, kde=False, ax=axes[i], color='green')
    axes[i].set_xlabel(f'{asset} Percent Returns')
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'{asset} Percent Returns Distribution')

fig.suptitle('Generated Percent Returns Distribution')
plt.savefig("graphs/gen_percent_output.png")
target_weights = {
    '^GSPC': 0.30,
    '^ACWX': 0.30,
    '^GLAB.L': 0.40
}

frequency = 'quarterly'  # Choose from 'monthly', 'quarterly', 'semi-annual', 'annual'

final_weights = generated_percent_data.rebalance_portfolio_over_time(target_weights, frequency)
print("Final Portfolio Weights:", {stock: f'{weight:.6f}' for stock, weight in final_weights.items()})
print(generated_percent_data._data)
