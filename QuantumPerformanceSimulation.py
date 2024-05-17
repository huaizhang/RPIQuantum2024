import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_finance.circuit.library.probability_distributions import NormalDistribution
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
import util
from stockDataProcessor import StockDataProcessor

def run_numpy_simulated_returns(cov_matrix, monthly_expected_log_returns):
    simulated_log_returns = np.random.multivariate_normal(
        monthly_expected_log_returns, cov_matrix, 360
    )
    simulated_log_returns = pd.DataFrame(
        simulated_log_returns,
        columns=["US Equities", "International Equities", "Global Fixed Income"],
    )
    # Set up the matplotlib figure
    plt.figure(figsize=(18, 6))
    # Draw a subplot for each asset category
    for i, column in enumerate(simulated_log_returns.columns, 1):
        plt.subplot(1, 3, i)  # 1 row, 3 columns, ith subplot
        sns.histplot(simulated_log_returns[column], bins=30, kde=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel('Log Returns')
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig("graphs/expectedoutput.png")

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

run_numpy_simulated_returns(data._cov_matrix,monthly_expected_log_returns)
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
util.create_new_xslx_monthly_dates(asset_samples,filename="data/output.xlsx")

#creating data object for the generated data
generated_Data = StockDataProcessor( 
    start=datetime.datetime(2004, 4, 30),
    end=datetime.datetime(2024, 3, 31),
    file_path="data/output.xlsx")
generated_Data.run()
generated_Data.print_stats()

# Plot the sampled distribution
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, asset in enumerate(data._tickers):
    sns.histplot(asset_samples[:, i], bins=15, kde=False, ax=axes[i], color='blue')
    axes[i].set_xlabel(f'{asset} Returns')
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'{asset} Returns Distribution (120 Samples)')

fig.suptitle('Sample Distribution of Multivariate Normal Distribution (120 Samples)')
plt.savefig("graphs/gen_output.png")

