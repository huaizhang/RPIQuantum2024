import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from qiskit_finance.circuit.library.probability_distributions import NormalDistribution
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_finance.circuit.library.probability_distributions import NormalDistribution
from qiskit import QuantumCircuit
import util
from StockDataProcessor import StockDataProcessor


def generate_quantum_normal_distribution(cov_matrix, monthly_expected_log_returns, num_qubits, stddev) -> QuantumCircuit:
    bounds = [(monthly_expected_log_returns[i] - 3*stddev[i], monthly_expected_log_returns[i] + 3*stddev[i]) for i in range(len(monthly_expected_log_returns))]
    mvnd = NormalDistribution(num_qubits,monthly_expected_log_returns, cov_matrix, bounds=bounds )
    qc = QuantumCircuit(sum(num_qubits))
    qc.append(mvnd, range(sum(num_qubits)))
    qc.measure_all()
    return qc

data = StockDataProcessor(
    start=datetime.datetime(2004, 4, 30),
    end=datetime.datetime(2024, 3, 31),
    file_path="./data/historic_data.xlsx"
)
data.run()
data.print_stats()

annual_expected_returns = np.array([0.1, 0.1, 0.06]) # Standard default probabilities
monthly_expected_log_returns = np.log(1 + annual_expected_returns) / 12
print(monthly_expected_log_returns)
num_qubits = [3,3,3]

qc = generate_quantum_normal_distribution(data._cov_matrix,monthly_expected_log_returns,num_qubits, data._stddev)

service = QiskitRuntimeService(channel="ibm_quantum", token="01dce3ab39fff4fcd01e41140ecd8e33c4145c3612dd02593c9c1b476ed339084e98b2cb6c1d088d86faf2ac26a49edb7d21344c33ac53fb0a40675b28857390")
backend = service.backend("ibm_rensselaer")
pm = generate_preset_pass_manager(backend=backend, optimization_level=3) #transpilation readable for quantum computer
isa_circuit = pm.run(qc)
isa_circuit.depth() 

sampler = SamplerV2(backend=backend)
job = sampler.run([isa_circuit], shots=1000)
print(job.job_id())
result = job.result()
counts = job.result()[0].data.meas.get_counts()
total_shots = sum(counts.values())
print(total_shots)
probability_distribution = {state: (count / total_shots) * 100 for state, count in counts.items()}
# Verify the sum of all values adds up to 100
total_percentage = sum(probability_distribution.values())
print(f"Total percentage: {total_percentage}")

asset_samples = np.array([util.binary_to_asset_values(sample, num_qubits, monthly_expected_log_returns, data._cov_matrix) for sample in probability_distribution])
util.create_new_xlsx_monthly_dates(asset_samples,filename="data/output_qc.xlsx")

#creating data object for the generated data
generated_Data = StockDataProcessor( 
    start=datetime.datetime(2004, 4, 30),
    end=datetime.datetime(2024, 3, 31),
    file_path="data/output_qc.xlsx")
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
plt.savefig("graphs/gen_output_qc.png")