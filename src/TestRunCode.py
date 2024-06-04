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

def nearest_probability_distribution(quasi_probabilities):
    """Takes a quasiprobability distribution and maps
    it to the closest probability distribution as defined by
    the L2-norm.


    Returns:
        ProbDistribution: Nearest probability distribution.
        float: Euclidean (L2) distance of distributions.

    Notes:
        Method from Smolin et al., Phys. Rev. Lett. 108, 070502 (2012).
    """
    sorted_probs = dict(sorted(quasi_probabilities.items(), key=lambda item: item[1]))
    num_elems = len(sorted_probs)
    new_probs = {}
    beta = 0
    diff = 0
    for key, val in sorted_probs.items():
        temp = val + beta / num_elems
        if temp < 0:
            beta += val
            num_elems -= 1
            diff += val * val
        else:
            diff += (beta / num_elems) * (beta / num_elems)
            new_probs[key] = sorted_probs[key] + beta / num_elems
    return new_probs

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
q = 4
num_qubits = [q,q,q]

qc = generate_quantum_normal_distribution(data._cov_matrix,monthly_expected_log_returns,num_qubits, data._stddev)

service = QiskitRuntimeService(channel="ibm_quantum", token="71fa7066cbcdde5ec3aa7ad8963dee340366302bb8575c6241cd54b28bafd6a9cf5331458c1bbd8a64abab7b98b202edf2102e77de87cd5e0d6b3e7446eb489e")
backend = service.backend("ibm_rensselaer")
pm = generate_preset_pass_manager(backend=backend,optimization_level=1) #transpilation readable for quantum computer
isa_circuit = pm.run(qc)
isa_circuit.depth() 

num_shots = 2000

sampler = SamplerV2(backend=backend)
job = sampler.run([isa_circuit], shots=num_shots)
print(job.job_id())
counts = job.result()[0].data.meas.get_counts()
print(counts)
print(len(counts))

total_counts = sum(counts.values())
quasi_probabilities = {key: value / total_counts for key, value in counts.items()}
print(quasi_probabilities)
nearest_pd = nearest_probability_distribution(quasi_probabilities)
print(nearest_pd)

binary_samples = [k for k, v in nearest_pd.items() for _ in range(int(v * num_shots))]

asset_samples = np.array([util.binary_to_asset_values(sample, num_qubits, monthly_expected_log_returns, data._cov_matrix) for sample in binary_samples])

util.create_new_xlsx_monthly_dates(asset_samples,filename="data/output_qc.xlsx")

#creating data object for the generated data
generated_Data = StockDataProcessor( 
    start=datetime.datetime(2024, 4, 30),
    end=datetime.datetime(2170, 11, 30),
    file_path="data/output_qc.xlsx")
generated_Data.run()
generated_Data.print_stats()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, asset in enumerate(data._tickers):
    sns.histplot(asset_samples[:, i], bins=2**q, kde=False, ax=axes[i], color='blue')
    axes[i].set_xlabel(f'{asset} Returns')
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'{asset} Returns Distribution ({num_shots} Samples)')

fig.suptitle(f'Sample Distribution of Multivariate Normal Distribution ({num_shots} Samples)')
plt.savefig("graphs/gen_output_qc.png")