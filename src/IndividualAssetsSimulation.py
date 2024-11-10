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
from qiskit.circuit.library import Initialize, Isometry
from scipy.stats import multivariate_normal


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

def split_dict_into_three(original_dict):
    # Calculate the number of items for each subdictionary
    total_items = len(original_dict)
    subdict_size = total_items // 3
    
    # Initialize subdictionaries
    subdict1, subdict2, subdict3 = {}, {}, {}
    
    # Iterator for dictionary items
    iterator = iter(original_dict.items())
    
    # Fill the first subdictionary
    for _ in range(subdict_size):
        key, value = next(iterator)
        subdict1[key] = value
    
    # Fill the second subdictionary
    for _ in range(subdict_size):
        key, value = next(iterator)
        subdict2[key] = value
    
    # Fill the third subdictionary with the remaining items
    for key, value in iterator:
        subdict3[key] = value
    #subdict1 = dict(sorted(subdict1.items(), key=lambda item: item[1]))
    #subdict2 = dict(sorted(subdict2.items(), key=lambda item: item[1]))
    #subdict3 = dict(sorted(subdict3.items(), key=lambda item: item[1]))

    return subdict1, subdict2, subdict3
def generate_quantum_normal_distribution(cov_matrix, monthly_expected_log_returns, num_qubits, stddev) -> QuantumCircuit:
    bounds = [(monthly_expected_log_returns[i] - 3*stddev[i], monthly_expected_log_returns[i] + 3*stddev[i]) for i in range(len(monthly_expected_log_returns))]
    mvnd = NormalDistribution(num_qubits,monthly_expected_log_returns, cov_matrix, bounds=bounds )
    qc = QuantumCircuit(sum(num_qubits))
    qc.append(mvnd, range(sum(num_qubits)))
    qc.measure_all()
    return qc

def generate_quantum_normal_distribution_all_assets(expected_log_returns, variances, num_qubits, stddevs):
    # Create a list to hold the quantum circuits for each asset
    quantum_circuits = []
    
    # Iterate over each asset
    i=0
    for i in range(len(expected_log_returns)):
        expected_log_return = expected_log_returns[i]
        print(expected_log_return)
        variance = variances[i]
        stddev = stddevs[i]
        
        # Calculate the bounds for the normal distribution
        lower_bound = expected_log_return - 3 * stddev
        upper_bound = expected_log_return + 3 * stddev
        bounds = [(lower_bound, upper_bound)]
        
        # Create the normal distribution circuit for the given parameters
        #mvnd = NormalDistribution(num_qubits=3, mu=[expected_log_return], sigma=[[variance]], bounds=bounds)
        inner = QuantumCircuit(3)
        x = np.linspace(lower_bound, upper_bound, num=2**3)  # type: Any
        # compute the normalized, truncated probabilities
        probabilities = multivariate_normal.pdf(x, expected_log_return, variance)
        normalized_probabilities = probabilities / np.sum(probabilities)

        initialize = Initialize(np.sqrt(normalized_probabilities))
        circuit = initialize.gates_to_uncompute().inverse()
        inner.compose(circuit, inplace=True)
        qc = QuantumCircuit(3)

        qc.append(inner.to_gate(), inner.qubits)
        # Initialize a quantum circuit
                
        # Measure all qubits
        qc.measure_all()
        
        # Add the quantum circuit to the list
        quantum_circuits.append(qc)
    
    return quantum_circuits


data = StockDataProcessor(
    start=datetime.datetime(2004, 4, 30),
    end=datetime.datetime(2024, 3, 31),
    file_path="../data/historic_data.xlsx"
)
data.run()
data.print_stats()

annual_expected_returns = np.array([0.1, 0.1, 0.06]) # Standard default probabilities
monthly_expected_log_returns = np.log(1 + annual_expected_returns) / 12
num_qubits = [3,3,3]

qc_array = generate_quantum_normal_distribution_all_assets(monthly_expected_log_returns, np.diag(data._cov_matrix), num_qubits, data._stddev)

service = QiskitRuntimeService(channel="ibm_quantum", token="YOUR_API_KEY")
backend = service.backend("ibm_rensselaer")
pm = generate_preset_pass_manager(backend=backend,optimization_level=1) #transpilation readable for quantum computer
all_asset_samples = []
i = 0
for qc in qc_array:
    isa_circuit = pm.run(qc)
    isa_circuit.depth()

    sampler = SamplerV2(backend=backend)
    job = sampler.run([isa_circuit], shots=2000)
    print(job.job_id())
    counts = job.result()[0].data.meas.get_counts()
    print(counts)
    print(len(counts))

    total_counts = sum(counts.values())
    quasi_probabilities = {key: value / total_counts for key, value in counts.items()}
    print(quasi_probabilities)
    nearest_pd = nearest_probability_distribution(quasi_probabilities)
    print(nearest_pd)

    binary_samples = [k for k, v in nearest_pd.items() for _ in range(int(v * 2000))]
    asset_samples = np.array([util.binary_to_asset_values_qc(sample, 3, [monthly_expected_log_returns[i]], data._cov_matrix) for sample in binary_samples])
    all_asset_samples.append(asset_samples)
    i += 1

all_asset_samples = np.array(all_asset_samples)
util.create_new_xlsx_monthly_dates(all_asset_samples,filename="../data/output_qc.xlsx")

for i, asset_samples in enumerate(all_asset_samples):
    # Reshape or flatten the asset samples as needed
    flattened_samples = asset_samples.reshape(-1)  # Adjust this based on the actual shape you need
    plt.figure()
    sns.histplot(flattened_samples, bins=16, kde=False, color='blue')
    plt.xlabel(f'Asset {i+1} Returns')
    plt.ylabel('Frequency')
    plt.title(f'Asset {i+1} Returns Distribution')
    plt.savefig(f"../graphs/asset_{i+1}_returns_distribution.png")
    plt.close()


#creating data object for the generated data
generated_Data = StockDataProcessor( 
    start=datetime.datetime(2024, 4, 30),
    end=datetime.datetime(2044, 11, 30),
    file_path="../data/output_qc.xlsx")
generated_Data.run()
print("[GENERATED DATA STATS]")
generated_Data.print_stats()    