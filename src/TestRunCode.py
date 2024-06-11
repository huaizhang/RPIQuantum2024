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

from collections import Counter


# Function to plot the data
def plot_data(data):
    # Extract elements for each position
    first_elements = [sublist[0] for sublist in data]
    second_elements = [sublist[1] for sublist in data]
    third_elements = [sublist[2] for sublist in data]

    # Count frequencies
    first_counter = Counter(first_elements)
    second_counter = Counter(second_elements)
    third_counter = Counter(third_elements)

    # Prepare data for plotting
    first_x, first_y = zip(*first_counter.items())
    second_x, second_y = zip(*second_counter.items())
    third_x, third_y = zip(*third_counter.items())

    # Plotting
    plt.figure(figsize=(18, 5))

    # First element plot
    plt.subplot(1, 3, 1)
    plt.bar(first_x, first_y)
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Frequency of 0th Element')

    # Second element plot
    plt.subplot(1, 3, 2)
    plt.bar(second_x, second_y)
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Frequency of 1st Element')

    # Third element plot
    plt.subplot(1, 3, 3)
    plt.bar(third_x, third_y)
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Frequency of 2nd Element')

    plt.tight_layout()
    plt.show()

def split_and_convert_to_decimal(binary_dict):
    result_list = []

    for binary_string, count in binary_dict.items():
        # Split the binary string into thirds
        part1 = binary_string[:3]
        part2 = binary_string[3:6]
        part3 = binary_string[6:]

        # Convert each part to a decimal number
        decimal1 = int(part1, 2)
        decimal2 = int(part2, 2)
        decimal3 = int(part3, 2)

        # Add the decimal values as a sublist to the result list
        for _ in range(count):
            result_list.append([decimal1, decimal2, decimal3])

    return result_list
def generate_quantum_normal_distribution(cov_matrix, monthly_expected_log_returns, num_qubits, stddev) -> QuantumCircuit:
    bounds = [(monthly_expected_log_returns[i] - 3*stddev[i], monthly_expected_log_returns[i] + 3*stddev[i]) for i in range(len(monthly_expected_log_returns))]
    #mvnd = NormalDistribution(num_qubits,[3.5,3.5,3.5], cov_matrix, bounds=[(0,7),(0,7),(0,7)])
    mvnd = NormalDistribution(num_qubits[0],monthly_expected_log_returns[0], cov_matrix[0][0], bounds=bounds[0])
    print(mvnd.values)
    print(mvnd.probabilities)
    #qc = QuantumCircuit(sum(num_qubits))
    #qc.append(mvnd, range(sum(num_qubits)))
    qc = QuantumCircuit(3)
    qc.append(mvnd, range(3))
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
q = 3
num_qubits = [q,q,q]

#qc = generate_quantum_normal_distribution(data._correlation,monthly_expected_log_returns,num_qubits, data._stddev)
qc = generate_quantum_normal_distribution(data._cov_matrix,monthly_expected_log_returns,num_qubits, data._stddev)

service = QiskitRuntimeService(channel="ibm_quantum", token="a9f26e0fa9d097c481d6d860162a64996acf1d166c3ef4084f80aaf8009e58627f94b74b51fe16678f97717e9330eeb1c717be836b39de3dc277bea8d0564c6c")
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

time_series = split_and_convert_to_decimal(counts)
util.binary_to_asset_values_timeseries(time_series, monthly_expected_log_returns, data._cov_matrix)
print(len(time_series))
print(time_series)
print(np.cov(np.array(time_series).T))

plot_data(time_series)

util.create_new_xlsx_monthly_dates(np.array(time_series),filename="data/output_qc.xlsx")

#creating data object for the generated data
generated_Data = StockDataProcessor( 
    start=datetime.datetime(2024, 4, 30),
    end=datetime.datetime(2190, 11, 30),
    file_path="data/output_qc.xlsx")
generated_Data.run()
generated_Data.print_stats()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, asset in enumerate(data._tickers):
    sns.histplot(time_series[:, i], bins=2**q, kde=False, ax=axes[i], color='blue')
    axes[i].set_xlabel(f'{asset} Returns')
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'{asset} Returns Distribution ({num_shots} Samples)')

fig.suptitle(f'Sample Distribution of Multivariate Normal Distribution ({num_shots} Samples)')
plt.savefig("graphs/gen_output_qc.png")