import pandas as pd
import numpy as np

from qiskit_ibm_runtime import QiskitRuntimeService

historic_data = pd.read_excel("./data/historic_data.xlsx")
historic_data = historic_data.sort_values("Date")
historic_data = historic_data.reset_index(drop=True)
