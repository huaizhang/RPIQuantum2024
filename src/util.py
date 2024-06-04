import openpyxl
import calendar
import os
import datetime
import numpy as np
from scipy.stats import norm
from scipy.linalg import cholesky

def binary_to_asset_values(binary_sample, num_qubits, mu, sigma):
    asset_values = []
    start_idx = 0 # Index to keep track of qubit groups
    for i, qubits in enumerate(num_qubits):
        end_idx = start_idx + qubits # End index for current asset's qubits
        asset_bin = binary_sample[start_idx:end_idx] # Get the binary string
        # Convert binary to float in [0, 1] range and scale to asset return
        asset_value = int(asset_bin, 2) / (2**qubits - 1)
        #z_value = np.
        from scipy.stats import norm
        z_value = norm.ppf(asset_value)
        value = mu[i] + np.sqrt(sigma[i][i]) * (6 * asset_value - 3) #(2.3 * z_value) #(4 * asset_value - 2)# 
        asset_values.append(value)
        start_idx = end_idx # Move to the next set of qubits
    return asset_values

def create_new_xlsx_monthly_dates(load_data, filename, secondTime = 0):

    def month_increment(start_date, num_months):
        # Calculate the new month and year
        new_month = (start_date.month + num_months - 1) % 12 + 1
        new_year = start_date.year + (start_date.month + num_months - 1) // 12
        
        # Calculate the last day of the new month
        last_day_of_month = calendar.monthrange(new_year, new_month)[1]
        
        # Ensure the new day is the last valid day of the new month if the original day doesn't exist in the new month
        new_day = min(start_date.day, last_day_of_month)
        return datetime.date(new_year, new_month, new_day)
    start_date = datetime.date(2024, 4, 30)
    monthly_dates = [month_increment(start_date, i) for i in range(load_data.shape[0])]

    if os.path.exists(filename):
            wb = openpyxl.load_workbook(filename)
    else:
        wb = openpyxl.Workbook()

    ws = wb.active
    ws.delete_rows(1, ws.max_row)
    ws.append(['Date', '^GSPC', '^ACWX', '^GLAB.L'])

    for i, row in enumerate(load_data):
        ws.append([monthly_dates[i].strftime('%Y-%m-%d')] + row.tolist())
    wb.save(filename)

