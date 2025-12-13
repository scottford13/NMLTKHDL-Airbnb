import csv
import numpy as np

def load_data(data_path):
    """Read data from CSV file to array"""
    rows = []
    with open(data_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader) 
        for row in reader:
            rows.append(row)

    return np.array(rows), header

def safe_convert(arr, default_val=0):
    """Convert string to float, handle empty array"""
    arr_mod = arr.copy()
    arr_mod[arr_mod == ''] = str(default_val)

    return arr_mod.astype(float)

def remove_outliers_iqr(data, col_idx):
    """Remove outliers using IQR method for a specific column"""
    # Change to float
    target_col = safe_convert(data[:, col_idx])
    
    Q1 = np.percentile(target_col, 25)
    Q3 = np.percentile(target_col, 75)
    IQR = Q3 - Q1
    
    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR

    print(f"IQR: {IQR}")
    print(f"Bounds: [{lower_bound}, {upper_bound}]")
    
    mask = (target_col > 0) & (target_col <= upper_bound)
    
    return data[mask]

def one_hot_encode(data):
    """One-hot encode a 1D array of categorical data"""
    unique_cats = np.unique(data)
    # (N, 1) == (M,) -> (N, M) boolean -> int
    return (data[:, None] == unique_cats).astype(int), unique_cats