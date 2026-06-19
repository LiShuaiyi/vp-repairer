import pandas as pd

# Load the two CSV files
file1 = "highD_inD_ttc_repair_np_der.csv"  # Replace with your file path
file2 = "highD_inD_ttc_sampling.csv"       # Replace with your file path

# Read the CSVs into DataFrames
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Ensure consistent column naming for the first two columns
df2.rename(columns={df2.columns[0]: df1.columns[0], df2.columns[1]: df1.columns[1]}, inplace=True)

# Identify common rows based on the first two columns
common_cols = [df1.columns[0], df1.columns[1]]
common_rows_file1 = df1[df1[common_cols].apply(tuple, axis=1).isin(df2[common_cols].apply(tuple, axis=1))]
common_rows_file2 = df2[df2[common_cols].apply(tuple, axis=1).isin(df1[common_cols].apply(tuple, axis=1))]

# Save the filtered results to separate files
common_rows_file1.to_csv("highD_inD_ttc_repair_np_der_common.csv", index=False)
common_rows_file2.to_csv("highD_inD_ttc_sampling_common.csv", index=False)

print("Filtered file1 saved to 'highD_inD_ttc_repair_np_der_common.csv'")
print("Filtered file2 saved to 'highD_inD_ttc_sampling_common.csv'")
