import pandas as pd

# Load both files
original = pd.read_csv('data/CICFlowMeter_out.csv')
preprocessed = pd.read_csv('data/unsw_nb15_preprocessed.csv')

# Check number of columns
print(f"Original file columns: {len(original.columns)}, Names: {original.columns.tolist()}")
print(f"Preprocessed file columns: {len(preprocessed.columns)}, Names: {preprocessed.columns.tolist()}")

# Compare
if len(original.columns) == len(preprocessed.columns) and original.columns.equals(preprocessed.columns):
    print("Confirmation: Both files have 84 columns and matching structure.")
else:
    print("Mismatch detected. Column counts or names differ.")