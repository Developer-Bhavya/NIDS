import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

try:
    # Load data using the file's existing header
    print("Attempting to load CICFlowMeter_out.csv...")
    data = pd.read_csv('data/CICFlowMeter_out.csv')  # Use the file's header
    print("Data loaded successfully. Columns:", data.columns.tolist())
    print("Data shape:", data.shape)

    # Check if Label is present
    if 'Label' not in data.columns:
        # Load labels if separate and Label is missing
        print("Attempting to load Label.csv...")
        labels = pd.read_csv('data/Label.csv')
        print("Labels loaded successfully. Shape:", labels.shape)

        # Merge data with labels if they are separate and row counts match
        if len(data) == len(labels):
            print("Merging data with labels...")
            data = data.merge(labels, left_index=True, right_index=True, how='left')
            print("Merge completed. Shape:", data.shape)
        else:
            print("Warning: Row counts mismatch between data and labels. Using data's Label if present.")
    else:
        print("Label column found in data. No merge needed.")

    # Define categorical columns based on the dataset
    categorical_cols = ['Protocol', 'Flow ID', 'Src IP', 'Dst IP', 'Timestamp']  # Added IP and Timestamp as categorical

    # Encode categorical features
    le = LabelEncoder()
    for col in categorical_cols:
        if col in data.columns:
            print(f"Encoding {col}...")
            data[col] = le.fit_transform(data[col].astype(str))

    # Select only numeric columns for normalization
    numerical_cols = [col for col in data.columns if col not in categorical_cols + ['Label'] and pd.api.types.is_numeric_dtype(data[col])]
    print("Normalizing numerical columns:", numerical_cols)

    # Normalize numerical features
    scaler = MinMaxScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    # Save preprocessed data
    print("Saving preprocessed data...")
    data.to_csv('data/unsw_nb15_preprocessed.csv', index=False)
    print("Preprocessing complete. Saved to data/unsw_nb15_preprocessed.csv")

except FileNotFoundError as e:
    print(f"Error: File not found - {e}. Check if files are in the data folder.")
except Exception as e:
    print(f"Error occurred: {e}")