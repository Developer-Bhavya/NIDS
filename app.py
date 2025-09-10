import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from transformers import BartForConditionalGeneration, BartTokenizer
import requests
from dotenv import load_dotenv
import os

# Load environment variables for VirusTotal API key
load_dotenv()
vt_api_key = os.getenv('VIRUSTOTAL_API_KEY')

# Initialize BART for summarization
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

st.title("Network Intrusion Detection System")

# File uploader for dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

if uploaded_file is not None:
    # Load uploaded dataset
    data = pd.read_csv(uploaded_file)
    st.write("Dataset uploaded successfully. Shape:", data.shape)

    # Preprocess the uploaded dataset
    categorical_cols = ['Protocol', 'Flow ID', 'Src IP', 'Dst IP', 'Timestamp']
    le = LabelEncoder()
    for col in categorical_cols:
        if col in data.columns:
            data[col] = le.fit_transform(data[col].astype(str))

    numerical_cols = [col for col in data.columns if col not in categorical_cols + ['Label']]
    scaler = MinMaxScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols].select_dtypes(include=[np.number]))
    st.write("Dataset preprocessed. Ready for analysis.")

    # Option to train autoencoder on uploaded dataset
    if st.button("Train Autoencoder on Uploaded Dataset"):
        X = data.drop('Label', axis=1).values if 'Label' in data.columns else data.values
        X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

        input_dim = X_train.shape[1]
        model = Sequential([
            Dense(32, activation='relu', input_shape=(input_dim,)),
            Dense(16, activation='relu'),
            Dense(8, activation='relu'),
            Dense(16, activation='relu'),
            Dense(32, activation='relu'),
            Dense(input_dim, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, X_train, epochs=10, batch_size=32, validation_data=(X_val, X_val))  # Reduced epochs for speed
        model.save('models/autoencoder.keras')
        st.write("Autoencoder trained and saved as models/autoencoder.keras")

    # Load the autoencoder model
    autoencoder = load_model('models/autoencoder.keras')

    if st.button("Analyze Network Traffic"):
        X = data.drop('Label', axis=1).values[:100] if 'Label' in data.columns else data.values[:100]
        X_pred = autoencoder.predict(X)
        mse = np.mean(np.power(X - X_pred, 2), axis=1)
        threshold = np.percentile(mse, 95)
        anomalies = mse > threshold
        st.write(f"**Detected {np.sum(anomalies)} anomalies** in network traffic sample")
        st.write(f"Threshold: {threshold}")

    if st.button("Show Anomaly Distribution"):
        X = data.drop('Label', axis=1).values[:1000] if 'Label' in data.columns else data.values[:1000]
        X_pred = autoencoder.predict(X)
        mse = np.mean(np.power(X - X_pred, 2), axis=1)
        st.write("Anomaly Scores:", mse)

    # Threat Intelligence Section
    st.header("Threat Intelligence Analysis")
    hash_input = st.text_input("Enter File Hash (e.g., MD5)", "d41d8cd98f00b204e9800998ecf8427e")
    if st.button("Analyze Threat"):
        # Query VirusTotal
        url = f"https://www.virustotal.com/api/v3/files/{hash_input}"
        headers = {"x-apikey": vt_api_key}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            vt_data = response.json()
            # Summarize with BART
            text = str(vt_data)
            inputs = bart_tokenizer(text, max_length=1024, return_tensors='pt', truncation=True)
            summary_ids = bart_model.generate(inputs['input_ids'], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
            summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            st.write("**Threat Summary (LLM-Powered):**", summary)
        else:
            st.error("Error querying VirusTotal. Check API key or hash.")
else:
    st.write("Please upload a dataset to start.")