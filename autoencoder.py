import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# Load preprocessed data
data = pd.read_csv('data/unsw_nb15_preprocessed.csv')
X = data.drop('Label', axis=1).values
X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

# Define autoencoder model
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

# Train the model
model.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_val, X_val))

# Save the model
model.save('models/autoencoder.keras')
print("Autoencoder model saved to models/autoencoder.keras")