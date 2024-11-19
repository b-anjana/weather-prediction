import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Load and preprocess the dataset
print("hi")
data = pd.read_csv('seattle-weather.csv')  # Replace with your file path
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Select relevant features
features = ['precipitation', 'temp_max', 'temp_min', 'wind']
target = ['temp_max', 'temp_min']

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features])

# Prepare data sequences
sequence_length = 30
X, y = [], []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i, 1:3])  # temp_max and temp_min

X, y = np.array(X), np.array(y)

# Split data into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define a function to build and train a custom LSTM model
def train_and_log_model(units_list, regularization_param, epochs, batch_size):
    print("hi")
    results = []
    for units in units_list:
        # Custom LSTM model definition
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(units=units[0], input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
            tf.keras.layers.LSTM(units=units[1]),
            tf.keras.layers.Dense(2)
        ])

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss='mean_squared_error')

        # Train the model
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)

        # Evaluate the model
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Rescale predictions
        y_pred_train_rescaled = scaler.inverse_transform(np.hstack((X_train[:, -1, :-2], y_pred_train)))
        y_pred_test_rescaled = scaler.inverse_transform(np.hstack((X_test[:, -1, :-2], y_pred_test)))
        y_train_rescaled = scaler.inverse_transform(np.hstack((X_train[:, -1, :-2], y_train)))
        y_test_rescaled = scaler.inverse_transform(np.hstack((X_test[:, -1, :-2], y_test)))

        # Calculate errors
        train_rmse = root_mean_squared_error(y_train_rescaled[:, -2:], y_pred_train_rescaled[:, -2:])
        test_rmse =root_mean_squared_error(y_test_rescaled[:, -2:], y_pred_test_rescaled[:, -2:])
        train_acc = 1 - train_rmse / np.mean(y_train_rescaled[:, -2:]) * 100
        test_acc = 1 - test_rmse / np.mean(y_test_rescaled[:, -2:]) * 100

        # Log the results
        results.append({
            'Units': units,
            'Regularization Parameter': regularization_param,
            'Epochs': epochs,
            'Batch Size': batch_size,
            'Training Accuracy (%)': train_acc,
            'Test Accuracy (%)': test_acc,
            'Training RMSE': train_rmse,
            'Test RMSE': test_rmse,
            'MSE':root_mean_squared_error(y_test_rescaled[:, -2:], y_pred_test_rescaled[:, -2:])
        })

    # Convert results to a DataFrame and return
    return pd.DataFrame(results)

# Define different sets of parameters to test
units_list = [(8, 8), (16, 8), (32, 16), (64, 32)]
regularization_param = 0.5
epochs = 55
batch_size = 64

# Run the training and log results
results_df = train_and_log_model(units_list, regularization_param, epochs, batch_size)


results_df.to_csv('lstm_training_results.csv', index=False)
