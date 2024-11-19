import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data = pd.read_csv('seattle-weather.csv')
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

features = ['precipitation', 'temp_max', 'temp_min', 'wind']
target = ['temp_max', 'temp_min']

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features])

sequence_length = 30
X, y = [], []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i, 1:3])

X, y = np.array(X), np.array(y)

train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

def train_and_log_model(units_list, regularization_param, epochs, batch_size):
    results = []
    for units in units_list:
        for regularization_param in regularization_params:
                for epoch in epochs:
                    for batch_size in batch_sizes:
                        model = tf.keras.Sequential([
                            tf.keras.layers.LSTM(units=units[0], input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
                            tf.keras.layers.LSTM(units=units[1]),
                            tf.keras.layers.Dense(2)
                        ])

                        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                                    loss='mean_squared_error')

                        history = model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, validation_split=0.1, verbose=0)

                        y_pred_train = model.predict(X_train)
                        y_pred_test = model.predict(X_test)

                        y_pred_train_rescaled = scaler.inverse_transform(np.hstack((X_train[:, -1, :-2], y_pred_train)))
                        y_pred_test_rescaled = scaler.inverse_transform(np.hstack((X_test[:, -1, :-2], y_pred_test)))
                        y_train_rescaled = scaler.inverse_transform(np.hstack((X_train[:, -1, :-2], y_train)))
                        y_test_rescaled = scaler.inverse_transform(np.hstack((X_test[:, -1, :-2], y_test)))

                        train_rmse = np.sqrt(mean_squared_error(y_train_rescaled[:, -2:], y_pred_train_rescaled[:, -2:]))
                        test_rmse = np.sqrt(mean_squared_error(y_test_rescaled[:, -2:], y_pred_test_rescaled[:, -2:]))
                        train_acc = 1 - train_rmse / np.mean(y_train_rescaled[:, -2:]) * 100
                        test_acc = 1 - test_rmse / np.mean(y_test_rescaled[:, -2:]) * 100

                        results.append({
                            'Units': units,
                            'Regularization Parameter': regularization_param,
                            'Epochs': epoch,
                            'Batch Size': batch_size,
                            'Training Accuracy (%)': train_acc,
                            'Test Accuracy (%)': test_acc,
                            'Training RMSE': train_rmse,
                            'Test RMSE': test_rmse,
                            'MSE': mean_squared_error(y_test_rescaled[:, -2:], y_pred_test_rescaled[:, -2:])
                        })

    return pd.DataFrame(results)

units_list = [(8, 8), (16, 8), (32, 16), (64, 32)]
regularization_params = [0.1, 0.5]
epochs = [30, 50]
batch_sizes = [32,64]

results_df = train_and_log_model(units_list, regularization_params, epochs, batch_sizes)

results_df.to_csv('lstm_training_results.csv', index=False)
