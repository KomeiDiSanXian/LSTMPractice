import numpy as np
import pandas as pd
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.src.layers import Dropout
from keras.src.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath, parse_dates=['time'], index_col='time')
    data.sort_values('time', inplace=True)
    return data

def normalize_data(data):
    scaler = StandardScaler()
    data[['total', 'growth']] = scaler.fit_transform(data[['total', 'growth']])
    return data, scaler

def generate_lag_features(data, num_lags=12):
    for col in ['total', 'growth']:
        data[[f'{col}_lag_{lag}' for lag in range(1, num_lags + 1)]] = data[col].shift(periods=range(1, num_lags + 1))
    data.dropna(inplace=True)
    return data

def split_data(data, train_ratio=0.8):
    X = data.drop(columns=['total', 'growth'])
    y = data[['total', 'growth']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, shuffle=False)
    return X_train, y_train, X_test, y_test


def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps].values)
    return np.array(Xs), np.array(ys)

def build_lstm_model(input_shape, optimizer='adam'):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(2))  # Predicting two values: total and growth
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, patience=10):
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, shuffle=False, callbacks=[early_stopping])
    return history

def evaluate_model(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)

    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test)

    plt.figure(figsize=(10, 6))
    plt.plot(y_test[:, 0], label='True total')
    plt.plot(predictions[:, 0], label='Predicted total')
    plt.title('Evaluation of total predictions 100 epochs')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(y_test[:, 1], label='True growth')
    plt.plot(predictions[:, 1], label='Predicted growth')
    plt.title('Evaluation of growth predictions 100 epochs')
    plt.legend()
    plt.show()

def future_prediction(model, X_test, scaler):
    last_data = X_test[-1]
    last_data = np.expand_dims(last_data, axis=0)
    future_pred = model.predict(last_data)
    future_pred = scaler.inverse_transform(future_pred)
    return future_pred

def main():
    data = load_and_preprocess_data('data.csv')
    data, scaler = normalize_data(data)
    data = generate_lag_features(data)
    
    X_train, y_train, X_test, y_test = split_data(data)
    
    time_steps = 12
    X_train, y_train = create_dataset(X_train, y_train, time_steps)
    X_test, y_test = create_dataset(X_test, y_test, time_steps)
    
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    train_model(model, X_train, y_train, epochs=100, batch_size=64, validation_split=0.2, patience=20)
    
    model.save('lstm_model.keras')
    
    evaluate_model(model, X_test, y_test, scaler)
    
    future_pred = future_prediction(model, X_test, scaler)
    print(f'Predicted total for next month: {future_pred[0, 0]}')
    print(f'Predicted growth for next month: {future_pred[0, 1]}')

if __name__ == "__main__":
    main()
