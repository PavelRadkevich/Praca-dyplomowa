import inspect
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from flask import Blueprint, jsonify, request, url_for
from keras.callbacks import Callback
from keras.layers import LSTM, Dense, Input
from keras.models import Sequential, load_model
from keras.src.layers import Dropout
from keras.src.optimizers import Adam
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split

import apps.predict.api_requests
from apps import socketio
from apps.config import LSTMConfig

# Blueprint
predict_bp = Blueprint('predict', __name__)

# Retrieve all API methods
all_methods = {
    name: func
    for name, func in inspect.getmembers(apps.predict.api_requests, inspect.isfunction)
}


@predict_bp.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symbol, start_year, end_year, parameters = data['company'], data['startYear'], data['endYear'], data[
            'parameters']
        socketio.emit('progress', {'status': 'Ładujemy ceny akcji i dywidendy...'}, namespace='/')

        stock_prices, dividends, results, error_response, status_code = fetch_data(symbol, start_year, end_year,
                                                                                   parameters)
        if error_response:
            return error_response, status_code
        socketio.emit('progress', {'status': 'Przygotowujemy dane...'}, namespace='/')
        X, y, all_features, stock_prices = prepare_data(stock_prices, dividends, results, parameters)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=LSTMConfig.test_size, random_state=42)

        model = build_lstm_model((X.shape[1], X.shape[2]))
        socketio.emit('progress', {'status': f'Trenujemy model (Epoka 0/{LSTMConfig.epochs})...'}, namespace='/')
        model.fit(X_train, y_train, batch_size=LSTMConfig.batch_size, epochs=LSTMConfig.epochs,
                  validation_split=LSTMConfig.validation_split, callbacks=[TrainingProgressCallback()])

        results = {"30_days": 0, "60_days": 0, "90_days": 0, "confusion_matrix": "None", "loss": 0,
                   "accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "auc": 0, "roc_auc_url": "",
                   "train_test_url": ""}
        socketio.emit('progress', {'status': f'Robimy prognozę...'}, namespace='/')
        results = predict_price(model, all_features, X, results)

        socketio.emit('progress', {'status': f'Zbieramy metryki...'}, namespace='/')
        optimizer = Adam(learning_rate=LSTMConfig.learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=LSTMConfig.metrics)
        results = evaluate_model(model, X_test, y_test, results, X_train, stock_prices)

        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": f"Failed to process request: {str(e)}"}), 500


def fetch_data(symbol, start_year, end_year, parameters):
    """Loads stock price, dividend, and other parameter data."""
    stock_prices, trading_days, dividends, last_dividend_date = all_methods["get_stock_data_with_dividends"](symbol,
                                                                                                             start_year,
                                                                                                             end_year)
    socketio.emit('last_date', {'status': str(last_dividend_date)[:-9]}, namespace='/')

    results = {}
    for param in parameters:
        socketio.emit('progress', {'status': f'Ładujemy parametr {param}...'}, namespace='/')
        method_name = f'get_{param}'
        if method_name in all_methods:
            try:
                results[param] = all_methods[method_name](symbol, int(start_year), int(end_year), trading_days)
                if not isinstance(results[param], list):
                    results[param] = results[param].tolist()
            except Exception as e:
                print(f"Error fetching {param}: {e}")
                return None, jsonify({"error": f"Unexpected error: {str(e)}"}), 500
        else:
            return None, jsonify({"error": f"Method {method_name} not found"}), 400

    return stock_prices, dividends, results, None, 200


def prepare_data(stock_prices, dividends, results, parameters, time_step=LSTMConfig.time_step):
    """Prepares data for the LSTM model."""
    stock_prices = np.array(stock_prices, dtype=float).reshape(-1, 1)
    dividends = np.array(dividends, dtype=float).reshape(-1, 1)
    additional_features = np.column_stack([
        np.array(results[param], dtype=float).reshape(-1, 1) for param in parameters
    ])
    all_features = np.hstack((stock_prices, dividends, additional_features))

    X, y_30, y_60, y_90 = [], [], [], []
    for i in range(len(all_features) - time_step - 90):
        X.append(all_features[i:i + time_step, :])
        y_30.append(int(stock_prices[i + time_step + 30] > stock_prices[i + time_step]))
        y_60.append(int(stock_prices[i + time_step + 60] > stock_prices[i + time_step]))
        y_90.append(int(stock_prices[i + time_step + 90] > stock_prices[i + time_step]))

    X, y_30, y_60, y_90 = np.array(X), np.array(y_30), np.array(y_60), np.array(y_90)
    return X, np.column_stack([y_30, y_60, y_90]), all_features, stock_prices


def build_lstm_model(input_shape, layers=LSTMConfig.layers, neurons=LSTMConfig.neurons,
                     dropout_rate=LSTMConfig.dropout):
    """Creates and compiles the LSTM model."""
    model = Sequential()

    model.add(Input(shape=input_shape))

    for i in range(layers):
        if i == layers - 1:
            model.add(LSTM(neurons, return_sequences=False))
        else:
            model.add(LSTM(neurons, return_sequences=True))
        model.add(Dropout(dropout_rate))

    model.add(Dense(LSTMConfig.dense_units, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(3, activation='sigmoid'))

    optimizer = Adam(learning_rate=LSTMConfig.learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                  metrics=LSTMConfig.metrics)
    return model


class TrainingProgressCallback(Callback):
    """Sends training status updates via WebSocket."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        socketio.emit('progress', {
            'status': f'Trenujemy model (Epoka {epoch + 1}/{LSTMConfig.epochs})...',
            'accuracy': logs.get('accuracy', 0),
            'loss': logs.get('loss', 0)
        }, namespace='/')


def predict_price(model, all_features, X, results):
    """Predicts stock price movements using the trained model."""
    last_window = all_features[-LSTMConfig.time_step:].tolist()
    # Reshape into required form (1, time_step, features)
    input_data = np.array(last_window).reshape(1, LSTMConfig.time_step, X.shape[2])

    # Get one prediction for each time period
    optimizer = Adam(learning_rate=LSTMConfig.learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=LSTMConfig.metrics)
    predicted_prob = model.predict(input_data)

    # Store only the first prediction for each time period
    results["30_days"] = str(predicted_prob[0][0])
    results["60_days"] = str(predicted_prob[0][1])
    results["90_days"] = str(predicted_prob[0][2])

    return results


def evaluate_model(model, X_test, y_test, results, X_train, stock_prices):
    """Evaluates the model and generates plots."""
    metrics_values = model.evaluate(X_test, y_test, verbose=1)
    f1 = 2 * (metrics_values[2] * metrics_values[3]) / (metrics_values[2] + metrics_values[3])
    results.update({
        "loss": metrics_values[0],
        "accuracy": metrics_values[1],
        "precision": metrics_values[2],
        "recall": metrics_values[3],
        "auc": metrics_values[4],
        "f1": f1
    })

    # Confusion Matrix

    # Предсказания на тестовом наборе
    y_pred_prob = model.predict(X_test)

    # Преобразуем вероятности в классы (0 или 1) по порогу 0.5
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Вычисляем confusion matrix для каждого горизонта (30, 60, 90 дней)
    conf_matrix_30 = confusion_matrix(y_test[:, 0], y_pred[:, 0])
    conf_matrix_60 = confusion_matrix(y_test[:, 1], y_pred[:, 1])
    conf_matrix_90 = confusion_matrix(y_test[:, 2], y_pred[:, 2])

    # Записываем результаты в словарь
    results["confusion_matrix"] = {
        "30_days": conf_matrix_30.tolist(),
        "60_days": conf_matrix_60.tolist(),
        "90_days": conf_matrix_90.tolist()
    }

    # ROC AUC
    fpr, tpr, _ = roc_curve(y_test[:, 0], model.predict(X_test)[:, 0])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join('apps', 'static', 'images', 'roc_curve.png'))
    results["roc_auc_url"] = url_for('static', filename='images/roc_curve.png')

    # Train / test plot
    plt.figure(figsize=(10, 6))
    train_indices = np.arange(len(X_train))
    plt.plot(train_indices, stock_prices[train_indices], label="Training Data", color='green', alpha=0.6)
    test_indices = np.arange(len(X_train), len(X_train) + len(X_test))
    plt.plot(test_indices, stock_prices[test_indices], label="Testing Data", color='red', alpha=0.6)
    plt.legend()
    plt.title("Stock Prices with Training and Testing Data")
    plt.xlabel("Days")
    plt.ylabel("Stock Price")
    image_path = os.path.join('apps', 'static', 'images', 'train_test.png')
    plt.savefig(image_path)
    results["train_test_url"] = url_for('static', filename='images/train_test.png')

    return results
