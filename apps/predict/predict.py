from datetime import datetime

import numpy as np
from flask import Blueprint, jsonify, request

import inspect
import apps.predict.api_requests

from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler

# Get all methods from api_requests
all_methods = {
    name: func
    for name, func in inspect.getmembers(apps.predict.api_requests, inspect.isfunction)
}

predict_bp = Blueprint('predict', __name__)


@predict_bp.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symbol = data['company']
        start_year = data['startYear']
        end_year = data['endYear']
        parameters = data['parameters']

        stock_prices, trading_days = all_methods["get_stock_prices"](symbol, start_year, end_year)

        results = {}
        for param in parameters:
            method_name = f'get_{param}'
            if method_name in all_methods:
                try:
                    results[param] = all_methods[method_name](symbol, int(start_year), int(end_year), trading_days)
                    if type(results[param]) is not list:
                        results[param] = results[param].tolist()
                except KeyError as ke:
                    return jsonify({"error": str(ke)}), 400
                except Exception as e:
                    return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
            else:
                return jsonify({"error": f"Method {method_name} not found"}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to process request: {str(e)} (check your API limit)"}), 500

    for key, value in results.items():
        if len(stock_prices) != len(value):
            return jsonify({"error": "Parameter lengths do not match"}), 500



    # Преобразуем данные для LSTM
    stock_prices = np.array(stock_prices).reshape(-1, 1)
    print("stock_prices reshape:", stock_prices)
    # Сохраняем все параметры в одном массиве
    additional_features = np.column_stack(
        [np.array(results[param]).reshape(-1, 1) for param in parameters]
    )
    # Объединяем цены акций с другими параметрами
    all_features = np.hstack((stock_prices, additional_features))
    print("all_featurs: ", all_features)
    # Масштабируем все данные
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(all_features)
    print("scaled: ", scaled_features)
    # Создаём временные окна (time steps) для LSTM
    time_step = 60  # Длина временного окна
    X, y = [], []
    for i in range(len(scaled_features) - time_step - 1):
        X.append(scaled_features[i:(i + time_step), :])  # Все признаки
        y.append(scaled_features[i + time_step, 0])  # Цены акций
    X, y = np.array(X), np.array(y)
    # Изменяем форму для LSTM
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    print("X: ", X)
    print("Y: ", y)
    # Создаём LSTM-модель
    model = Sequential([
        Input(shape=(time_step, X.shape[2])),  # Учитываем все признаки
        LSTM(50, return_sequences=True),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Обучение модели
    model.fit(X, y, batch_size=1, epochs=1, verbose=0)
    # Предсказания на основе модели
    predictions = model.predict(X)
    # Обратное преобразование предсказаний в исходный масштаб
    predictions = scaler.inverse_transform(
        np.hstack((predictions, np.zeros((predictions.shape[0], all_features.shape[1] - 1))))
    )[:, 0].flatten().tolist()  # Берём только цены акций
    print("Predictions: ", predictions)
    # Добавляем результат предсказаний в response
    results['predicted_prices'] = predictions




    return jsonify(results), 200
