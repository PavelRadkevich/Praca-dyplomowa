import logging

import numpy as np
from flask import Blueprint, jsonify, request

import inspect

from flask_socketio import emit

import apps.predict.api_requests

from keras.models import Sequential
from keras.layers import LSTM, Dense, Input

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

        #stock_prices, trading_days = all_methods["get_stock_prices"](symbol, start_year, end_year)
        #dividends = all_methods["get_dividends"](symbol, start_year, end_year, trading_days)
        emit('progres', {'status': 'Ładujemy ceny akcji i dywidendy...'})
        stock_prices, trading_days, dividends = all_methods["get_stock_data_with_dividends"](symbol, start_year, end_year)
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
    stock_prices = np.array(stock_prices, dtype=float).reshape(-1, 1)
    dividends = np.array(dividends, dtype=float).reshape(-1, 1)
    # Сохраняем все параметры в одном массиве
    additional_features = np.column_stack(
        [np.array(results[param], dtype=float).reshape(-1, 1) for param in parameters]
    )
    # Объединяем цены акций с другими параметрами
    all_features = np.hstack((stock_prices, dividends, additional_features))

    # Создаём временные окна (time steps) для LSTM
    time_step = 60  # Длина временного окна

    X, y_30, y_60, y_90 = [], [], [], []
    for i in range(len(all_features) - time_step - 90):
        X.append(all_features[i:i + time_step, :])
        y_30.append(int(stock_prices[i + time_step + 30] > stock_prices[i + time_step]))
        y_60.append(int(stock_prices[i + time_step + 60] > stock_prices[i + time_step]))
        y_90.append(int(stock_prices[i + time_step + 90] > stock_prices[i + time_step]))

    X, y_30, y_60, y_90 = np.array(X), np.array(y_30), np.array(y_60), np.array(y_90)
    y = np.column_stack([y_30, y_60, y_90])
    model = Sequential([
        Input(shape=(time_step, X.shape[2])),
        LSTM(50, return_sequences=True),
        LSTM(50, return_sequences=False),
        Dense(25, activation='relu'),
        Dense(3, activation='sigmoid')  # Три выхода: 30, 60, 90 дней
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, batch_size=32, epochs=1, validation_split=0.2)

    last_window = all_features[-time_step:].tolist()

    # Для одного предсказания (не сдвигая окно)
    future_predictions = {"30_days": [], "60_days": [], "90_days": []}

    # Преобразуем в нужную форму (1, time_step, features)
    input_data = np.array(last_window).reshape(1, time_step, X.shape[2])

    # Получаем одно предсказание модели для каждого периода
    predicted_prob = model.predict(input_data)[0]

    # Сохраняем только первое предсказание для каждого периода времени
    future_predictions["30_days"].append(predicted_prob[0].tolist())
    future_predictions["60_days"].append(predicted_prob[1].tolist())
    future_predictions["90_days"].append(predicted_prob[2].tolist())

    # Добавляем будущие предсказания в результаты
    results["future_predictions"] = future_predictions

    print(future_predictions)

    return jsonify(future_predictions), 200
