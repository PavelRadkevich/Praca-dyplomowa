import itertools
import pandas as pd
import requests
import logging
import numpy as np

from apps.config import LSTMConfig
from apps.predict.predict import predict, predict_bp
from flask import Flask

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Flask server URL
FLASK_URL = "http://127.0.0.1:5000/predict"
company = "IBM"
start_year = 2000
end_year = 2025
parameters = ["ebitda", "net_income"]

# Experiment parameters (reduced set)
param_grid = {
    "epochs": [2, 5],              # Number of epochs
    "layers": [1, 2, 3],           # Number of layers
    "neurons": [50, 100, 200],     # Number of neurons per layer
    "test_size": [0.2, 0.4],       # Test data fraction
    "time_step": [30, 90, 150],    # Time steps for sequences
    "learning_rate": [0.001, 0.01],# Learning rate for optimizer
    "dropout": [0.2, 0.5]          # Dropout rate to prevent overfitting
}

# Number of runs to average results
num_runs = 10

request_data = {
    "company": company,
    "startYear": start_year,
    "endYear": end_year,
    "parameters": parameters
}

results_list = []
experiment_id = 0

logging.info("Rozpoczęcie eksperymentów...")

# Iterate over all parameter combinations from the reduced param_grid
for epochs, layers, neurons, test_size, time_step, learning_rate, dropout in itertools.product(
    param_grid["epochs"], param_grid["layers"], param_grid["neurons"],
    param_grid["test_size"], param_grid["time_step"], param_grid["learning_rate"],
    param_grid["dropout"]):

    logging.info(f"Uruchamianie eksperymentu: epochs={epochs}, layers={layers}, neurons={neurons}, "
                 f"test_size={test_size}, time_step={time_step}, learning_rate={learning_rate}, dropout={dropout}")

    # Override parameters in configuration
    LSTMConfig.epochs = epochs
    LSTMConfig.layers = layers
    LSTMConfig.neurons = neurons
    LSTMConfig.test_size = test_size
    LSTMConfig.time_step = time_step
    LSTMConfig.learning_rate = learning_rate
    LSTMConfig.dropout = dropout

    # Storage for averaging results
    aggregated_results = {
        "accuracy": [],
        "auc": [],
        "f1": [],
        "loss": [],
        "precision": [],
        "recall": [],
        "conf_matrix_30": [],
        "conf_matrix_60": [],
        "conf_matrix_90": []
    }

    for run in range(num_runs):
        logging.info(f"Uruchamianie {run + 1}/{num_runs} dla epochs={epochs}, layers={layers}")
        try:
            response = requests.post(FLASK_URL, json=request_data)
            if response.status_code == 200:
                result_data = response.json()
                logging.info(f"Otrzymano odpowiedź od serwera dla epochs={epochs}, layers={layers}")
            else:
                result_data = {"error": f"Błąd! Kod statusu: {response.status_code}"}
                logging.warning(result_data["error"])
        except Exception as e:
            result_data = {"error": str(e)}
            logging.error(f"Błąd podczas wysyłania żądania do serwera: {e}")

        # Append results for averaging
        aggregated_results["accuracy"].append(result_data.get("accuracy", 0))
        aggregated_results["auc"].append(result_data.get("auc", 0))
        aggregated_results["f1"].append(result_data.get("f1", 0))
        aggregated_results["loss"].append(result_data.get("loss", 0))
        aggregated_results["precision"].append(result_data.get("precision", 0))
        aggregated_results["recall"].append(result_data.get("recall", 0))

        # Process confusion matrix
        conf_matrix = result_data.get("confusion_matrix", {})
        aggregated_results["conf_matrix_30"].append(conf_matrix.get("30_days", [[0, 0], [0, 0]]))
        aggregated_results["conf_matrix_60"].append(conf_matrix.get("60_days", [[0, 0], [0, 0]]))
        aggregated_results["conf_matrix_90"].append(conf_matrix.get("90_days", [[0, 0], [0, 0]]))

    # Calculate average values
    averaged_results = {
        "accuracy": np.mean(aggregated_results["accuracy"]),
        "auc": np.mean(aggregated_results["auc"]),
        "f1": np.mean(aggregated_results["f1"]),
        "loss": np.mean(aggregated_results["loss"]),
        "precision": np.mean(aggregated_results["precision"]),
        "recall": np.mean(aggregated_results["recall"]),
    }

    # Average confusion matrices
    def average_conf_matrix(matrices):
        return np.mean(np.array(matrices), axis=0).astype(int).tolist()

    averaged_results["conf_matrix_30"] = average_conf_matrix(aggregated_results["conf_matrix_30"])
    averaged_results["conf_matrix_60"] = average_conf_matrix(aggregated_results["conf_matrix_60"])
    averaged_results["conf_matrix_90"] = average_conf_matrix(aggregated_results["conf_matrix_90"])

    # Add hyperparameters to experiment results
    averaged_results.update({
        "id": experiment_id,
        "company": company,
        "startYear": start_year,
        "endYear": end_year,
        "parameters": ", ".join(parameters),
        "epochs": epochs,
        "layers": layers,
        "neurons": neurons,
        "test_size": test_size,
        "time_step": time_step,
        "learning_rate": learning_rate,
        "dropout": dropout
    })

    results_list.append(averaged_results)
    logging.info(f"Zakończono eksperyment {experiment_id}. Uśrednione wyniki: {averaged_results}")
    experiment_id += 1

logging.info("Wszystkie eksperymenty zakończone. Zapisywanie wyników...")

# Convert results to DataFrame
df_results = pd.DataFrame(results_list)
print('Wyniki', df_results)

# Define column order
column_order = ["id", "company", "startYear", "endYear", "parameters", "epochs", "layers", "neurons",
                "test_size", "time_step", "learning_rate", "dropout"]

# Add any additional keys from JSON response except roc_auc_url and train_test_url
json_keys = [key for key in df_results.columns if key not in column_order and key not in ["roc_auc_url", "train_test_url"]]

# Arrange DataFrame columns
df_results = df_results[column_order + json_keys]

# Save to Excel
output_filename = "hyperparameter_experiments.xlsx"
df_results.to_excel(output_filename, index=False)
logging.info(f"Wyniki zapisane do pliku {output_filename}")
