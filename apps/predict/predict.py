from datetime import datetime

from flask import Blueprint, render_template, jsonify, request

predict_bp = Blueprint('predict', __name__)


@predict_bp.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        company = data['company']
        start_year = data['startYear']
        end_year = datetime.now().year
        parameters = data['parameters']

        return jsonify({"Hello": 'hello'}), 200
    except Exception as e:
        print(f"Error processing data: {e}")
        return jsonify({"error": "Failed to process data"}), 500
