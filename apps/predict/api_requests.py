import os

import requests

ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")


def get_ebitda(symbol, start_year, end_year):
    url = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={ALPHA_VANTAGE_KEY}'
    response = requests.get(url)
    quarterly_reports = response.json()['quarterlyReports']

    ebitda_values = {}

    for report in quarterly_reports:
        fiscal_year = int(report.get('fiscalDateEnding', '0000')[:4])
        if fiscal_year >= start_year and fiscal_year <=end_year:

