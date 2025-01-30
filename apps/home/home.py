import os
import requests
from flask import Blueprint, render_template, jsonify
from bs4 import BeautifulSoup

from apps.predict import api_requests

home_bp = Blueprint('home', __name__, template_folder='templates')
FINNHUB_KEY = os.environ.get("FINNHUB_API_KEY")
ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")


@home_bp.route('/')
def home():
    #stock_prices, trading_days = api_requests.get_stock_prices("IBM", 2000)
    #api_requests.get_dividends("IBM", 2000, trading_days)
    return render_template('home.html', title="Praca Dyplomowa",
                           dividendsCalendarCompanies=get_nearest_companies(),
                           allCompanies=get_all_companies())


@home_bp.route('/api/get_company_years/<symbol>', methods=['GET'])
def get_company_years(symbol):
    try:
        url = f'https://www.alphavantage.co/query'
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'outputsize': 'full',
            'apikey': ALPHA_VANTAGE_KEY
        }
        response = requests.get(url, params=params)
        data = response.json()

        if 'Time Series (Daily)' not in data:
            return jsonify({'error': 'No data available for this company'})

        dates = list(data['Time Series (Daily)'].keys())
        years = sorted(set(date.split('-')[0] for date in dates))

        return jsonify({'years': years})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def get_nearest_companies():
    url = "https://www.investing.com/dividends-calendar/"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    dividends_table = soup.find('table', {'id': 'dividendsCalendarData'})
    companies = []

    # TODO: Фильтрация по размеру дивиденд (проценты и volume?) чтобы убрать слишком маленькие фирмы. Найти правильный сайт
    if dividends_table:
        body = dividends_table.find('tbody')
        rows = body.find_all('tr')

        # We take 5 first companies on the website to show in our calendar
        for row in rows[:5]:
            cols = row.find_all('td')
            if len(cols) > 2:  # Check for correct number of columns (to avoid e.g. header)
                company = cols[1].text.strip()
                date = cols[2].text.strip()
                dividend = cols[6].text.strip()
                companies.append({'company': company, 'date': date, 'dividend': dividend})
    else:
        print("Nie znaleziono")
    return companies


def get_all_companies():
    url = f'https://finnhub.io/api/v1/stock/symbol?exchange=US&token={FINNHUB_KEY}'  # Get all companies from US stock
    # markets (we can change country ..exchange=US..)
    response = requests.get(url)
    if response.status_code == 200:
        companies = response.json()
        result = []
        for company in companies:
            if company['symbol'] != "" and company['description'] != "":
                result.append({'symbol': company['symbol'], 'name': company['description']})
        result.sort(key=lambda x: x['name'].lower())
        return result
    else:
        return jsonify({"error": "Failed to fetch companies"}), 500
