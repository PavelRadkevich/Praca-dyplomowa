import os
import requests
from flask import Blueprint, render_template, jsonify
from bs4 import BeautifulSoup
from dotenv import load_dotenv


home_bp = Blueprint('home', __name__, template_folder='../../templates')


@home_bp.route('/')
def home():
    return render_template('home.html', title="Praca Dyplomowa",
                           dividendsCalendarCompanies=getNearestCompanies(),
                           allCompanies=getAllCompanies())


def getNearestCompanies():
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

def getAllCompanies():
    FINNHUB_KEY = os.environ.get("FINNHUB_API_KEY")
    print(FINNHUB_KEY)
    url =f'https://finnhub.io/api/v1/stock/symbol?exchange=US&token={FINNHUB_KEY}' # Get all companies from US stock
    # markets (we can change country ..exchange=US..)
    response = requests.get(url)
    print(response.json)
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