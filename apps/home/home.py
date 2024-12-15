import os
import requests
from flask import Blueprint, render_template
from bs4 import BeautifulSoup

home_bp = Blueprint('home', __name__, template_folder='../../templates')

url = "https://www.investing.com/dividends-calendar/"
headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, 'html.parser')

dividends_table = soup.find('table', {'id': 'dividendsCalendarData'})

# TODO: Фильтрация по размеру дивиденд (проценты и volume?) чтобы убрать слишком маленькие фирмы. Найти правильный сайт
if dividends_table:
    body = dividends_table.find('tbody')
    rows = body.find_all('tr')

    for row in rows[:5]:
        cols = row.find_all('td')
        if len(cols) > 2:  # Убедимся, что строка содержит нужное количество столбцов
            company = cols[0].text.strip()  # Компания
            date = cols[1].text.strip()     # Дата выплаты
            dividend = cols[2].text.strip() # Сумма дивиденда

            print(f"Компания: {company}, Дата выплаты: {date}, Дивиденд: {dividend}")
else:
    print("Nie znaleziono")

@home_bp.route('/')
def home():
    return render_template('home.html', title="Praca Dyplomowa")
