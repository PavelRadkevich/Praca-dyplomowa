from datetime import date, timedelta, datetime

import requests


# Universal method for retrieving data from quarterly reports. function_name: used when building a url,
# key_name when retrieving from already retrieved data (key name in JSON format).
def get_quarterly_data_from_alpha_vantage(symbol, start_year, function_name, ALPHA_VANTAGE_KEY, key_name):
    url = f'https://www.alphavantage.co/query?function={function_name}&symbol={symbol}&apikey={ALPHA_VANTAGE_KEY}'
    response = requests.get(url)
    quarterly_reports = response.json()['quarterlyReports']

    parameter_quarterly_values = {}
    previous_value = 0

    for report in quarterly_reports:
        fiscal_year = int(report.get('fiscalDateEnding', '0000')[:4])
        if start_year <= fiscal_year:
            parameter_quarterly_values[report.get('fiscalDateEnding')] = report.get(key_name)
        elif previous_value == 0:
            previous_value = report.get(key_name)

    parameter_daily_values = quarterly_to_daily(parameter_quarterly_values, start_year, previous_value)

    if check_length_of_data(parameter_daily_values, start_year):
        return parameter_daily_values
    else:
        return []


# The first value is retrieved in advance from the api, the next ones are checked one by one by date,
# if the first known date is 2025-10-31 and the second is 2025-12-31, then the list between these dates
# will be filled with the values from 2025-10-31
# quarterly_date should be: {YYYY-MM-DD: value, ...}
def quarterly_to_daily(quarterly_data, start_year, previous_value):
    total_days, start_date = get_the_number_of_days(start_year)

    daily_data = [None] * total_days

    # Convert dates from strings to date objects
    quarter_dates = {date.fromisoformat(key): value for key, value in quarterly_data.items()}

    current_value = previous_value
    for i in range(total_days):
        current_date = start_date + timedelta(days=i)
        if current_date in quarter_dates:
            current_value = quarter_dates[current_date]
        daily_data[i] = current_value

    return daily_data


def get_the_number_of_days(start_year):
    start_date = date(start_year, 1, 1)
    end_date = datetime.now().date()
    total_days = (end_date - start_date).days + 1
    return total_days, start_date


def check_length_of_data(data, start_year):
    total_days = get_the_number_of_days(start_year)
    if len(data) == total_days:
        return True
    else:
        return False
