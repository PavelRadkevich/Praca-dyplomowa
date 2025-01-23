import requests
import yfinance as yf
import pandas as pd


def get_quarterly_data_from_alpha_vantage(symbol, start_year, function_name, ALPHA_VANTAGE_KEY, key_name, trading_days):
    """
    Universal method for retrieving data from quarterly reports.
    :param symbol: company symbol
    :param start_year: year which user chose
    :param function_name: used when building an url,
    :param key_name: when retrieving from already retrieved data (key name in JSON format).
    """
    url = f'https://www.alphavantage.co/query?function={function_name}&symbol={symbol}&apikey={ALPHA_VANTAGE_KEY}'
    response = requests.get(url)
    response_json = response.json()

    name_of_reports = 'quarterlyReports'

    # Check if the 'quarterlyReports' key is in the response.
    if 'quarterlyReports' not in response_json:
        name_of_reports = 'quarterlyEarnings'
        if 'quarterlyEarnings' not in response_json:
            raise KeyError(f" Sprawdź czy firma jest dostępna w alpha vantage. Nie znaleziono: {key_name}, "
                           f"dla symbolu: {symbol}, w funkcji: {function_name}")

    quarterly_reports = response_json[name_of_reports]
    parameter_quarterly_values = {}
    previous_value = 0

    for report in quarterly_reports:
        fiscal_year = int(report.get('fiscalDateEnding', '0000')[:4])
        if start_year <= fiscal_year:
            parameter_quarterly_values[report.get('fiscalDateEnding')] = report.get(key_name)
        elif previous_value == 0:
            previous_value = report.get(key_name)

    parameter_daily_values = quarterly_to_daily(parameter_quarterly_values, previous_value, trading_days, False)

    return parameter_daily_values


def quarterly_to_daily(quarterly_data, previous_value, trading_days, is_dividends):
    """
    The first value is retrieved in advance from the api, the next ones are checked one by one by date,
    if the first known date is 2025-10-31 and the second is 2025-12-31, then the list between these dates
    will be filled with the values from 2025-10-31
    quarterly_date should be: {YYYY-MM-DD: value, ...}
    """
    daily_data = [None] * len(trading_days)
    quarterly_data = adjust_quarterly_dates_to_trading_days(quarterly_data, trading_days)

    quarter_dates = {key: value for key, value in quarterly_data.items()}

    current_value = 0 if is_dividends else previous_value

    for i, current_date in enumerate(trading_days):
        if current_date in quarter_dates:
            current_value = quarter_dates[current_date]
        if is_dividends:
            daily_data[i] = current_value
            current_value = 0
        else:
            daily_data[i] = current_value

    if len(trading_days) == len(daily_data):
        return daily_data
    else:
        raise IOError


def adjust_quarterly_dates_to_trading_days(quarterly_data, trading_days):
    """
    Binds the dates of quarterly reports to the nearest trading days.
    """
    adjusted_data = {}
    trading_days = pd.Series(trading_days)

    for report_date, value in quarterly_data.items():
        report_date = pd.Timestamp(report_date)
        if report_date not in trading_days.values:
            closest_date = trading_days.iloc[(trading_days - report_date).abs().argsort()[0]]
            """"
            We subtract the report date from the array of trading days (we get an array of deltas), 
            remove the sign (because we are interested in the distance to the report date), 
            sort in ascending order and take the first day
            """
        else:
            closest_date = report_date

        adjusted_data[closest_date] = value

    return adjusted_data


def adjust_split_to_trading_day(date, trading_days):
    if date not in trading_days:
        date = min(filter(lambda x: x > date, trading_days))
    return date


def consider_splits(symbol, start_year, data_df, trading_days, isDividend):
    # Take all the splits, and leave only the ones greater than start_year.
    ticker = yf.Ticker(symbol)
    splits = ticker.splits
    splits.index = pd.to_datetime(splits.index).tz_localize(None)
    start_date = pd.Timestamp(f"{start_year}-01-01")
    filtered_splits = splits[splits.index >= start_date]

    data_df = data_df.copy()
    data_df['value'] = pd.to_numeric(data_df['value'], errors='coerce')

    if isDividend:
        for split_date, split_ratio in filtered_splits.items():
            next_dates_in_df = data_df.index[data_df.index >= split_date]
            if not next_dates_in_df.empty:
                first_date = next_dates_in_df[-1]
                data_df.loc[first_date:, 'value'] /= split_ratio
            else:
                continue
    else:
        for split_date, split_ratio in filtered_splits.items():
            adjust_split_to_trading_day(split_date, trading_days)
            # If split happens after start_date, we correct all next values
            if split_date in data_df.index:
                data_df.loc[split_date:, 'value'] /= split_ratio

    return data_df


def restore_splits(symbol, start_year, data_df):
    ticker = yf.Ticker(symbol)
    splits = ticker.splits
    splits.index = pd.to_datetime(splits.index).tz_localize(None)
    start_date = pd.Timestamp(f"{start_year}-01-01")
    filtered_splits = splits[splits.index >= start_date]

    data_df = data_df.copy()
    data_df['value'] = pd.to_numeric(data_df['value'], errors='coerce')

    for split_date, split_ratio in filtered_splits.items():
        # If split happens after start_date, we correct all next values
        if split_date in data_df.index:
            data_df.loc[split_date:, 'value'] *= split_ratio

    return data_df
