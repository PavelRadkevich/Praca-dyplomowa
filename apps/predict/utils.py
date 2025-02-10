import numpy as np
import requests
import yfinance as yf
import pandas as pd


def get_quarterly_data_from_alpha_vantage(symbol, start_year, end_year, function_name, ALPHA_VANTAGE_KEY, key_name,
                                          trading_days):
    """
    Universal method for retrieving data from quarterly reports.
    :param symbol: Company symbol.
    :param start_year: Selected start year.
    :param end_year: Selected end year.
    :param function_name: Used to construct the API URL.
    :param key_name: Key name in the JSON response to extract the required data.
    :param trading_days: List of trading days.
    """
    url = f'https://www.alphavantage.co/query?function={function_name}&symbol={symbol}&apikey={ALPHA_VANTAGE_KEY}'
    response = requests.get(url)
    response_json = response.json()

    name_of_reports = 'quarterlyReports'

    # Check if 'quarterlyReports' key exists in the response
    if 'quarterlyReports' not in response_json:
        name_of_reports = 'quarterlyEarnings'
        if 'quarterlyEarnings' not in response_json:
            raise KeyError(f"Check if the company is available on Alpha Vantage. Not found: {key_name}, "
                           f"for symbol: {symbol}, in function: {function_name}")

    quarterly_reports = response_json[name_of_reports]
    parameter_quarterly_values = {}
    previous_value = 0

    for report in quarterly_reports:
        fiscal_year = int(report.get('fiscalDateEnding', '0000')[:4])
        if start_year <= fiscal_year <= end_year:
            parameter_quarterly_values[report.get('fiscalDateEnding')] = report.get(key_name)
        elif previous_value == 0 and fiscal_year < start_year:
            previous_value = report.get(key_name)

    parameter_daily_values = quarterly_to_daily(parameter_quarterly_values, previous_value, trading_days, False)
    return parameter_daily_values


def get_daily_data_from_alpha_vantage(symbol, start_year, end_year, trading_days, function_name, ALPHA_VANTAGE_KEY,
                                      key_name):
    """
    Retrieves daily data from Alpha Vantage.
    """
    url = (f'https://www.alphavantage.co/query?function={function_name}&symbol={symbol}'
           f'&outputsize=full&apikey={ALPHA_VANTAGE_KEY}')
    response = requests.get(url)
    time_series_daily = response.json()['Time Series (Daily)']
    time_series_df = pd.DataFrame.from_dict(time_series_daily, orient='index')
    time_series_df.index = pd.to_datetime(time_series_df.index).tz_localize(None)
    time_series_df = time_series_df.loc[time_series_df.index.isin(trading_days)]
    time_series_df = time_series_df[[key_name]]
    time_series_df = time_series_df.rename(columns={key_name: 'value'})
    time_series_df = time_series_df[::-1]  # Reverse order
    time_series_df = consider_splits(symbol, start_year, end_year, time_series_df, trading_days, False)
    time_series = np.array(time_series_df, dtype=float).reshape(-1, 1)
    return time_series


def quarterly_to_daily(quarterly_data, previous_value, trading_days, is_dividends):
    """
    Converts quarterly data to daily values by filling in missing dates.
    The last known quarterly value is carried forward until the next quarter.
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
        raise IOError("Mismatch between trading days and processed daily data.")


def adjust_quarterly_dates_to_trading_days(quarterly_data, trading_days):
    """
    Adjusts quarterly report dates to the nearest available trading days.
    """
    adjusted_data = {}
    trading_days = pd.Series(trading_days)

    for report_date, value in quarterly_data.items():
        report_date = pd.Timestamp(report_date)
        if report_date not in trading_days.values:
            closest_date = trading_days.iloc[(trading_days - report_date).abs().argsort()[0]]
        else:
            closest_date = report_date

        adjusted_data[closest_date] = value

    return adjusted_data


def adjust_split_to_trading_day(date, trading_days):
    """
    Adjusts stock split dates to the closest trading day.
    """
    if date not in trading_days:
        date = min(filter(lambda x: x > date, trading_days))
    return date


def consider_splits(symbol, start_year, end_year, data_df, trading_days, is_dividend):
    """
    Adjusts data for stock splits.
    """
    if data_df is None or data_df.empty:
        raise ValueError("data_df is None or empty")
    if 'value' not in data_df.columns:
        raise KeyError("Column 'value' is missing in data_df")

    ticker = yf.Ticker(symbol)
    splits = ticker.splits
    splits.index = pd.to_datetime(splits.index).tz_localize(None)
    start_date = pd.Timestamp(f"{start_year}-01-01")
    end_date = pd.Timestamp(f"{end_year}-01-01")
    filtered_splits = splits[(splits.index >= start_date) & (splits.index <= end_date)]

    data_df['value'] = pd.to_numeric(data_df['value'], errors='coerce')

    for split_date, split_ratio in filtered_splits.items():
        split_date = adjust_split_to_trading_day(split_date, trading_days)
        data_df.loc[split_date:, 'value'] /= split_ratio

    return data_df


def restore_splits(symbol, start_year, end_year, data_df, trading_days):
    """
    Restores stock splits by multiplying values back to their original form.
    """
    ticker = yf.Ticker(symbol)
    splits = ticker.splits
    splits.index = pd.to_datetime(splits.index).tz_localize(None)
    start_date = pd.Timestamp(f"{start_year}-01-01")
    end_date = pd.Timestamp(f"{end_year}-01-01")
    filtered_splits = splits[(splits.index >= start_date) & (splits.index <= end_date)]

    data_df = data_df.copy()
    data_df['value'] = pd.to_numeric(data_df['value'], errors='coerce')

    for split_date, split_ratio in filtered_splits.items():
        split_date = adjust_split_to_trading_day(split_date, trading_days)
        data_df.loc[split_date:, 'value'] *= split_ratio

    return data_df
