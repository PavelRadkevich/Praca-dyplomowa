import os

import numpy as np
import pandas as pd
import requests

import apps.predict.utils as utils

ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")
FINNHUB_KEY = os.environ.get("FINNHUB_API_KEY")


def get_stock_data_with_dividends(symbol, start_year, end_year):
    stock_df, trading_days = get_stock_prices(symbol, start_year, end_year)
    dividends, last_dividend_date = get_dividends(symbol, start_year, end_year, trading_days)
    last_dividend_date = pd.Timestamp(last_dividend_date)

    last_nonzero_idx = max(i for i, val in enumerate(dividends) if val != 0)

    # Trim the list up to this point
    dividends = dividends[: last_nonzero_idx + 1]
    trading_days = trading_days[: last_nonzero_idx + 1]
    stock_df = stock_df.iloc[: last_nonzero_idx + 1]

    return stock_df, trading_days, dividends, last_dividend_date


# Return stock prices without splits
def get_stock_prices(symbol, start_year, end_year):
    url = (f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}'
           f'&outputsize=full&apikey={ALPHA_VANTAGE_KEY}')
    response = requests.get(url)
    time_series_daily = response.json()['Time Series (Daily)']
    time_series_df = pd.DataFrame.from_dict(time_series_daily, orient='index')
    time_series_df.index = pd.to_datetime(time_series_df.index).tz_localize(None)
    time_series_df = time_series_df[time_series_df.index.year >= int(start_year)]
    time_series_df = time_series_df[time_series_df.index.year <= int(end_year)]
    time_series_df = time_series_df[['4. close']]
    time_series_df = time_series_df.rename(columns={'4. close': 'value'})

    time_series_df = time_series_df[::-1]  # Reverse

    trading_days = time_series_df.index.tolist()
    time_series_df = utils.consider_splits(symbol, start_year, end_year, time_series_df, trading_days, False)
    return time_series_df, trading_days


def get_dividends(symbol, start_year, end_year, trading_days):
    url = (f'https://www.alphavantage.co/query?function=DIVIDENDS&symbol={symbol}'
           f'&apikey={ALPHA_VANTAGE_KEY}')
    response = requests.get(url).json()

    # Convert to DataFrame where index: ex_dividend_date, value: amount
    data = [{"date": item["ex_dividend_date"], "amount": float(item["amount"])} for item in response["data"]]
    df = pd.DataFrame(data)

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df.rename(columns={'amount': 'value'})

    df_before_start_year = df[df.index.year < int(start_year)]
    last_date_before_start = df_before_start_year.index.max()
    previous_value = 0

    df = df[df.index.year >= int(start_year)]
    df = df[df.index.year <= int(end_year)]

    dividends_splits = utils.consider_splits(symbol, start_year, end_year, df, trading_days, True)
    dividends_splits_dict = {str(index): float(row['value']) for index, row in dividends_splits.iterrows()}
    last_dividend_date = max(dividends_splits_dict.keys())
    dividends_daily = utils.quarterly_to_daily(dividends_splits_dict, previous_value, trading_days, True)
    return dividends_daily, last_dividend_date

def get_volume(symbol, start_year, end_year, trading_days):
    total_volume = utils.get_daily_data_from_alpha_vantage(symbol, start_year, end_year, trading_days,
                                                           'TIME_SERIES_DAILY',
                                                           ALPHA_VANTAGE_KEY, "5. volume")
    return total_volume

def get_open_price(symbol, start_year, end_year, trading_days):
    open_price = utils.get_daily_data_from_alpha_vantage(symbol, start_year, end_year, trading_days,
                                                          'TIME_SERIES_DAILY',
                                                          ALPHA_VANTAGE_KEY, "1. open")
    return open_price

def get_low_price(symbol, start_year, end_year, trading_days):
    low_price = utils.get_daily_data_from_alpha_vantage(symbol, start_year, end_year, trading_days,
                                                         'TIME_SERIES_DAILY',
                                                         ALPHA_VANTAGE_KEY, "3. low")
    return low_price

def get_high_price(symbol, start_year, end_year, trading_days):
    high_price = utils.get_daily_data_from_alpha_vantage(symbol, start_year, end_year, trading_days,
                                                          'TIME_SERIES_DAILY',
                                                          ALPHA_VANTAGE_KEY, "2. high")
    return high_price

def get_ebitda(symbol, start_year, end_year, trading_days):
    ebitda_values = utils.get_quarterly_data_from_alpha_vantage(symbol, start_year, end_year,
                                                                'INCOME_STATEMENT',
                                                                ALPHA_VANTAGE_KEY, 'ebitda',
                                                                trading_days)
    return ebitda_values


def get_total_debt(symbol, start_year, end_year, trading_days):
    debt_values = utils.get_quarterly_data_from_alpha_vantage(symbol, start_year, end_year,
                                                              'BALANCE_SHEET',
                                                              ALPHA_VANTAGE_KEY, 'totalLiabilities',
                                                              trading_days)
    return debt_values


def get_total_revenue(symbol, start_year, end_year, trading_days):
    revenue_values = utils.get_quarterly_data_from_alpha_vantage(symbol, start_year, end_year,
                                                                 'INCOME_STATEMENT',
                                                                 ALPHA_VANTAGE_KEY, 'totalRevenue',
                                                                 trading_days)
    return revenue_values


def get_net_income(symbol, start_year, end_year, trading_days):
    income_values = utils.get_quarterly_data_from_alpha_vantage(symbol, start_year, end_year,
                                                                'INCOME_STATEMENT',
                                                                ALPHA_VANTAGE_KEY, 'netIncome',
                                                                trading_days)
    return income_values


def get_gross_profit(symbol, start_year, end_year, trading_days):
    gross_profit_values = utils.get_quarterly_data_from_alpha_vantage(symbol, start_year, end_year,
                                                                      'INCOME_STATEMENT',
                                                                      ALPHA_VANTAGE_KEY, 'grossProfit',
                                                                      trading_days)
    return gross_profit_values


def get_total_assets(symbol, start_year, end_year, trading_days):
    total_assets_values = utils.get_quarterly_data_from_alpha_vantage(symbol, start_year, end_year,
                                                                      'BALANCE_SHEET',
                                                                      ALPHA_VANTAGE_KEY, 'totalAssets',
                                                                      trading_days)
    return total_assets_values


def get_operating_income(symbol, start_year, end_year, trading_days):
    operating_income_values = utils.get_quarterly_data_from_alpha_vantage(symbol, start_year, end_year,
                                                                          'INCOME_STATEMENT',
                                                                          ALPHA_VANTAGE_KEY, 'operatingIncome',
                                                                          trading_days)
    return operating_income_values


def get_earnings_per_share(symbol, start_year, end_year, trading_days):
    earnings_per_share = utils.get_quarterly_data_from_alpha_vantage(symbol, start_year, end_year,
                                                                     'EARNINGS',
                                                                     ALPHA_VANTAGE_KEY, 'reportedEPS',
                                                                     trading_days)
    return earnings_per_share


def get_current_ratio(symbol, start_year, end_year, trading_days):
    total_assets = np.array(get_total_assets(symbol, start_year, end_year, trading_days), dtype=float)
    total_debt = np.array(get_total_debt(symbol, start_year, end_year, trading_days), dtype=float)

    total_current_ratio = np.divide(total_assets, total_debt, out=np.zeros_like(total_assets), where=total_debt != 0)

    return total_current_ratio


def get_daily_volatility(symbol, start_year, end_year, trading_days):
    total_high = np.array(utils.get_daily_data_from_alpha_vantage(symbol, start_year, end_year, trading_days,
                                                                  'TIME_SERIES_DAILY',
                                                                  ALPHA_VANTAGE_KEY, "2. high"), dtype=float)
    total_low = np.array(utils.get_daily_data_from_alpha_vantage(symbol, start_year, end_year, trading_days,
                                                                 'TIME_SERIES_DAILY',
                                                                 ALPHA_VANTAGE_KEY, "3. low"), dtype=float)
    total_open = np.array(utils.get_daily_data_from_alpha_vantage(symbol, start_year, end_year, trading_days,
                                                                  'TIME_SERIES_DAILY',
                                                                  ALPHA_VANTAGE_KEY, "2. open"), dtype=float)
    total_high_low = total_high - total_low
    total_daily_volatility = np.divide(total_high_low, total_open, out=np.zeros_like(total_high_low),
                                       where=total_open != 0)

    return total_daily_volatility
