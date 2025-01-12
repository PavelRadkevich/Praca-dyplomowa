import os

import requests
import apps.predict.utils as utils

ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")
FINNHUB_KEY = os.environ.get("FINNHUB_API_KEY")


def get_ebitda(symbol, start_year):
    ebitda_values = utils.get_quarterly_data_from_alpha_vantage(symbol, start_year,
                                                                'INCOME_STATEMENT',
                                                                ALPHA_VANTAGE_KEY, 'ebitda')

    return ebitda_values


def get_debt(symbol, start_year):
    debt_values = utils.get_quarterly_data_from_alpha_vantage(symbol, start_year,
                                                              'BALANCE_SHEET',
                                                              ALPHA_VANTAGE_KEY, 'totalLiabilities')
    return debt_values


def get_revenue(symbol, start_year):
    revenue_values = utils.get_quarterly_data_from_alpha_vantage(symbol, start_year,
                                                                 'INCOME_STATEMENT',
                                                                 ALPHA_VANTAGE_KEY, 'totalRevenue')


def get_income(symbol, start_year):
    income_values = utils.get_quarterly_data_from_alpha_vantage(symbol, start_year,
                                                                'INCOME_STATEMENT',
                                                                ALPHA_VANTAGE_KEY, 'netIncome')
