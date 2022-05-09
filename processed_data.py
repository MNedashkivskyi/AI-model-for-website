import json
import itertools
import pandas as pd
import numpy as np
import datetime


def prepare_data():
    users_data = pd.read_json("data/users.jsonl", lines=True)
    deliveries_data = pd.read_json("data/deliveries.jsonl", lines=True)
    events_data = pd.read_json("data/sessions.jsonl", lines=True)
    products_data = pd.read_json("data/products.jsonl", lines=True)

    max_timestamp = events_data.timestamp.max()

    events_data['year'] = events_data.timestamp.dt.year
    events_data['month'] = events_data.timestamp.dt.month

    years = list(events_data.year.unique())
    months = list(events_data.month.unique())
    user_ids = list(events_data.user_id.unique())

    triplets = []
    for triplet in itertools.product(years, months, user_ids):
        triplets.append(triplet)

    processed_data = pd.DataFrame(triplets, columns=['year', 'month', 'user_id'])
    processed_data['timestamp'] = pd.to_datetime(events_data.timestamp)
    processed_data = processed_data[
        (processed_data['timestamp'] > '2022-03-01') & (processed_data['timestamp'] < '2022-05-01')]
    processed_data['month'] = processed_data['timestamp'].dt.month
    processed_data['year'] = processed_data['timestamp'].dt.year

    processed_data['month_sin'] = np.sin(events_data['month'] * (2. * np.pi / 12))
    processed_data['month_cos'] = np.cos(events_data['month'] * (2. * np.pi / 12))

    all_events = events_data.groupby(['user_id', 'year']).aggregate({"session_id": "count"}) \
        .rename(columns={"session_id": "all_sessions"}) \
        .reset_index()

    buying_events = events_data[events_data['event_type'] == 'BUY_PRODUCT'] \
        .groupby(['user_id', 'year']) \
        .aggregate({"session_id": "count"}) \
        .rename(columns={"session_id": "buying_sessions"}) \
        .reset_index()

    buying_events_first_month = events_data[events_data['event_type'] == 'BUY_PRODUCT'].groupby(
        ['user_id', 'month', 'year']).aggregate({'session_id': 'count'}).rename(
        columns={'session_id': 'buying_events_first_month'}).reset_index()
    buying_events_first_month = buying_events_first_month[
        (buying_events_first_month['month'] == 3) & (buying_events_first_month['year'] == 2022)]

    buying_events_second_month = events_data[events_data['event_type'] == 'BUY_PRODUCT'].groupby(
        ['user_id', 'month', 'year']).aggregate({'session_id': 'count'}).rename(
        columns={'session_id': 'buying_events_second_month'}).reset_index()
    buying_events_second_month = buying_events_second_month[
        (buying_events_second_month['month'] == 4) & (
                buying_events_second_month['year'] == 2022)]

    events_ratio = pd.merge(all_events, buying_events, how='left', on=['year', 'user_id'])
    events_ratio['buying_sessions'].fillna(0, inplace=True)

    processed_data = pd.merge(processed_data, events_ratio.drop('all_sessions', axis=1), how='left',
                              on=['year', 'user_id'])
    processed_data['buying_sessions'].fillna(0, inplace=True)

    buying_sessions = events_data[events_data['event_type'] == 'BUY_PRODUCT']
    deals = pd.merge(buying_sessions, products_data, how='left', on=['product_id'])
    deals['final_price'] = deals['price']
    deals = deals[(deals['timestamp'] > '2022-03-01') & (deals['timestamp'] < '2022-05-01')]
    pd.set_option("display.max_rows", None, "display.max_columns", None)

    spent_money_first_month = deals[deals.timestamp.dt.month == 3].groupby('user_id').aggregate(
        {'final_price': 'sum'}).rename(columns={'final_price': 'first_month'}).reset_index()
    spent_money_second_month = deals[deals.timestamp.dt.month == 4].groupby('user_id').aggregate(
        {'final_price': 'sum'}).rename(
        columns={'final_price': 'second_month'}).reset_index()

    monthly_deals = deals.groupby(['year', 'user_id']).aggregate({'final_price': 'sum'}).rename(
        columns={'final_price': 'spent_money'}).reset_index()

    processed_data = pd.merge(processed_data, monthly_deals, how='left', on=['year', 'user_id'])
    processed_data = pd.merge(processed_data, spent_money_first_month, how='left', on=['user_id'])
    processed_data = pd.merge(processed_data, spent_money_second_month, how='left', on=['user_id'])
    processed_data = pd.merge(processed_data, buying_events_first_month, how='left', on=['year', 'month', 'user_id'])
    processed_data = pd.merge(processed_data, buying_events_second_month, how='left', on=['year', 'month', 'user_id'])
    processed_data['spent_money'].fillna(0, inplace=True)
    processed_data['first_month'].fillna(0, inplace=True)
    processed_data['second_month'].fillna(0, inplace=True)
    processed_data['buying_events_first_month'].fillna(0, inplace=True)
    processed_data['buying_events_second_month'].fillna(0, inplace=True)
    processed_data = processed_data.sort_values(by=['user_id']).reset_index()

    processed_data.to_csv('processed_data.csv')


prepare_data()
