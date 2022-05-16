from pandas.core.arrays import datetimes
import json
import itertools
import pandas as pd
import numpy as np
import datetime


def prepare_data(data_begin, data_end):
    events_data = pd.read_json("/content/drive/My Drive/IUM/sessions.jsonl", lines=True)
    products_data = pd.read_json("/content/drive/My Drive/IUM/products.jsonl", lines=True)

    events_data['year'] = events_data.timestamp.dt.year
    events_data['month'] = events_data.timestamp.dt.month

    years = list(events_data.year.unique())
    months = list(events_data.month.unique())
    user_ids = list(events_data.user_id.unique())

    triplets = []
    for triplet in itertools.product(years, months, user_ids):
        triplets.append(triplet)

    dates=pd.date_range(data_begin,data_end, 
              freq='MS').tolist()

  
    processed_data = pd.DataFrame(triplets, columns=['year', 'month', 'user_id'])
 

    buying_events_month = events_data[events_data['event_type'] == 'BUY_PRODUCT'].groupby(
        ['user_id', 'month', 'year']).aggregate({'session_id': 'count'}).rename(
        columns={'session_id': 'buying_events_month'}).reset_index()


    all_events_month = events_data.groupby(
    ['user_id', 'month', 'year']).aggregate({'session_id': 'count'}).rename(
    columns={'session_id': 'all_events_month'}).reset_index()

    for i, date in enumerate(dates):
      #print(date)
      first_month, first_year = date.month, date.year
     
      buying_events_first_month = buying_events_month[
          (buying_events_month['month'] == first_month) & (buying_events_month['year'] == first_year)]

      if first_month == 12:
        second_month = 1
        second_year=first_year+1
      else:
        second_month = first_month + 1
        second_year=first_year

      buying_events_second_month = buying_events_month[
          (buying_events_month['month'] == second_month) & (buying_events_month['year'] == second_year)]

      ######## just to make merge easier
      buying_events_second_month['month'] = first_month
      buying_events_second_month['year'] = first_year

 
      all_events_first_month = all_events_month[
          (all_events_month['month'] == first_month) & (all_events_month['year'] == first_year)]
     
      all_events_second_month = all_events_month[
          (all_events_month['month'] == second_month) & (all_events_month['year'] == second_year)]


      ######## just to make merge easier
      all_events_second_month['month'] = first_month
      all_events_second_month['year'] = first_year


      #print("first", all_events_first_month.head())
      #print("second", all_events_second_month.head())


      df=pd.merge(buying_events_first_month, buying_events_second_month, how='left', on=['year', 'month', 'user_id'])#, 
  
      df=pd.merge(df, all_events_first_month, how='left', on=['year', 'month', 'user_id'])
      df=pd.merge(df, all_events_second_month, how='left', on=['year', 'month', 'user_id'])

      df.fillna(0,inplace=True)


      df['first_month_sin']=np.sin(df['month'] * (2. * np.pi / 12))
      df['first_month_cos']=np.cos(df['month'] * (2. * np.pi / 12))


      buying_sessions = events_data[events_data['event_type'] == 'BUY_PRODUCT']
  

      deals = pd.merge(buying_sessions, products_data[['product_id','price']], how='left', on=['product_id'])
      deals['final_price'] = deals['price']
      #deals = deals[(deals['timestamp'] > data_begin) & (deals['timestamp'] < data_end)]

      pd.set_option("display.max_rows", None, "display.max_columns", None)

      spent_money_first_month = deals[(deals.timestamp.dt.month == first_month) & (deals.timestamp.dt.year == first_year)].groupby('user_id').aggregate(  
          {'final_price': 'sum'}).rename(columns={'final_price': 'spent_money_first_month'}).reset_index()

      spent_money_second_month = deals[(deals.timestamp.dt.month == second_month) & (deals.timestamp.dt.year ==second_year)].groupby('user_id').aggregate(
          {'final_price': 'sum'}).rename(
          columns={'final_price': 'spent_money_second_month'}).reset_index()

      if second_month == 12:
        third_month = 1
        third_year=second_year+1
      else:
        third_month = second_month + 1
        third_year=second_year


      spent_money_third_month = deals[(deals.timestamp.dt.month == third_month) & (deals.timestamp.dt.year == third_year)].groupby('user_id').aggregate(
          {'final_price': 'sum'}).rename(
          columns={'final_price': 'spent_money_third_month'}).reset_index()

      spent_money_average = pd.merge(spent_money_first_month, spent_money_second_month, how='left', on=['user_id'])
      spent_money_average['spent_money_second_month'].fillna(0, inplace=True)
      spent_money_average['spent_money_total_average'] = spent_money_average[['spent_money_first_month',
                                                                              'spent_money_second_month']].sum(axis=1) / 2


      spent_money_average = pd.merge(spent_money_average, spent_money_third_month, how='left', on=['user_id'])
      spent_money_average['spent_money_third_month'].fillna(0, inplace=True)

      spent_money_average['label'] = np.where(spent_money_average['spent_money_third_month'] - spent_money_average['spent_money_total_average'] > 100, 1, 0)
      processed_data = pd.merge(df, spent_money_average[['spent_money_first_month', 'spent_money_second_month', 'user_id', 'label']], how='left', on=['user_id'])

      processed_data['label'].fillna(0, inplace=True)
      processed_data = processed_data.sort_values(by=['user_id']).reset_index()


      del processed_data['index']
      del processed_data['user_id']
      del processed_data['month']
 
      processed_data = processed_data.drop_duplicates(keep='first')
      #print(processed_data.shape)
      if i==0:
        dataset=processed_data
      else:
        dataset=pd.concat([dataset, processed_data])

    return dataset
      
df_train=prepare_data('2019.09.01', '2021.12.31')
df_train.to_csv('processed_data_train.csv')
