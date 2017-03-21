"""This script creates the features involving number of bids per bidder_id, within and across auctions,
then combines all of the created features into one csv file. This is done for the training and test sets
of the kaggle data"""

import pandas as pd
from pandasql import sqldf
import numpy as np

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_bid = pd.read_csv('bids.csv')

pysqldf = lambda q: sqldf(q, globals())

q = """SELECT bidder_id, SUM(aucbids) AS numbids, MAX(aucbids) AS maxaucbids, AVG(aucbids) AS avgaucbids FROM (
    SELECT bidder_id, count(*) AS aucbids FROM df_bid GROUP BY bidder_id, auction) GROUP BY bidder_id;
"""
df_numbids = pysqldf(q)

df_feature = pd.read_csv('feature.csv')
df_feature_test = pd.read_csv('feature_test.csv')
df_lag = pd.read_csv('lag_time_reduced.csv')
df_lag_test = pd.read_csv('test_lag_time_reduced.csv')

intermediate = pd.merge(df_train, df_numbids, on='bidder_id', how='left')
intermediate = intermediate.fillna(0)
intermediate_test = pd.merge(df_test, df_numbids, on='bidder_id', how='left')
intermediate_test = intermediate_test.fillna(0)
intermediate2 = pd.merge(intermediate, df_feature, on='bidder_id', how='left')
intermediate2_test = pd.merge(intermediate_test, df_feature_test, on='bidder_id', how='left')

final = pd.merge(intermediate2, df_lag, on='bidder_id', how='left')
final_test = pd.merge(intermediate2_test, df_lag_test, on='bidder_id', how='left')

final.to_csv('full_features.csv')
final_test.to_csv('full_features_test.csv')



