import pandas as pd

## train data
df_merchandise = pd.read_csv('m.csv')
df_country = pd.read_csv('country.csv')
df_ip = pd.read_csv('ip.csv')
df_dummy2 = pd.get_dummies(df_merchandise, columns=["merchandise"])

df_feature0 = pd.merge(df_country, df_ip, on=['bidder_id', 'bidder_id'], how='left')
df_feature = pd.merge(df_feature0, df_dummy2, on=['bidder_id', 'bidder_id'], how='left')
df_feature.to_csv('feature.csv')

## test data
df_merchandise = pd.read_csv('m_test.csv')
df_country = pd.read_csv('country_test.csv')
df_ip = pd.read_csv('ip_test.csv')
df_dummy2 = pd.get_dummies(df_merchandise, columns=["merchandise"])

df_feature0 = pd.merge(df_country, df_ip, on=['bidder_id', 'bidder_id'], how='left')
df_feature = pd.merge(df_feature0, df_dummy2, on=['bidder_id', 'bidder_id'], how='left')
df_feature.to_csv('feature_test.csv')