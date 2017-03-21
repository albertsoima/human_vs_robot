import pandas as pd

### all features
df_bid = pd.read_csv('bids.csv')

### train feature
df_train = pd.read_csv('train.csv')
df_train_all = pd.merge(df_train, df_bid, on=['bidder_id', 'bidder_id'], how='left')

# convert merchandise to dummy variables
df_dummy = pd.get_dummies(df_train_all, columns=["merchandise"])
df_m = df_dummy.groupby(['bidder_id']).sum()
df = df_m.iloc[:, 3:]
m = df.idxmax(axis=1)
m.to_csv('m.csv')

# get unique number of ip and country per bidder
df_train_all["ip"] = df_train_all.ip.apply(lambda x: str(x).replace('.',''))
df_ip = df_train_all.groupby(['bidder_id']).ip.nunique()
df_country = df_train_all.groupby(['bidder_id']).country.nunique()
df_ip.to_csv('ip.csv')
df_country.to_csv('country.csv')


### test feature
df_test = pd.read_csv('test.csv')
df_test_all = pd.merge(df_test, df_bid, on=['bidder_id', 'bidder_id'], how='left')

# convert merchandise to dummy variables
df_dummy = pd.get_dummies(df_test_all, columns=["merchandise"])
df_m = df_dummy.groupby(['bidder_id']).sum()
df = df_m.iloc[:, 2:]
m = df.idxmax(axis=1)
m.to_csv('m_test.csv')

# get unique number of ip and country per bidder
df_test_all["ip"] = df_test_all.ip.apply(lambda x: str(x).replace('.',''))
df_ip = df_test_all.groupby(['bidder_id']).ip.nunique()
df_country = df_test_all.groupby(['bidder_id']).country.nunique()
df_ip.to_csv('ip_test.csv')
df_country.to_csv('country_test.csv')