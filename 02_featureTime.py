import pandas as pd
import numpy as np

#Import data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_bid = pd.read_csv('bids.csv')

#Join features with bidder_id for both train and test set
df_train_all = df_bid.set_index('bidder_id').join(df_train.set_index('bidder_id'), how="right")
df_train_all['bidder_id'] = df_train_all.index
df_test_all = df_bid.set_index('bidder_id').join(df_test.set_index('bidder_id'), how="right")
df_test_all['bidder_id'] = df_test_all.index

#Reduce time magnitude
df_train_all["time"] = df_train_all.time.apply(lambda x: x/1000000000000)
df_train_all["time"] = df_train_all.time.apply(lambda x: x/1000000000000)

#Calculate lag time and replace with median for one time bids
df_train_all["lag_within_auction"] = df_train_all.groupby(['bidder_id','auction']).time.transform(np.diff)
med1 = np.nanmedian(df_train_all['lag_within_auction'])
df_train_all.fillna(med1)

df_train_all["lag_between_auction"] = df_train_all.groupby(['bidder_id']).time.transform(np.diff)
med2 = np.nanmedian(df_train_all['lag_between_auction'])
df_train_all.fillna(med2)

df_test_all["lag_within_auction"] = df_test_all.groupby(['bidder_id','auction']).time.transform(np.diff)
med1 = np.nanmedian(df_test_all['lag_within_auction'])
df_test_all.fillna(med1)

df_test_all["lag_between_auction"] = df_test_all.groupby(['bidder_id']).time.transform(np.diff)
med2 = np.nanmedian(df_test_all['lag_between_auction'])
df_test_all.fillna(med2)

#Replace with median for abnormally large number (potentially bids between days)
df_train_all["lag_within_auction"] = df_train_all["lag_within_auction"].apply(lambda x: x if x < 9000 else med1)
df_train_all["lag_between_auction"] = df_train_all["lag_between_auction"].apply(lambda x: x if x < 9000 else med2)
df_test_all["lag_within_auction"] = df_test_all["lag_within_auction"].apply(lambda x: x if x < 9000 else med1)
df_test_all["lag_between_auction"] = df_test_all["lag_between_auction"].apply(lambda x: x if x < 9000 else med2)

#Get mean and min of each bidder_id within and between auction
df_time =  pd.DataFrame()
df_time["time_within_minimum"] = df_train_all.groupby(['bidder_id']).lag_within_auction.min()
df_time["time_within_average"] = df_train_all.groupby(['bidder_id']).lag_within_auction.mean()
df_time["time_between_minimum"] = df_train_all.groupby(['bidder_id']).lag_between_auction.min()
df_time["time_between_avergae"] = df_train_all.groupby(['bidder_id']).lag_between_auction.mean()

df_time_test =  pd.DataFrame()
df_time_test["time_within_minimum"] = df_test_all.groupby(['bidder_id']).lag_within_auction.min()
df_time_test["time_within_average"] = df_test_all.groupby(['bidder_id']).lag_within_auction.mean()
df_time_test["time_between_minimum"] = df_test_all.groupby(['bidder_id']).lag_between_auction.min()
df_time_test["time_between_avergae"] = df_test_all.groupby(['bidder_id']).lag_between_auction.mean()

#Save features to csv
df_time.to_csv("lag_time_reduced.csv", header=True, index=True)
df_time_test.to_csv("test_lag_time_reduced.csv", header=True, index=True)


