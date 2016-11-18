import pandas as pd
import time

a = time.time()

df_train = pd.read_csv('train.csv')
df_bid = pd.read_csv('bids.csv')


df_train_all = df_bid.set_index('bidder_id').join(df_train.set_index('bidder_id'), how="right")

print df_train_all

train_X = df_train_all.iloc[:, :-1].values
