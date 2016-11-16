import csv
import pandas as pd
import time

a = time.time()

all = []
train = []
with open('bids.csv', 'rb') as f:
    raw = csv.reader(f, delimiter=',')
    for row in raw:
        all.append(row)

with open('train.csv', 'rb') as f:
    raw = csv.reader(f, delimiter=',')
    for row in raw:
        train.append(row)

dfinfo = pd.DataFrame(all)
dftrain = pd.DataFrame(train)

print dfinfo
print time.time() - a
