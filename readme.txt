This file describes the flow and functions of the project scripts. Scripts are sorted by chronological order with 00 prefix being the first and 05 prefix being the last. This zip file also contain the final feature set for building our model.

#00_featureIPMerchandiseCountry.py
Extracting features based on Ip, merchandise and country

#01_mergeIPMerchandiseCountry.py
Merging Ip, merchandise and country into on csv file

#02_featureTime.py
Extracting features based on time

#03_featureBidsVolumn.py
Extracting features based on bids volume. This script also combines all the previous generated features into one csv file which contains all the features to build the model. Please see full_features.csv and full_features_test.csv

#04_modelSelection.py
Fitting different models by performing cross validation on the training set

#05_kagglePrediction
Use the training model to obtain Kaggle test prediction

#full_features.csv and full_features_test.csv
Full features set used for training model and test set

Note: the raw data is available at the Kaggle site:
https://www.kaggle.com/c/facebook-recruiting-iv-human-or-bot