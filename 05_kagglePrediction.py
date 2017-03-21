import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
import numpy as np

#Read in data
bid = pd.read_csv("full_features.csv", index_col=0)
bid = bid.drop(["address", "payment_account"], axis=1)
bid = bid[(bid.outcome==0) | (bid.numbids > 10)]
test = pd.read_csv("full_features_test.csv", index_col=0)
test = test.drop("address", axis=1)
X = bid.iloc[:,2:]
Y = bid.iloc[:,1]
testX = test.iloc[:,2:]
testY = test.iloc[:,1]

#SVM
lsvc = LinearSVC(C=1, penalty="l1", dual=False).fit(X, Y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)
testX = model.transform(testX)


#Random Forest
print 'Random Forest'
algo_rf = RandomForestClassifier(280)
algo_rf.fit(X_new,Y)
hyp = algo_rf.predict(X_new)
kfold = KFold(n_splits=20, shuffle=True, random_state=200)
score = cross_val_score(algo_rf, X_new, Y, cv=kfold, scoring="roc_auc")
preds = algo_rf.predict_proba(X_new)
print "On Train: ", metrics.roc_auc_score(Y, preds[:,1])
print "Cross-Val: ", np.mean(score)

#Get test prediction and write to csv for Kaggle evaluation
prediction = algo_rf.predict_proba(testX)
kaggle =  pd.DataFrame()
kaggle["bidder_id"] = test.iloc[:,0]
kaggle["prediction"] = prediction[:,1]
kaggle.to_csv("kaggle.csv", header=True, index=False)






