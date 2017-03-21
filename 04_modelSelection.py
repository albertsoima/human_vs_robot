import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import random

#Import generated features and outcomes
bid = pd.read_csv("full_features.csv", index_col=0)
bid = bid.drop(["address", "payment_account"], axis=1)
bid = bid[(bid.outcome==0) | (bid.numbids > 10)]
full_time = pd.read_csv("lag_time.csv", index_col=0)
X = bid.iloc[:,2:]
Y = bid.iloc[:,1]

#SVM for feature selection
lsvc = LinearSVC(C=1, penalty="l1", dual=False).fit(X, Y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)

##########################################
# Calculate ROC score for each algorithm #
##########################################

#Logisitic Regression
print 'Logistic'
algo_log = LogisticRegression()
algo_log.fit(X_new,Y)
hyp = algo_log.predict(X_new)
kfold = KFold(n_splits=20, shuffle=True, random_state=100)
score = cross_val_score(algo_log, X_new, Y, cv=kfold, scoring="roc_auc")
preds = algo_log.predict_proba(X_new)
print "On Train: ", metrics.roc_auc_score(Y, preds[:,1])
print "Cross-Val: ", np.mean(score)

#LDA
print 'LDA'
prior_vector = [0.9,0.1]
algo_LDA = LinearDiscriminantAnalysis(priors=prior_vector)
algo_LDA.fit(X_new,Y)
hyp = algo_LDA.predict(X_new)
kfold = KFold(n_splits=20, shuffle=True, random_state=100)
score = cross_val_score(algo_LDA, X_new, Y, cv=kfold, scoring="roc_auc")
preds = algo_LDA.predict_proba(X_new)
print "On Train: ", metrics.roc_auc_score(Y, preds[:,1])
print "Cross-Val: ", np.mean(score)

#Random Forest
print 'Random Forest'
algo_rf = RandomForestClassifier(170)
algo_rf.fit(X_new,Y)
hyp = algo_rf.predict(X_new)
kfold = KFold(n_splits=20, shuffle=True, random_state=100)
score = cross_val_score(algo_rf, X_new, Y, cv=kfold, scoring="roc_auc")
preds = algo_rf.predict_proba(X_new)
print "On Train: ", metrics.roc_auc_score(Y, preds[:,1])
print "Cross-Val: ", np.mean(score)

#SVM
print 'SVM'
algo_svm = SVC(probability=True)
algo_svm.fit(X_new,Y)
hyp = algo_svm.predict(X_new)
kfold = KFold(n_splits=20, shuffle=True, random_state=100)
score = cross_val_score(algo_svm, X_new, Y, cv=kfold, scoring="roc_auc")
preds = algo_svm.predict_proba(X_new)
print "On Train:", metrics.roc_auc_score(Y, preds[:,1])
print "Cross-Val: ", np.mean(score)

#AdaBoost
print "AdaBoost"
algo_ada = AdaBoostClassifier(learning_rate=0.3, n_estimators=170)
algo_ada.fit(X_new,Y)
kfold = KFold(n_splits=20, shuffle=True, random_state=100)
score = cross_val_score(algo_ada, X_new, Y, cv=kfold, scoring="roc_auc")
preds = algo_ada.predict_proba(X_new)
print "On Train: ", metrics.roc_auc_score(Y, preds[:,1])
print "Cross-Val: ", np.mean(score)

#Decision Tree
print "Decision Tree"
parameters = {'max_depth':range(3,20)}
algo_dt = GridSearchCV(DecisionTreeClassifier(), parameters)
algo_dt.fit(X_new,Y)
kfold = KFold(n_splits=20, shuffle=True, random_state=100)
score = cross_val_score(algo_dt, X_new, Y, cv=kfold, scoring="roc_auc")
preds = algo_dt.predict_proba(X_new)
print "On Train: ", metrics.roc_auc_score(Y, preds[:,1])
print "Cross-Val: ", np.mean(score)

#Bagging
print "Bagging"
parameters = {'n_estimators':range(10, 100, 10)}
algo_bag = GridSearchCV(BaggingClassifier(), parameters)
algo_bag.fit(X_new,Y)
kfold = KFold(n_splits=20, shuffle=True, random_state=100)
score = cross_val_score(algo_bag, X_new, Y, cv=kfold, scoring="roc_auc")
preds = algo_bag.predict_proba(X_new)
print "On Train: ", metrics.roc_auc_score(Y, preds[:,1])
print "Cross-Val: ", np.mean(score)

#KNN
print "KNN"
parameters = {'n_neighbors':range(2,10)}
algo_knn = GridSearchCV(KNeighborsClassifier(), parameters)
algo_knn.fit(X_new,Y)
kfold = KFold(n_splits=20, shuffle=True, random_state=100)
score = cross_val_score(algo_knn, X_new, Y, cv=kfold, scoring="roc_auc")
preds = algo_knn.predict_proba(X_new)
print "On Train: ", metrics.roc_auc_score(Y, preds[:,1])
print "Cross-Val: ", np.mean(score)

#Plotting ROC curve for some models
random.seed(100)
msk = np.random.rand(len(bid)) < 0.5
train = bid[msk]
test = bid[~msk]
trainX = train.iloc[:,2:]
trainY = train.iloc[:,1]
testX = test.iloc[:,2:]
testY = test.iloc[:,1]
trainX = model.transform(trainX)
testX = model.transform(testX)
rf_plot = RandomForestClassifier(170)
rf_plot.fit(trainX, trainY)
log_plot = LogisticRegression()
log_plot.fit(trainX, trainY)
svm_plot = SVC(probability=True)
svm_plot.fit(trainX, trainY)
LDA_plot = LinearDiscriminantAnalysis(priors=prior_vector)
LDA_plot.fit(trainX, trainY)
ada_plot = AdaBoostClassifier(learning_rate=0.3, n_estimators=170)
ada_plot.fit(trainX, trainY)
y_pred_rf = rf_plot.predict_proba(testX)[:, 1]
y_pred_log = algo_log.predict_proba(testX)[:, 1]
y_pred_svm = svm_plot.predict_proba(testX)[:, 1]
y_pred_LDA = LDA_plot.predict_proba(testX)[:, 1]
y_pred_ada = ada_plot.predict_proba(testX)[:, 1]
fpr_rf, tpr_rf, _ = metrics.roc_curve(testY, y_pred_rf)
fpr_log, tpr_log, _ = metrics.roc_curve(testY, y_pred_log)
fpr_svm, tpr_svm, _ = metrics.roc_curve(testY, y_pred_svm)
fpr_LDA, tpr_LDA, _ = metrics.roc_curve(testY, y_pred_LDA)
fpr_ada, tpr_ada, _ = metrics.roc_curve(testY, y_pred_ada)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_log, tpr_log, label='Logistic')
plt.plot(fpr_LDA, tpr_LDA, label='LDA')
plt.plot(fpr_rf, tpr_rf, label='Random Forest')
plt.plot(fpr_ada, tpr_ada, label='AdaBoost')
plt.plot(fpr_svm, tpr_svm, label='SVM')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
