from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

from IPython.display import SVG
from graphviz import Source
from IPython.display import display
from IPython.display import HTML
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_curve, auc, accuracy_score
import numpy as np


style = '<style>svg{width:10% !importamt;height:10% !important;}<style>'
HTML(style)

pd.set_option('display.max_columns', None)
titanic_data = pd.read_csv('train.csv', index_col='PassengerId')
titanic_test = pd.read_csv('test.csv', index_col='PassengerId')

print(titanic_data.info())
print(titanic_test.info())
#print(titanic_data.isnull().sum())
X = titanic_data.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis=1)#
Y_train = titanic_data.Survived
X = pd.get_dummies(X)
X_train = X.fillna({'Age': X.Age.median()})
print(X.head())
# X_test = titanic_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)
# X_test = pd.get_dummies(X_test)
# X_test = X_test.fillna({'Age': X_test.Age.median()})
# X_test = X_test.fillna({'Fare': X_test.Fare.median()})
#
# print(X_test.isnull().sum())

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.33, random_state=42)



# clf = tree.DecisionTreeClassifier()
# parametrs = {'criterion': ['gini', 'entropy'], 'max_depth': range(1, 30)}
# grid_search_cv_clf = GridSearchCV(clf, parametrs, cv=5)
# grid_search_cv_clf.fit(X_train, Y_train)
# print(grid_search_cv_clf.best_params_)
# best_clf = grid_search_cv_clf.best_estimator_

rf = RandomForestClassifier(criterion='entropy', max_depth=6, n_estimators=93)
# parametrs = {'n_estimators': range(30, 100, 3), 'criterion': ['gini', 'entropy'], 'max_depth': range(1, 30)}
# grid_search_cv_clf = GridSearchCV(rf, parametrs, cv=5)
# grid_search_cv_clf.fit(X_train, Y_train)
# print(grid_search_cv_clf.best_params_)
# best_clf = grid_search_cv_clf.best_estimator_
#print(best_clf.score(X_train, Y_train))
rf.fit(X_train, Y_train)
best_clf = rf

print('score', best_clf.score(X_test, Y_test))
print('cross_val_score', cross_val_score(best_clf, X_test, Y_test).mean())

print(cross_val_score(best_clf, X_test, Y_test).mean())
y_pred = best_clf.predict(X_test)
print(accuracy_score(Y_test, y_pred))
# Submission = pd.DataFrame({'Survived': y_pred})
# Submission.index = X_test.index
# print(Submission.head())
# Submission.to_csv('Submission.csv')


print('prcision:', precision_score(Y_test, y_pred), 'recall:', recall_score(Y_test, y_pred))
y_predicted_prob = best_clf.predict_proba(X_test)
#print(y_predicted_prob)#так хранятся вероятности каждого предсказания
y_pred = np.where(pd.Series(y_predicted_prob[:, 1]) > 0.4, 1, 0)#изменяя порог отсичения, изменяем в разные стороны precidion и recall
print('prcision:', precision_score(Y_test, y_pred), 'recall:', recall_score(Y_test, y_pred))
print(accuracy_score(Y_test, y_pred))

#
#
#
# fpr, tpr, thresholds = roc_curve(Y_test, y_predicted_prob[:,1])
# roc_auc= auc(fpr, tpr)
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()
#
# plt.show()