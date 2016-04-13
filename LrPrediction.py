# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 14:49:37 2016

@author: Bleach
"""
import pandas as pd
from sklearn import linear_model

# 用正则取出我们要的属性值
train_np = train_df.as_matrix()
data_test = pd.read_csv('test.csv')

# y即Survival结果
y = train_np[:, 0]
# X即特征属性值
X = train_np[:, 1:]

# fit到RandomForestRegressor之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X, y)

clf

predictions = clf.predict(test_df)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("logistic_regression_predictions.csv", index=False)