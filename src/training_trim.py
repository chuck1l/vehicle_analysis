import pandas as pd
import numpy as np
from preprocessing import *
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn import preprocessing

df = pd.read_csv('../data/Training_DataSet.csv')
# Feature egineering function applied
df = preprocess(df)
# Dropping the Vehicle Price, no data leakage 
df.drop(columns='Dealer_Listing_Price', axis=1, inplace=True)
# Establish the features and targets DFs
y = df['Vehicle_Trim']
labels=y.unique()
#y = preprocessing.LabelEncoder().fit_transform(y)

X = df.drop(columns='Vehicle_Trim')

# Dummify the remaining text columns
X = pd.get_dummies(X)
# Ensure all columns are float as per xgb preference
cols = X.columns
X[cols] = X[cols].astype(float).round(2)

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=True, random_state=42
    )

hyper_param_tuning = False

if hyper_param_tuning:

    xgb_model = XGBClassifier(eval_metric='mlogloss')

    param_grid = {
            'n_estimators': [450, 500, 550, 600],
            'eta': [.2, .5, 1],
            'max_depth': [17, 20, 22],
            'colsample_bytree': [0.1, 0.5, 0.7],
            'reg_lambda': [1, 3, 5],
            'reg_alpha':[0.05, 1e-1, .5],
            'subsample': [.1, .3, 0.5]
            }

    rs_clf = RandomizedSearchCV(
            xgb_model, param_grid, n_iter=20,
            n_jobs=1, verbose=2, cv=3,
            scoring='neg_log_loss', refit=False,
            random_state=42
        )

    print('Randomized Search for Best Parameters...')
    rs_clf.fit(X_train, y_train)
    
    print('Optimal hyperparameters have been identified.')

    best_score = rs_clf.best_score_
    best_params = rs_clf.best_params_

    print('Best Score: ', best_score)
    print('Best Params: ')
    for param_name in best_params.keys():
        print(param_name, best_params[param_name])

# Perform analysis with best hyperparameters
test_params = [1, 1.3, 1.7, 3]

for i in test_params:
    xgb_model = XGBClassifier(
            eval_metric='mlogloss',
            n_estimators=600, 
            eta=.1, 
            max_depth=15, 
            colsample_bytree=0.8, 
            reg_lambda=i, 
            reg_alpha=1e-7, 
            subsample=1 
        )

    xgb_model.fit(X_train, y_train)
    y_hat = xgb_model.predict(X_test)

    confusion = confusion_matrix(y_test, y_hat)
    accuracy = accuracy_score(y_test, y_hat)
    precision = precision_score(y_test, y_hat, average='macro')
    recall = recall_score(y_test, y_hat, average='macro')
    f1 = f1_score(y_test, y_hat, average='macro')

    print('max_depth: ', i)
    print(f'Accuracy Score: {accuracy} \nPrecision Score: {precision} \nRecall Score: {recall} \nF1 Score: {f1}')
