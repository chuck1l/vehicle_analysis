import pandas as pd
from preprocessing import *
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

df = pd.read_csv('../data/Training_DataSet.csv')
# Feature egineering function applied
df = preprocess(df)
# Dropping the Vehicle Trim, no data leakage 
df.drop(columns='Vehicle_Trim', axis=1, inplace=True)
# Establish the features and targets DFs
y = df['Dealer_Listing_Price']
X = df.drop(columns='Dealer_Listing_Price')

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

    xgb_model = XGBRegressor()

    param_grid = {
            'n_estimators': [450],
            'eta': [.05],
            'max_depth': [5, 10],
            'colsample_bytree': [0.8],
            'reg_lambda': [0.5],
            'reg_alpha':[1e-15, 1e-10, 1e-7],
            'subsample': [0.5]
            }
    rs_reg = RandomizedSearchCV(
            xgb_model, param_grid, n_iter=10,
            n_jobs=1, verbose=2, cv=3,
            scoring='neg_root_mean_squared_error', refit=False,
            random_state=42
        )

    print('Randomized Search for Best Parameters...')
    rs_reg.fit(X_train, y_train)
    
    print('Optimal hyperparameters have been identified.')

    best_score = rs_reg.best_score_
    best_params = rs_reg.best_params_

    print('Best Score: ', best_score)
    print('Best Params: ')
    for param_name in best_params.keys():
        print(param_name, best_params[param_name])

# Perform analysis with best hyperparameters
xgb_model = XGBRegressor(
        n_estimators=450,
        eta=.05,
        max_depth=10,
        colsample_bytree=0.8,
        reg_lambda=.5,
        reg_alpha=1e-07,
        subsample=0.5
    )

xgb_model.fit(X_train, y_train)
y_hat = xgb_model.predict(X_test)
# Baseline RMSE from defautl params = 38.65
rmse = np.sqrt(mean_absolute_error(y_hat, y_test))
print(rmse)

