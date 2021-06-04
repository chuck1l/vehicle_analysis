import pandas as pd
from preprocessing import *
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def get_price_prediction(df, df_final, y):
    '''
    Get the final prediction for price
    '''
    X = df.copy()
    X_final = df_final.copy()

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
    y_final = xgb_model.predict(X_final)
    # Baseline RMSE from defautl params = 38.65
    rmse = np.sqrt(mean_absolute_error(y_hat, y_test))
    print(rmse)

    X_test = X_test.reset_index()
    X_test['PriceTrue'] = y_test
    X_test['PricePredicted'] = y_hat
    df_test_out = X_test[['ListingID', 'PriceTrue', 'PricePredicted']]

    X_final = X_final.reset_index()
    X_final['PricePredicted'] = y_final
    X_final['PricePredicted'] = X_final['PricePredicted'].round(2)
    df_out = X_final[['ListingID', 'PricePredicted']] 

    return df_out, df_test_out

if __name__ == '__main__':

    price = get_price_prediction()

    breakpoint()
    print('stophere')