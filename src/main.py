from training_trim import get_trim_prediction
from training_price import get_price_prediction
from preprocessing import preprocess
import pandas as pd
import numpy as np


data = pd.read_csv('../data/Training_DataSet.csv')
data_final = pd.read_csv('../data/Test_Dataset.csv')

data = data[data['Vehicle_Trim'].notnull()].reset_index(drop=True)
data = data[data['Dealer_Listing_Price'].notnull()].reset_index(drop=True)

test_ids = data_final['ListingID'].to_list()

y_price = data['Dealer_Listing_Price']
y_trim = data['Vehicle_Trim']

data.drop(columns=['Vehicle_Trim', 'Dealer_Listing_Price'], axis=1, inplace=True)

preprocess_df = pd.concat([data, data_final])
processed_df = preprocess(preprocess_df)

processed_df = pd.get_dummies(processed_df)
processed_df['ListingID'] = processed_df['ListingID'].astype(int)

df = processed_df[~processed_df['ListingID'].isin(test_ids)]
df_final = processed_df[processed_df['ListingID'].isin(test_ids)]

# Setting the ListingID as index
df.set_index('ListingID', inplace=True)
df_final.set_index('ListingID', inplace=True)

# Ensure all columns are float for XGBoost
cols = df.columns
df[cols] = df[cols].astype(float).round(2)
df_final[cols] = df_final[cols].astype(float).round(2)

y_hat_price, df_test_price = get_price_prediction(df, df_final, y_price)
y_hat_trim, df_test_trim = get_trim_prediction(df, df_final, y_trim)

results = pd.merge(y_hat_trim, y_hat_price, how='left', left_on='ListingID', right_on='ListingID')
results.to_csv('../prediction_results.csv', index=False)

testing_results = pd.merge(df_test_trim, df_test_price, how='outer', left_on='ListingID', right_on='ListingID')
testing_results.to_csv('../testing_results_df.csv', index=False)

breakpoint()
print('stop here')