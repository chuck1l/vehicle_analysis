from typing import Tuple
from vehicle_average_pricing import *
import pandas as pd
import numpy as np
from feature_engineering import *


def preprocess(data):
    df = data.copy()
    # Remove Vehicle Body Style, type, transmission, no value to computation ratio
    df.drop(columns=['VehBodystyle', 'VehType', 'VehTransmission'], inplace=True)

    # After viewing the data, nan Veh Trim columns are nan in a lot of the features too
    # Removing them to ideally minimize the noise
    cols = df.columns
    if 'Vehicle_Trim' in cols:
        df = df[df['Vehicle_Trim'].notnull()].reset_index(drop=True)
        
    if 'Dealer_Listing_Price' in cols:
        df = df[df['Dealer_Listing_Price'].notnull()].reset_index(drop=True)

    # Assuming a fraction of a day won't provide enough value for added computation, int
    df['VehListdays'] = df['VehListdays'].fillna(df['VehListdays'].mean())
    df['VehListdays'] = df['VehListdays'].astype(int)

    df['SellerListSrc'] = df['SellerListSrc'].fillna('unspecified')

    # Address the issues within the Seller Name
    df['SellerName'] = df['SellerName'].str.replace(
        '&amp;', ' & ').str.replace('&#x27;', "'")

    df['SellerName'] = df['SellerName'].apply(lambda x: ' '.join(x.split())) 

    # Address all found issues in the Seller Notes
    df = CleanText(df, 'VehSellerNotes').perform_all_tasks()

    # Lower case all string columns in the dataframe
    df = df.applymap(lambda x: x.lower() if type(x) == str else x)

    # Reduce the excessive number of exterior colors
    df = ext_color_reduction(df, 'VehColorExt')
    df = int_color_reduction(df, 'VehColorInt')
    df = drive_reduction(df, 'VehDriveTrain')
    df = engine_reduction(df, 'VehEngine')
    df = fuel_cleaning(df, 'VehFuel')
    df = history_cleaning(df, 'VehHistory')
    df = feats_preparation(df, 'VehFeats')
    df = num_pricelabel(df, 'VehPriceLabel')
    df = check_notes_trim(df, 'VehSellerNotes')
    df = vect_notes(df, 'VehSellerNotes')

    create_avg_pricing = False
    if create_avg_pricing:
        average_pricing(df)

    df = create_range(df)

    avg_price_df = pd.read_excel('../data/avg_price.xlsx')

    join_cols_avg = [
        'SellerState', 'SellerCity', 'VehCertified', 'VehFuel',
        'mile_range', 'VehModel', 'VehYear', 'VehDriveTrain',
        'VehEngine', 'VehColorExt'
    ]

    df = df.merge(avg_price_df, how='left', left_on=join_cols_avg, right_on=join_cols_avg)
    df.drop(columns='mile_range', axis=1, inplace=True)

    # Converting Sell is private and Vehicle certified to 0 and 1
    df['SellerIsPriv'] = np.where(df['SellerIsPriv'] == True, 1, 0)
    df['VehCertified'] = np.where(df['VehCertified'] == True, 1, 0)

    # Removing zip code from the df to reduce dimensions, assuming colinear with
    # City and State
    df.drop(columns='SellerZip', axis=1, inplace=True)

    # Removing the Vehicle Make, since there is only one model per make
    df.drop(columns='VehMake', axis=1, inplace=True)
    
    return df

if __name__ == '__main__':

    # Testing the code
    df = pd.read_csv('../data/Training_DataSet.csv')

    preprocess(df)