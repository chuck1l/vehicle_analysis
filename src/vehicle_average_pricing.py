import pandas as pd
import numpy as np


def create_range(data):

    df = data.copy()
    miles_cond = [
        df['VehMileage'] == 0,
        df['VehMileage'].between(1, 10000),
        df['VehMileage'].between(10001, 20000),
        df['VehMileage'].between(20001, 30000),
        df['VehMileage'].between(30001, 40000),
        df['VehMileage'].between(40001, 50000),
        df['VehMileage'].between(50001, 60000),
        df['VehMileage'].between(60001, 70000),
        df['VehMileage'].between(70001, 80000),
        df['VehMileage'] > 80001
    ]
    miles_val = [
        'new', '10k', '20k', '30k', '40k', '50k', '60k', '70k', '80k', '>80k'
    ]

    df['mile_range'] = np.select(
        miles_cond, miles_val)
    
    return df


def average_pricing(data):
    '''
    This function is sourcing an average price by grouped features,
    and merges it with the dataframe. Similar to as if we were able 
    to use a Kelly Blue Book value or something. All information is
    from the training data. (See Tableau Workbook for how it was derived)

    Parameters: The dataframe

    Returns: The DataFrame joined with the created pricing list.
    '''
    df = data.copy()
    cols = [
        'VehColorExt',
        'SellerState',
        'SellerCity',
        'VehCertified',
        'VehFuel',
        'VehMileage',
        'VehModel',
        'VehYear',
        'VehDriveTrain',
        'VehEngine',
        'Dealer_Listing_Price'
    ]
    df = df[cols]

    df = create_range(df)

    df.drop(columns=['VehMileage'])
    # Group by desired columns to create average pricing
    grouping = [
        'VehColorExt',
        'SellerState',
        'SellerCity',
        'VehCertified',
        'VehFuel',
        'mile_range',
        'VehModel',
        'VehYear',
        'VehDriveTrain',
        'VehEngine'
    ]

    df = df.groupby(grouping)[
        'Dealer_Listing_Price'].mean().reset_index(name='avg_price')

    df['avg_price'] = df['avg_price'].astype(int)

    df.to_excel('../data/avg_price.xlsx')

    return None


if __name__ == '__main__':
    
    df_train = pd.read_csv('../data/Training_DataSet.csv')
    df_test = pd.read_csv('../data/Test_Dataset.csv')

    test = average_pricing(df_train)
