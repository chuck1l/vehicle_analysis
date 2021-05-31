import pandas as pd
import numpy as np
from data_cleaning_functions import CleanText, ext_color_reduction 
from data_cleaning_functions import int_color_reduction, drive_reduction
from data_cleaning_functions import ingine_reduction


def preprocess(df):
    # Remove Vehicle Body Style, all SUV so no value to computation ratio
    df.drop(columns='VehBodystyle', inplace=True)
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

    breakpoint()



    breakpoint()
    print('stop here for analysis')
    print(df.info())

if __name__ == '__main__':

    # Testing the code
    df = pd.read_csv('../data/Training_DataSet.csv')

    preprocess(df)