import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('../data/Training_DataSet.csv')
df_test = pd.read_csv('../data/Test_Dataset.csv')

df.to_excel('../data/train_excel.xlsx')



