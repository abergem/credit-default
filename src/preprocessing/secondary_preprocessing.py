"""
Python module to do secondary preprocessing

Creates processed_train and processed_test .csv files
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.parser import parse
import os

def feature_engineering(df):
    """
    Function to calcualte debt-to-income
    """

    df['dti'] = df['installment'] / (df['annual_inc'] / 12)

def convert_date_to_num(date_col):
    """
    Function to convert date variables to number format
    """

    return date_col.apply(lambda x: (parse(x) - datetime(1900, 1, 1)).days)


def data_preprocessing(df, ohe=False):
    """
    Function for data pre-processing

    The parameter ohe lets the user choose whether to do one-hot-encoding or transform those variables to categoricals

    returns processed DataFrame
    """
    df_new = df.copy()

    feature_engineering(df_new)

    # Columns to drop
    cols_to_drop = ['emp_title', 'zip_code', 'application_type', 'desc', 'funded_amnt', 'funded_amnt_inv', 'grade',
                    'pymnt_plan', 'title', 'issue_d', ]

    # Drop columns
    df_new.drop(labels=cols_to_drop, axis=1, inplace=True)

    # Transform date column to int
    df_new['earliest_cr_line'] = convert_date_to_num(df_new['earliest_cr_line'])

    # Clean employment length feature
    df_new['emp_length'].replace('10+ years', '10 years', inplace=True)
    df_new['emp_length'].replace('< 1 year', '0 years', inplace=True)
    df_new['emp_length'].replace('n/a', np.nan, inplace=True)
    df_new['emp_length'] = df_new['emp_length'].apply(lambda x: x if pd.isnull(x) else np.int8(x.split()[0]))

    # Clean home ownership feature
    df_new['home_ownership'].replace(to_replace=['NONE', 'ANY'], value='OTHER', inplace=True)

    cat_cols = df_new.select_dtypes(include=['object']).columns

    # Performs ohe or transforming to categoricals
    if ohe:
        dummies = pd.get_dummies(df_new[cat_cols])
        df_new = df_new.drop(cat_cols, axis=1).join(dummies)
    else:
        for col in cat_cols:
            df_new[col] = df_new[col].astype('category')

    return df_new


interim_data_path = r'C:\Users\adrian.bergem\Google Drive\Data science\Projects\AI Credit Default\data\interim'
train = pd.read_csv(os.path.join(interim_data_path, 'train.csv'), index_col=0)
test = pd.read_csv(os.path.join(interim_data_path, 'test.csv'), index_col=0)

# Preprocess train data (one with one-hot-encoding and one without)
train_processed = data_preprocessing(train, ohe=False).reset_index(drop=True)
train_processed_ohe = data_preprocessing(train, ohe=True).reset_index(drop=True)

# Preprocess test data (one with one-hot-encoding and one without)
test_processed = data_preprocessing(test, ohe=False).reset_index(drop=True)
test_processed_ohe = data_preprocessing(test, ohe=True).reset_index(drop=True)

# Export preprocessed data
train_processed.to_pickle(r'C:\Users\adrian.bergem\Google Drive\Data science\Projects\AI Credit Default\data\processed\train_processed.pkl')
train_processed_ohe.to_pickle(r'C:\Users\adrian.bergem\Google Drive\Data science\Projects\AI Credit Default\data\processed\train_processed_ohe.pkl')
test_processed.to_pickle(r'C:\Users\adrian.bergem\Google Drive\Data science\Projects\AI Credit Default\data\processed\test_processed.pkl')
test_processed_ohe.to_pickle(r'C:\Users\adrian.bergem\Google Drive\Data science\Projects\AI Credit Default\data\processed\test_processed_ohe.pkl')
