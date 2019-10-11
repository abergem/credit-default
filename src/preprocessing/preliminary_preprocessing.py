"""
Python module to do preliminary preprocessing

Creates train and test .csv files
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os

seed = 42
raw_data_dir = r'C:\Users\adrian.bergem\Google Drive\Data science\Projects\AI Credit Default\data\raw'
# Load in data
borrower = pd.read_csv(os.path.join(raw_data_dir, 'Borrower Information.csv'))
loan = pd.read_csv(os.path.join(raw_data_dir, 'Loan Classification Information.csv'))

df = pd.merge(borrower, loan, on='member_id')

# Limit the dataset into only terminated ones
df = df[df['loan_status'].isin(['Charged Off', 'Fully Paid'])]

# Transform the loan status variable to 1s and 0s
df['target'] = df['loan_status'].apply(lambda x: 1 if x == 'Charged Off' else 0)
# Drop the loan status feature as we will no longer need it
df.drop(labels='loan_status', axis=1, inplace=True)

# Remove id columns
df.drop(labels=['Unnamed: 0_x', 'Unnamed: 0_y', 'member_id', 'id'], axis=1, inplace=True)

# Remove features with more than 90% NaNs
nans = pd.DataFrame(index=df.columns, columns=['percentage NaNs'])

for feature in df.columns:
    nans.loc[feature] = df[feature].isnull().sum()/(len(df))

cols_to_remove = nans[nans['percentage NaNs'] > 0.90].index.tolist()
df.drop(labels=cols_to_remove, axis=1, inplace=True)

# Omit old data
df = df[df['issue_d'] > '01-01-2010']

# Split data into target (y) and features (X)
X = df.drop(labels='target', axis=1)
y = df['target']

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

# Reset indexes in train and test data sets
X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

# Recombine features and target in train/test data
train = pd.concat([y_train, X_train], axis=1).reset_index(drop=True)
test = pd.concat([y_test, X_test], axis=1).reset_index(drop=True)

# Export train and test to .csv
train.to_csv(r'C:\Users\adrian.bergem\Google Drive\Data science\Projects\AI Credit Default\data\interim\train.csv')
test.to_csv(r'C:\Users\adrian.bergem\Google Drive\Data science\Projects\AI Credit Default\data\interim\test.csv')
