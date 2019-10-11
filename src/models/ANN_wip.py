# -*- coding: utf-8 -*-
"""
Created on Fri May  3 13:34:27 2019

@author: adrian.bergem
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

seed = 42

home_dir = r'C:\Users\adrian.bergem\Google Drive\Data science\Projects\credit-default'
train = pd.read_pickle(os.path.join(home_dir, r'data\processed\train_processed_ohe.pkl'))

X_train = train.iloc[:, 1:]
y_train = train.iloc[:, 0]

# Create pipeline with both imputation and standard scaling
impute_scale_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Fit to pre-processed training data (with ohe)
X_train_scaled = impute_scale_pipeline.fit_transform(X_train)

X_train, X_valid, y_train, y_valid = train_test_split(X_train_scaled, y_train, test_size=0.33, random_state=seed)

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=16, kernel_initializer='uniform', activation='relu', input_dim=X_train.shape[1]))
classifier.add(Dropout(rate=0.1))

# Adding the 2nd hidden layer
classifier.add(Dense(units=32, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(rate=0.1))

# Adding the 3rd hidden layer
classifier.add(Dense(units=64, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(rate=0.1))

# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, epochs=2, batch_size=32)

# Predicting the Test set results
y_pred_proba = classifier.predict(X_valid)
y_pred = (y_pred_proba > 0.5)
cm = confusion_matrix(y_valid, y_pred)

print(cm)
print(roc_auc_score(y_valid, y_pred))