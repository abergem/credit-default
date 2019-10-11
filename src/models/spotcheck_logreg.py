
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
import os

seed = 42

home_dir = r'C:\Users\adrian.bergem\Google Drive\Data science\Projects\AI Credit Default'
train_processed = pd.read_pickle(os.path.join(home_dir, r'data\processed\train_processed_ohe.pkl'))

X_train_processed = train_processed.iloc[:, 1:]
y_train = train_processed.iloc[:, 0]

# Create pipeline with both imputation and standard scaling
logreg_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', SGDClassifier(loss='log', max_iter=1000, tol=1e-3, random_state=seed, warm_start=True))
])

# Create parameters grid
logreg_param_grid = {
    'model__alpha': [10**-5, 10**-2, 10**1],
    'model__penalty': ['l1', 'l2']}

# Create GridSearchCV object for the logistic regression
GridSearchCV_logreg = GridSearchCV(estimator=logreg_pipeline, param_grid=logreg_param_grid,
                                   scoring='roc_auc', n_jobs=-1, pre_dispatch='2*n_jobs', cv=5,
                                   verbose=1, return_train_score=True)

# Fit to pre-processed training data (with ohe)
GridSearchCV_logreg.fit(X_train_processed, y_train)

print('GridSearchCV for logreg best AUC score: ', GridSearchCV_logreg.best_score_)
print('GridSearchCV for logreg best params: ', GridSearchCV_logreg.best_params_)

joblib.dump(GridSearchCV_logreg, os.path.join(home_dir, 'models/GridSearchCV_logreg.pkl'))
