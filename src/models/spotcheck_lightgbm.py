
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import lightgbm as lgb
import os

seed = 42

home_dir = r'C:\Users\adrian.bergem\Google Drive\Data science\Projects\AI Credit Default'
train_processed = pd.read_pickle(os.path.join(home_dir, r'data\processed\train_processed.pkl'))

X_train_processed = train_processed.iloc[:, 1:]
y_train = train_processed.iloc[:, 0]

lgb_model = lgb.LGBMClassifier(seed=seed, boosting_type='gbdt')

# Create parameter grid for LGB
lgb_param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'subsample': [0.8, 1.0]
}

# Create GridSearchVB object for LGBM
GridSearchCV_lgb = GridSearchCV(estimator=lgb_model, param_grid=lgb_param_grid,
                                   scoring='roc_auc', n_jobs=-1, pre_dispatch='2*n_jobs', cv=5,
                                   verbose=1, return_train_score=True)

# Fit to pre-processed training data (without ohe)
GridSearchCV_lgb.fit(X_train_processed, y_train)

print('GridSearchCV for logreg best AUC score: ', GridSearchCV_lgb.best_score_)
print('GridSearchCV for logreg best params: ', GridSearchCV_lgb.best_params_)

joblib.dump(GridSearchCV_lgb, os.path.join(home_dir, 'models/GridSearchCV_lgb.pkl'))
