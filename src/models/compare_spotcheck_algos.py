
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.externals import joblib
import re
import os

# Use regex to match the keys of interst in the GridSearchCV object
regex = re.compile('split._test_score')

home_dir = r'C:\Users\adrian.bergem\Google Drive\Data science\Projects\AI Credit Default'

GridSearchCV_logreg = joblib.load(os.path.join(home_dir, r'models\GridSearchCV_logreg.pkl'))
GridSearchCV_lgb = joblib.load(os.path.join(home_dir, r'models\GridSearchCV_lgb.pkl'))

models = [('SGD Logistic Regression', GridSearchCV_logreg),
          ('LightGBM', GridSearchCV_lgb)]

# Create empty DataFrame for results
results = pd.DataFrame()

# Iterate through each (name, GridSearchCV object) tuple in in the models list
# and extract the results in the cv with the best hyperparameters
for name, grid_cv in models:
    ix = grid_cv.best_index_
    for key in grid_cv.cv_results_:
        if re.match(regex, key):
            results.loc[key[5], name] = grid_cv.cv_results_[key][ix]

# Unstack the dataset for plotting
results = results.unstack().reset_index(level=0)
results.columns = ['Model', 'cv results']

# Plot the results and save figure
fig, ax = plt.subplots(figsize=(8,5))
sns.swarmplot(x='Model', y='cv results', data=results, ax=ax)
ax.set_ylabel('AUC')
fig.savefig(os.path.join(home_dir, r'models\algo_comparison.jpg'))
