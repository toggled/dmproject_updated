from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor
from sklearn.metrics import mean_squared_error,make_scorer
from sklearn import pipeline,grid_search
import numpy as np
import pandas
df_dir = '../data/'
K_fold = 2
num_train = 74000

def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

RMSE  = make_scorer(fmean_squared_error, greater_is_better=False)


df = pandas.read_csv(df_dir+'my_df_all.csv')
df = df[:num_train].drop(['Unnamed: 0'], axis = 1)
rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
clf2 = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
clf2 = rf
clf = pipeline.Pipeline([('rfr', rf)])
param_grid = {'rfr__n_estimators': [350],  # 300 top
              'rfr__max_depth': [8],  # list(range(7,8,1))
              }
# param_grid = {'rfr__n_estimators':list(range(34,50,1)),
#               'rfr__max_depth':list(range(13,15,1))}
model = grid_search.GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1, cv=5, verbose=0, scoring=RMSE)
errors = []
X_train = df.drop(['product_uid', 'id', 'relevance'], axis=1).values
y_train = df['relevance'].values
model.fit(X_train, y_train)

print("Best parameters found by grid search:")
print(model.best_params_)
print("Best CV score:")
print(model.best_score_)

del X_train, y_train



kf = KFold(df.shape[0], n_folds=K_fold)
for train_index, test_index in kf:
    train_set = df.iloc[train_index]
    test_set = df.iloc[test_index]

    y_train = train_set['relevance'].values
    X_train = train_set.drop(['product_uid', 'id', 'relevance'], axis=1).values
    y_test = train_set['relevance'].values
    X_test = test_set.drop(['product_uid', 'id', 'relevance'], axis=1).values

    clf2.fit(X_train,y_train)

    result = clf2.predict(X_test)
    error = np.sqrt(mean_squared_error(result,y_test))
    errors.extend([error])
print np.mean(errors)