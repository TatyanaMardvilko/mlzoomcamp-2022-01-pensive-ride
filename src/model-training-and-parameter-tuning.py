#!/usr/bin/env python
# coding: utf-8

# In[154]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction import DictVectorizer


# #  Data preparation

# In[1]:


DATA_DIR = 'd:/maschineLearning/zoomcamp/mlzoomcamp-2022-01-pensive-ride/data'
MODEL_DIR = 'd:/maschineLearning/zoomcamp/mlzoomcamp-2022-01-pensive-ride/models'
SEED = 42 # used to fix random_state


# In[299]:


df_train = pd.read_csv(DATA_DIR + '/processed/train.csv', index_col=0)
df_val = pd.read_csv(DATA_DIR + '/processed/val.csv', index_col=0)
df_test = pd.read_csv(DATA_DIR + '/processed/test.csv', index_col=0)


# In[300]:


data_train = df_train.copy()
data_val = df_val.copy()
data_test = df_test.copy()


# In[18]:


data_test


# In[301]:


data_train.reset_index(drop=True)
data_val.reset_index(drop=True)
data_test.reset_index(drop=True)


# In[302]:


y_train = data_train['average_score'].values
y_val = data_val['average_score'].values
y_test = data_test['average_score'].values


# In[303]:


del data_train['average_score']
del data_val['average_score']
del data_test['average_score']


# In[25]:


data_test


# In[27]:


dv = DictVectorizer(sparse=False)

train_dict = data_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = data_val.to_dict(orient='records')
X_val = dv.transform(val_dict)

test_dict = data_test.to_dict(orient='records')
X_test = dv.transform(test_dict)


# In[104]:


features = dv.get_feature_names_out()
features


# In[170]:


len(features)


# In[30]:


X_test[0]


# # Liner regression

# In[31]:


from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


# In[34]:


Y_train = np.log1p(y_train)
Y_val = np.log1p(y_val)
Y_test = np.log1p(y_test)


# In[36]:


model_line_regr = Ridge(alpha=0, solver="sag", random_state=SEED)


# In[37]:


model_line_regr.fit(X_train, Y_train)


# In[39]:


y_pred_lr = model_line_regr.predict(X_val)


# In[281]:


rmse = np.sqrt(mean_squared_error(Y_val, y_pred_lr))
rmse.round(3)


# In[49]:


alpha = [0, 0.01, 0.1, 1, 10]
solver= {'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'}


# In[85]:


solver_list = []
alpha_list = []
rmse_list = []
for sol in solver:
    for a in alpha:
        model = Ridge(alpha=a, solver=sol, random_state=SEED)
        model.fit(X_train, Y_train)
        y_pred_lr = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(Y_val, y_pred_lr))
        solver_list.append(sol)
        alpha_list.append(a)
        rmse_list.append(rmse)
        print(f'solver = {sol}, alpha = {a}, rmse = {rmse.round(7)}')


# In[87]:


df_tuning_param = pd.DataFrame()
df_tuning_param['solver'] = solver_list
df_tuning_param['alpha'] = alpha_list
df_tuning_param['rmse'] = rmse_list
df_tuning_param = df_tuning_param.sort_values(by='rmse', ascending=True)
df_tuning_param = df_tuning_param.reset_index(drop=True)
df_tuning_param


# In[88]:


model_line_regr = Ridge(alpha=df_tuning_param['alpha'][0], solver=df_tuning_param['solver'][0], random_state=SEED)
model_line_regr.fit(X_train, Y_train)


# # Random Forest

# In[125]:


from sklearn.ensemble import RandomForestRegressor


# In[90]:


rf = RandomForestRegressor(n_estimators=10, random_state=SEED, n_jobs=-10)
rf.fit(X_train, y_train)


# In[91]:


y_pred = rf.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_pred, y_val))
rmse 


# In[121]:


scores = []

for depth in [10, 15, 20, 25]:
    for s in [1, 3, 5, 10, 50]:
        for n in range(10, 201, 10):
            rf = RandomForestRegressor(n_estimators=n, random_state = SEED, min_samples_leaf=s, max_depth=depth)
            rf.fit(X_train, y_train)

            y_pred = rf.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_pred, y_val))

            scores.append((n, depth, s, rmse))


# In[122]:


columns = ['n_estimators','max_depth', 'min_samples_leaf', 'rmse']
df_scores = pd.DataFrame(scores, columns=columns)
df_scores = df_scores.sort_values(by='rmse', ascending=True).reset_index(drop=True)
df_scores


# In[123]:


rf = RandomForestRegressor(n_estimators=df_scores['n_estimators'][0], random_state = SEED, min_samples_leaf=df_scores['min_samples_leaf'][0], max_depth=df_scores['max_depth'][0])
rf.fit(X_train, y_train)


# # Xgboost

# Tuning the following parameters:
# 
# eta
# 
# max_depth
# 
# min_child_weight

# In[107]:


import xgboost as xgb


# In[108]:


dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)


# In[109]:


wachlist = [(dtrain, 'train'), (dval, 'val')]


# In[110]:


def parse_xgb_output(output):
    results = []
    
    for line in output.stdout.strip().split('\n'):
        it_line, train_line, val_line = line.split('\t')
        
        it = int(it_line.strip('[]'))
        train = float(train_line.split(':')[1])
        val = float(val_line.split(':')[1])
        
        results.append((it, train, val))
        
    columns = ['num_iter', 'train_rmse', 'val_rmse']
    df_results = pd.DataFrame(results, columns=columns)
    
    return df_results


# ## Tuning eta

# In[206]:


scores = {}


# In[218]:


get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': 1.0, \n    'max_depth': 6,\n    'min_child_weight': 1,\n    \n    'objective': 'reg:squarederror',\n    'nthread': 8,\n    \n    'seed': 1,\n    'verbosity': 1,\n}\n\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200, verbose_eval=5, evals=wachlist)")


# In[219]:


key = 'eta=%s' % (xgb_params['eta'])
scores[key] = parse_xgb_output(output)
key


# In[221]:


for eta, df_score in scores.items():
    plt.plot(df_score.num_iter, df_score.val_rmse, label=eta)

plt.ylim(10, 30)
plt.legend()


# eta = 0.1

# ## Tuning max_depth

# In[222]:


scores = {}


# In[235]:


get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': 0.1, \n    'max_depth': 17,\n    'min_child_weight': 1,\n    \n    'objective': 'reg:squarederror',\n    'nthread': 8,\n    \n    'seed': 1,\n    'verbosity': 1,\n}\n\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200, verbose_eval=5, evals=wachlist)")


# In[236]:


key = 'max_depth=%s' % (xgb_params['max_depth'])
scores[key] = parse_xgb_output(output)
key


# In[237]:


for max_depth, df_score in scores.items():
    plt.plot(df_score.num_iter, df_score.val_rmse, label=max_depth)

plt.ylim(11, 15)
plt.legend()


# max_depth = 3

# ## Tuning min_child_weight

# In[259]:


scores = {}


# In[272]:


get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': 0.1, \n    'max_depth': 3,\n    'min_child_weight': 15,\n    \n    'objective': 'reg:squarederror',\n    'nthread': 8,\n    \n    'seed': 1,\n    'verbosity': 1,\n}\n\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200, verbose_eval=5, evals=wachlist)")


# In[273]:


key = 'min_child_weight=%s' % (xgb_params['min_child_weight'])
scores[key] = parse_xgb_output(output)
key


# In[275]:


for min_child_weight, df_score in scores.items():
    plt.plot(df_score.num_iter, df_score.val_rmse, label=min_child_weight)

plt.ylim(12, 13)
plt.legend()


# min_child_weight = 15

# In[277]:


xgb_params = {
    'eta': 0.1, 
    'max_depth': 3,
    'min_child_weight': 15,
    
    'objective': 'reg:squarederror',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=75)
model


# # Selecting the final model

# ## Linear regression

# In[279]:


model_line_regr = Ridge(alpha=df_tuning_param['alpha'][0], solver=df_tuning_param['solver'][0], random_state=SEED)
model_line_regr.fit(X_train, Y_train)


# In[284]:


y_pred_lr_log = model_line_regr.predict(X_val)


# In[285]:


y_pred_lr = np.expm1(y_pred_lr_log)


# In[287]:


rmse_lr = np.sqrt(mean_squared_error(y_pred_lr, y_val))
rmse_lr


# ## Random Forest

# In[288]:


rf = RandomForestRegressor(n_estimators=df_scores['n_estimators'][0], 
                           random_state = SEED, min_samples_leaf=df_scores['min_samples_leaf'][0], 
                           max_depth=df_scores['max_depth'][0])


# In[290]:


rf.fit(X_val, y_val)


# In[297]:


y_pred_rf =  rf.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_pred_rf, y_val))
rmse 


# ## Xgboosts

# In[292]:


xgb_params = {
    'eta': 0.1, 
    'max_depth': 3,
    'min_child_weight': 15,
    
    'objective': 'reg:squarederror',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
}

model_xgb = xgb.train(xgb_params, dtrain, num_boost_round=75)


# In[295]:


y_pred_xgb = model_xgb.predict(dval)


# In[298]:


y_pred_xgb
rmse = np.sqrt(mean_squared_error(y_pred_xgb, y_val))
rmse 


# ## The best model

# the best model is Random Forest

# In[306]:


df_full_train = pd.concat([df_train, df_val]).reset_index(drop = True)


# In[307]:


df_full_train


# In[308]:


y_full_train = df_full_train['average_score'].values


# In[315]:


len(y_full_train)


# In[309]:


del df_full_train['average_score']


# In[310]:


df_full_train


# In[316]:


df_full_train_dict = df_full_train.to_dict(orient='records')
X_full_train = dv.fit_transform(df_full_train_dict)


# In[317]:


rf = RandomForestRegressor(n_estimators=df_scores['n_estimators'][0], 
                           random_state = SEED, min_samples_leaf=df_scores['min_samples_leaf'][0], 
                           max_depth=df_scores['max_depth'][0])


# In[318]:


rf.fit(X_full_train, y_full_train)


# In[319]:


y_pred_rf =  rf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_pred_rf, y_test))
rmse 


# # Save model

# In[320]:


import pickle


# In[323]:


output_file = 'RandomForestModel.bin'


# In[324]:


with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)


# In[ ]:




