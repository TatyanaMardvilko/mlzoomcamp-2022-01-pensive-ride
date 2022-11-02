import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle

DATA_DIR = '../data'
MODEL_DIR = '../models'
SEED = 42  # used to fix random_state
output_file = MODEL_DIR + '/RandomForestModel.bin'

df_train = pd.read_csv(DATA_DIR + '/processed/train.csv', index_col=0)
df_val = pd.read_csv(DATA_DIR + '/processed/val.csv', index_col=0)
df_test = pd.read_csv(DATA_DIR + '/processed/test.csv', index_col=0)

data_train = df_train.copy()
data_val = df_val.copy()
data_test = df_test.copy()

data_train.reset_index(drop=True)
data_val.reset_index(drop=True)
data_test.reset_index(drop=True)

y_train = data_train['average_score'].values
y_val = data_val['average_score'].values
y_test = data_test['average_score'].values

del data_train['average_score']
del data_val['average_score']
del data_test['average_score']

dv = DictVectorizer(sparse=False)

train_dict = data_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = data_val.to_dict(orient='records')
X_val = dv.transform(val_dict)

test_dict = data_test.to_dict(orient='records')
X_test = dv.transform(test_dict)

features = dv.get_feature_names_out()

# training the final model

print('training the final model')
df_full_train = pd.concat([df_train, df_val]).reset_index(drop=True)

y_full_train = df_full_train['average_score'].values

del df_full_train['average_score']

df_full_train_dict = df_full_train.to_dict(orient='records')
X_full_train = dv.fit_transform(df_full_train_dict)

model = RandomForestRegressor(n_estimators=190,
                              random_state=SEED, min_samples_leaf=10,
                              max_depth=15)

model.fit(X_full_train, y_full_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_pred, y_test))

print(f'rmse={rmse}')

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')
