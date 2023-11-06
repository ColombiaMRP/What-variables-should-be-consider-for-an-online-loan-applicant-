

# ###  Load Packages and data


import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as sfm

from matplotlib.widgets import Slider, Button, RadioButtons
from scipy import interp
from scipy.optimize import fsolve
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold ,train_test_split
from sklearn.metrics import roc_auc_score
from datetime import datetime as dt
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import re
import pickle
import wget

from IPython.display import display


### Parameters

xgb_params = {
    'eta': 0.05, 
    'max_depth': 3,
    'min_child_weight': 30,
    
    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}
ouput_file='model_xgboost.bin'


### Data preparation

data= "https://raw.githubusercontent.com/ColombiaMRP/What-variables-should-be-consider-for-an-online-loan-applicant-/main/Data/Lending_club_cleaned_2.csv"
data= wget.download(data)
df= pd.read_csv(data)

df.int_rate = df.int_rate.str.rstrip('%').astype('float')


numerical = ["annual_inc","int_rate","loan_amnt"]
categorical = ["purpose","verification_status","emp_length","home_ownership","term","grade"]

for i in numerical:
    df[i] = (df[i] - df[i].mean())/df[i].std()
    df[i] = (df[i] - df[i].mean())/df[i].std()
    df[i] = (df[i] - df[i].mean())/df[i].std()

df.loan_status = (df.loan_status == 'Fully Paid').astype(int)

#### 5. Setting up the validation framework


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)



df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


y_train = df_train.loan_status.values
y_val = df_val.loan_status.values
y_test = df_test.loan_status.values

del df_train['loan_status']
del df_val['loan_status']
del df_test['loan_status']


# ### dict-vectorizer


dv = DictVectorizer(sparse=False)

train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)


# ###  Gradient boosting and XGBoost

features = dv.get_feature_names_out()
features = [re.sub(r'[\[\]<>]', '', feature) for feature in features]


dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)


model = xgb.train(xgb_params, dtrain, num_boost_round=175)

y_pred = model.predict(dval)
roc_auc_score(y_val, y_pred)

y_pred = model.predict(dval)
roc_auc_score(y_val, y_pred)


# ### Save the model

with open(ouput_file,'wb') as f_out:
    pickle.dump((dv,model),f_out)

