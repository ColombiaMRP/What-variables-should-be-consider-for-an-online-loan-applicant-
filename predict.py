import pickle
import xgboost as xgb
import re

model_file='model_xgboost.bin'

with open(model_file,'rb') as f_in:
    dv,model=pickle.load(f_in)


customer ={
    'annual_inc':-0.4,
    'verification_status':'Verified',
    'emp_length':'6 years',
    'home_ownership':'MORTGAGE',
    'int_rate':1,
    'loan_amnt':1.5,
    'purpose':'home_improvement',
    'term':'36 months',
    'grade':'C'
}

features = dv.get_feature_names_out()
features = [re.sub(r'[\[\]<>]', '', feature) for feature in features]

x = dv.transform([customer])
dval = xgb.DMatrix(x, feature_names=features)


y_pred = model.predict(dval)
print('default probability',y_pred[0])