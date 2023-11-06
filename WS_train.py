from flask import Flask
from flask import request
from flask import jsonify
import pickle
import xgboost as xgb
import re

model_file='model_xgboost.bin'

with open(model_file,'rb') as f_in:
    dv,model=pickle.load(f_in)

app = Flask('app')


@app.route('/WS_train', methods=['POST'])
def predict():
    customer = request.get_json()

    features = dv.get_feature_names_out()
    features = [re.sub(r'[\[\]<>]', '', feature) for feature in features]

    x = dv.transform([customer])
    dval = xgb.DMatrix(x, feature_names=features)

    y_pred = model.predict(dval)
    fully_pay = y_pred >= 0.5

    result = {
        'fully pay probability': float(y_pred),
        'fully pay': bool(fully_pay)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

