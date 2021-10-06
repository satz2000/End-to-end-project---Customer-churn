import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route("/")
def home_page():
    return render_template('home.html')

@app.route("/", methods=['POST'])
def predict():

    """ Selected feature are Dependents, tenure, OnlineSecurity,
        OnlineBackup, DeviceProtection, TechSupport, Contract,
        PaperlessBilling, MonthlyCharges, TotalCharges """

    Dependents = request.form['Dependents']
    tenure = float(request.form['tenure'])
    OnlineSecurity = request.form['OnlineSecurity']
    OnlineBackup = request.form['OnlineBackup']
    DeviceProtection = request.form['DeviceProtection']
    TechSupport = request.form['TechSupport']
    Contract = request.form['Contract']
    PaperlessBilling = request.form['PaperlessBilling']
    MonthlyCharges = float(request.form['MonthlyCharges'])
    TotalCharges = float(request.form['TotalCharges'])

    model = pickle.load(open('Model.sav', 'rb'))
    data = [[Dependents, tenure, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, Contract, PaperlessBilling, MonthlyCharges, TotalCharges]]
    df = pd.DataFrame(data, columns=['Dependents', 'tenure', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'Contract',
        'PaperlessBilling', 'MonthlyCharges', 'TotalCharges'])

    categorical_feature = {feature for feature in df.columns if df[feature].dtypes == 'O'}

    encoder = LabelEncoder()
    for feature in categorical_feature:
        df[feature] = encoder.fit_transform(df[feature])

    single = model.predict(df)
    probability = model.predict_proba(df)[:, 1]
    probability = probability*100

    if single == 1:
        op1 = "This Customer is likely to be Churned!"
        op2 = f"Confidence level is {np.round(probability[0], 2)}"
    else:
        op1 = "This Customer is likely to be Continue!"
        op2 = f"Confidence level is {np.round(probability[0], 2)}"

    return render_template("home.html", op1=op1, op2=op2,
                           Dependents=request.form['Dependents'],
                           tenure=request.form['tenure'],
                           OnlineSecurity=request.form['OnlineSecurity'],
                           OnlineBackup=request.form['OnlineBackup'],
                           DeviceProtection=request.form['DeviceProtection'],
                           TechSupport=request.form['TechSupport'],
                           Contract=request.form['Contract'],
                           PaperlessBilling=request.form['PaperlessBilling'],
                           MonthlyCharges=request.form['MonthlyCharges'],
                           TotalCharges=request.form['TotalCharges'])


if __name__ == '__main__':
    app.run()