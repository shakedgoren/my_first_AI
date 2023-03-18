import json
from flask import Flask, request
from flask_cors import CORS
from sklearn.tree import DecisionTreeClassifier
import joblib
import csv
import pandas as pd

app = Flask(__name__)
CORS(app)

@app.route("/predictions",methods=['GET','POST'])
def predicti():
    predictions=[]
    if request.method == 'POST':
        myData =json.loads(request.data)
        age=myData['age']
        gender=myData['gender']
        model=joblib.load( 'our_pridction.joblib')
        predictions= model.predict([[age,gender]])
        msg=int(predictions[0])
    return json.dumps(msg)

@app.route('/add', methods=['GET', 'POST'])
def add_children():
    if request.method == 'POST':
        myData =json.loads(request.data)
        age=myData['age']
        gender=myData['gender']
        children=myData['children']
        f = open('childrens.csv', 'a',newline='')
        writer = csv.writer(f)
        data = [age, gender, children]
        writer.writerow(data)
        f.close()
    return ""

@app.route("/learn",methods=['GET'])
def learn():
    childrens_dt=pd.read_csv('childrens.csv')
    X=childrens_dt.drop(columns=['children']) 
    Y=childrens_dt['children'] 
    model = DecisionTreeClassifier()
    model.fit(X,Y) 
    joblib.dump(model, 'our_pridction.joblib') 
    return ""

if __name__ == '__main__':
    app.run(debug=True)