from flask import Flask,render_template,request,jsonify
import pickle
import numpy as np


with open('lr.pkl','rb') as f:
    lr = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("iris.html")

@app.route('/iris',methods=['POST'])
def predict():
    #SepalLengthCm = float(request.form['sepal length (cm)'])
    SepalWidthCm = float(request.form['sepal width (cm)'])
    #PetalLengthCm = float(request.form['petal length (cm)'])
    PetalWidthCm = float(request.form['petal width (cm)'])
    data = np.array([SepalWidthCm,PetalWidthCm],ndmin=2)
    result = lr.predict(data)
    result = int(result[0][0])
    print(result)
    pred = "Sorry"
    if result==0:
        pred = "Iris-setosa"
    if result==1:
        pred = "Iris-versicolor"
    if result==2:
        pred = "Iris-virginica"
    return render_template("iris.html",prediction = pred)

if __name__ == "__main__":
    app.run(debug = False, host='0.0.0.0',port=8080)