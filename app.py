from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
model = pickle.load(open('winequality.pkl', 'rb'))

cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'density',
       'pH' ,'sulphates', 'alcohol']

minMax = [[4,15],[0,2],[0,1],[0,15],[0,1],[1,72],[0,1],[2,4],[0,2],[8,15]]

@app.route('/', methods=['GET'])
@app.route('/home', methods=['GET'])
def home():
    global cols,cols2
    if request.method == 'GET':
        print(cols,"1000")

        return render_template('base.html', columns=cols,value='none')
        # global cols,cols2
    if request.method == 'POST':
        inputList = []
        print(cols,"1")
        
        for i in cols:
            inputList.append(request.form[i])
        input_arr = np.array(inputList).reshape(1,-1)
        prediction = model.predict(input_arr)
        print(cols,"2")
    return render_template('base.html', value=prediction,columns=cols)

@app.route('/predict', methods=['POST', 'GET'])
def pred():
    global cols,cols2
    if request.method == 'POST':
        inputList = []
        for i in cols:
            inputList.append(request.form[i])
        input_arr = np.array(inputList).reshape(1,-1)
        prediction = model.predict(input_arr)
        prediction = prediction[0]
        if prediction == 0:
            pred = 'Very Bad'
        elif prediction > 0 and prediction < 4:
            pred = 'Bad'
        elif prediction > 3 and prediction < 5:
            pred = 'Average'
        elif prediction > 6 and prediction < 9:
            pred = 'Good'
        else:
            pred = 'Best'
    return render_template('result.html', result=pred, predNum=prediction, inputs=inputList,columns=cols)

if __name__ == '__main__':
    app.run(debug=True)