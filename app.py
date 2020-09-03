# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 21:28:04 2020

@author: This PC
"""


from flask import Flask, render_template, request
import pickle

# Load the Model and Count Vectorize

classifier = pickle.load(open('movie-genre-model.pkl', 'rb'))
cv = pickle.load(open('cv-transform.pkl','rb'))

app=Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        message=request.form['message']
        data=[message]
        vect=cv.transform(data).toarray()
        my_prediction=classifier.predict(vect)
        return render_template('result.html',prediction=my_prediction)    


if __name__ == '__main__':
	app.run(debug=True)    