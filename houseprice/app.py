 #Importing essential libraries
from flask import Flask, render_template, request
import joblib
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'housepriceprediction.pkl'
#model = pickle.load(open(filename, 'rb'))
model = joblib.load(filename)

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        area = int(request.form['area'])
        bedrooms = int(request.form.get['bedrooms'])
        bathrooms = int(request.form.get['bathrooms'])
        stories = int(request.form['stories'])
        mainroad = int(request.form['mainroad'])
        guestroom = int(request.form.get['guestroom'])
        basement = int(request.form['basement'])
        hotwaterheating = int(request.form['hotwaterheating'])
        airconditioning = int(request.form.get['airconditioning'])
        parking = int(request.form['parking'])
        prefarea = int(request.form.get['prefarea'])
        furnishingstatus = int(request.form['furnishingstatus'])

        
        data = np.array([[area,bedrooms,bathrooms,stories,mainroad,guestroom,basement,hotwaterheating,airconditioning,parking,prefarea,furnishingstatus]])
        my_prediction = model.predict(data)
        
        return render_template('result.html', prediction=my_prediction)
        
        

if __name__ == '__main__':
	app.run(port=500)

