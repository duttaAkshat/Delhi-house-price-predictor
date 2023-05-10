import os
import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'data', 'new_data.csv')

current_dir = os.path.dirname(os.path.abspath(__file__))
pkl_file_path = os.path.join(current_dir, 'data', 'lrModel.pkl')

if not os.path.isfile(pkl_file_path):
    raise FileNotFoundError(f"File not found: {pkl_file_path}")

with open(pkl_file_path, 'rb') as file:
    pipe = pickle.load(file)

if not os.path.isfile(data_path):
    raise FileNotFoundError(f"File not found: {data_path}")

data = pd.read_csv(data_path)
data = data.drop('Unnamed: 0', axis=1)

@app.route('/')
def index():
    print(data.head())
    location = sorted(data['location'].unique())
    return render_template('index.html', location=location)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('sqft')

    print(location, bhk, bath, sqft)
    input = pd.DataFrame([[location,sqft,bath,bhk]], columns=['location','area','Bathrooms','Bedrooms'])
    prediction = pipe.predict(input)[0]
    return str(np.round(prediction, 2))


if __name__ == '__main__':
    app.run()

