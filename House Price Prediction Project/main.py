# flask, scikit-learn, pandas, pickle-mixin
import pandas as pd
from flask import Flask, render_template, request
import pickle



app=Flask(__name__)
data = pd.read_csv('house_price_data.csv')
pipe = pickle.load(open("networkmodel.pkl", 'rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    bed = request.form.get('bed')
    bath = request.form.get('bath')
    sqftliv = request.form.get('sqftliv')
    sqftlot = request.form.get('sqftlot')
    floors = request.form.get('floors')
    waterfront = request.form.get('waterfront')
    condition = request.form.get('condition')
    year = request.form.get('year')

    print(bed, bath, sqftliv, sqftlot, floors, waterfront, condition, year)
    input = pd.DataFrame([[bed,bath,sqftliv,sqftlot,floors,waterfront,condition,year]],columns=['bed','bath','sqftliv','sqftlot','floors','waterfront','condition','year'])
    prediction = pipe.predict(input)[0]

    
    return str(prediction)


if __name__=="__main__":
    app.run(debug=True)    
