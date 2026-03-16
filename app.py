from flask import Flask,jsonify,render_template,request
import pandas as pd
import pickle

app = Flask(__name__)


## endpoints
@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/form', methods=['POST'])
def home2():
    return render_template('index.html')
def get_cleaned_data(form_data):
    gestation = float(form_data['gestation'])
    parity=int(form_data['parity'])
    age=float(form_data['age'])
    height=float(form_data['height'])
    weight=float(form_data['weight'])
    smoke=float(form_data['smoke'])

    cleaned_data={"gestation":[gestation],
                  "parity" : [parity],
                  "age" :[age],
                  "height":[height],
                  "weight":[weight],
                  "smoke":[smoke]
                }
    return cleaned_data

@app.route('/predict', methods=['POST'])
def get_prediction():
    #get data from user
    baby_data_form = request.form
    baby_data_cleand = get_cleaned_data(baby_data_form)

    #convert ino data frame
    baby_df = pd.DataFrame(baby_data_cleand)
    #load ML tarined maiodel
    with open("model.pkl", 'rb') as obj:
        model = pickle.load(obj)


    #make prediction on user data
    prediction = model.predict(baby_df)

    prediction = round(prediction[0],2)

    #return responce in a json format

    responce={"prediction" : prediction}

    return render_template('final.html', responce=responce)





if __name__ == "__main__":
    app.run(debug=True)