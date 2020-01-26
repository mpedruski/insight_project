from flask import Flask, render_template, request
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import OneHotEncoder
from joblib import load
import numpy as np
import pandas as pd
from pathlib import Path
import datetime
import logging

def predict_title(author, publisher, country, doc_type):
    title_data = np.array([author_labels[author],publisher_labels[publisher],
        country_labels[country],type_labels[doc_type]]).astype(int)
    onehotlabels_new = enc.transform(title_data.reshape(1,-1))
    prediction = clf.predict(onehotlabels_new)
    return prediction

### Determine paths to datasets and load into pandas dataframe
# test_folder = Path("../data/test")
processed_folder = Path("../data/processed/")
test_folder = Path("../data/test/")
variable_data = test_folder / 'cluster_labels_for_each_variable.csv'
model_data = processed_folder / 'SGD_model_parameters.joblib'
encoding_data = processed_folder / 'OHE_parameters.joblib'

df = pd.read_csv(variable_data)
# df = pd.read_csv(variable_data,dtype={'ID':'int','Author_names':'str','Author_labels':'int',
#     'Publisher_names':'str','Publisher_labels':'int','Country_names':'int',
#     'Country_labels':'int','Type_names':'str','Type_labels':'int'})
### Load in label data to categorize new title:

author_labels = df.set_index('Author_names')['Author_labels'].to_dict()
publisher_labels = df.set_index('Publisher_names')['Publisher_labels'].to_dict()
country_labels = df.set_index('Country_names')['Country_labels'].to_dict()
type_labels = df.set_index('Type_names')['Type_labels'].to_dict()
### Load saved model
clf = load(model_data)
enc = load(encoding_data)

### Create the application object
app = Flask(__name__)

@app.route('/',methods=["GET","POST"])
def home_page():
    return render_template('index.html')

@app.route('/output')
def recommendation_output():

       ### Pull input from form
       title_input =request.args.get('title')
       author_input =request.args.get('author')
       publisher_input =request.args.get('publisher')
       country_input =request.args.get('country')
       type_input =request.args.get('type_info')

       ### If any input is empty encourage user input to all fields
       ### If all inputs are filled run model and return results
       if author_input =="" or publisher_input == "" or country_input=="" or type_input=="":
           return render_template("index.html",
                                  my_input = "",
                                  my_form_result="Empty")
       else:
           some_output = "Predicted demand for {} is {} copies:".format(title_input, predict_title(author_input,publisher_input,country_input,type_input)[0])
           return render_template("index.html",
                              my_input="Results for: {}, {}, {}, and {}".format(author_input,publisher_input,country_input,type_input),
                              title_output = title_input,
                              my_output=some_output,
                              my_form_result="NotEmpty")


### Start the server
if __name__ == "__main__":
    app.run(debug=True)
