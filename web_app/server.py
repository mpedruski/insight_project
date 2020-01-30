from flask import Flask, render_template, request
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import OneHotEncoder
from joblib import load
import numpy as np
import pandas as pd
import re
from pathlib import Path
import datetime
import logging

def predict_title(author, publisher, country, doc_type, year_offset, page_count, language):
    title_data = np.array([author_labels[author],publisher_labels[publisher],
        country_labels[country],type_labels[doc_type], year_offset, page_count,
        language_labels[language]]).astype(int)
    prediction = regr_1.predict(title_data.reshape(1,-1))
    return prediction

def publisher_formatting_singleton(publisher_name):
    '''[str] -> [int]
    Accepts a publisher names, removes common words, and then uses REGEX
    to remove punctuation to create uniform author names'''
    stop_words = ['publishers','books','book','co.','ltée','canada', ' ']
    logging.debug("Incoming text to reformat function: {}".format(publisher_name))
    if isinstance(publisher_name, str):
        publisher_name=publisher_name.lower()
        for j in stop_words:
            publisher_name = publisher_name.replace(j,'')
        pattern = re.compile(r'[a-zA-ZÀ-ÿ].*[a-zA-ZÀ-ÿ]')
        pattern_output = pattern.search(str(publisher_name))
        if pattern_output is None:
            return None
        else:
            return pattern_output.group()
    else:
        return None

### Determine paths to datasets and load into pandas dataframe
# test_folder = Path("../data/test")
processed_folder = Path("../data/processed/")
test_folder = Path("../data/test/")
author_dictionary = processed_folder / 'author_dictionary_clusters.joblib'
publisher_dictionary = processed_folder / 'publisher_dictionary_clusters.joblib'
country_dictionary = processed_folder / 'country_dictionary_clusters.joblib'
type_dictionary = processed_folder / 'type_dictionary_categories.joblib'
language_dictionary = processed_folder / 'language_dictionary_categories.joblib'

model_data = processed_folder / 'decision_tree_parameters.joblib'

### Load in label data to categorize new title:
author_labels = load(author_dictionary)
publisher_labels = load(publisher_dictionary)
country_labels = load(country_dictionary)
type_labels = load(type_dictionary)
language_labels = load(language_dictionary)
print(language_labels)
### Load saved model
regr_1 = load(model_data)

### Create the application object
app = Flask(__name__)

@app.route('/',methods=["GET","POST"])
def home_page():
    return render_template('index.html')

@app.route('/output')
def recommendation_output():

       ### Pull input from form
       info_to_pull = ['title','author','publisher','country','type_info',
        'year','page_count','language']
       inputs = []
       for i in info_to_pull:
           inputs.append(request.args.get(i))
       ### Reformat publisher following rules in data cleaning
       publisher_input = publisher_formatting_singleton(inputs[2])
       year_offset = 2020-int(inputs[5])
       logging.debug("Output text from reformat function: {}".format(inputs[2]))


       ### If any input is empty encourage user input to all fields
       ### If all inputs are filled run model and return results
       if "" in inputs:
           return render_template("index.html",
                                  my_input = "",
                                  my_form_result="Empty")
       else:
           some_output = "Predicted demand for {} is {} copies:".format(inputs[0],
            predict_title(inputs[1],publisher_input,inputs[3],inputs[4],year_offset,int(inputs[6]),inputs[7])[0])
           return render_template("index.html",
                              my_input="Results for: {}, {}, {}, and {}".format(inputs[1],
                                inputs[2],inputs[3],inputs[4], inputs[5],inputs[6],inputs[7]),
                              title_output = inputs[0],
                              my_output=some_output,
                              my_form_result="NotEmpty")


### Start the server
if __name__ == "__main__":
    app.run(debug=True)
