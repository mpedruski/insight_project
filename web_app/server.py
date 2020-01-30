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
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s')


def predict_title(author, publisher, country, doc_type, year_offset, page_count, language):
    title_data = np.array([author_labels.get(author,0), publisher_labels.get(publisher,0),
        country_labels[country],type_labels[doc_type], year_offset, page_count,
        language_labels[language]]).astype(int)
    prediction = regr_1.predict(title_data.reshape(1,-1))
    print(title_data)
    return prediction

def publisher_formatting_singleton(publisher_name):
    '''[str] -> [int]
    Accepts a publisher names, removes common words, and then uses REGEX
    to remove punctuation to create uniform author names'''
    stop_words = ['publishers','books','book','co.','ltée','canada', ' ']
    logging.debug("Incoming text to reformat function: {}".format(publisher_name))
    if isinstance(publisher_name, str):
        print(publisher_name)
        publisher_name=publisher_name.lower()
        for j in stop_words:
            publisher_name = publisher_name.replace(j,'')
            print(publisher_name)
        pattern = re.compile(r'[a-zA-ZÀ-ÿ].*[a-zA-ZÀ-ÿ]')
        pattern_output = pattern.search(str(publisher_name))
        print(pattern_output)
        if pattern_output is None:
            logging.debug('Pattern found nothing')
            return None
        else:
            return pattern_output.group()
    else:
        logging.debug("It's not a string")
        return None


### Determine paths to datasets and load into pandas dataframe
# test_folder = Path("../data/test")
processed_folder = Path("../data/processed/")
test_folder = Path("../data/test/")
author_dictionary = processed_folder / 'author_dictionary.joblib'
publisher_dictionary = processed_folder / 'publisher_dictionary.joblib'
country_dictionary = processed_folder / 'country_dictionary.joblib'
type_dictionary = processed_folder / 'type_dictionary_categories.joblib'
language_dictionary = processed_folder / 'language_dictionary_categories.joblib'
# coded_variables_file = processed_folder / 'coded_variables.joblib'

model_data = processed_folder / 'decision_tree_parameters.joblib'

### Load in label data to categorize new title:
author_labels = load(author_dictionary)
publisher_labels = load(publisher_dictionary)
country_labels = load(country_dictionary)
type_labels = load(type_dictionary)
language_labels = load(language_dictionary)
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
       # info_to_pull = ['title','author','publisher','country',
       #  'year','page_count','language','type_info']
       # inputs = []
       title_input = request.args.get('title')
       author_input = request.args.get('author')
       publisher_input_initial = request.args.get('publisher')
       country_input = request.args.get('country')
       page_count_input = request.args.get('page_count')
       language_input = request.args.get('language')
       document_type_input = request.args.get('type_info')

       inputs = [title_input,author_input,publisher_input_initial,country_input
        ,page_count_input,language_input,document_type_input]

       ### Reformat publisher following rules in data cleaning
       publisher_input = publisher_formatting_singleton(publisher_input_initial)
       ### Preprocessing author and publisher to prevent key errors

       ### If any input is empty encourage user input to all fields
       ### If all inputs are filled run model and return results
       if "" in inputs:
           return render_template("index.html",
                                  my_input = "",
                                  my_form_result="Empty")
       else:
           yr1_output = "Predicted demand for {} after 1 year is {} copies:".format(title_input,
            predict_title(author_input,publisher_input,country_input,
                document_type_input,1,int(page_count_input),language_input)[0])
           yr3_output = "Predicted demand for {} after 3 years is {} copies:".format(title_input,
            predict_title(author_input,publisher_input,country_input,
                document_type_input,3,int(page_count_input),language_input)[0])
           yr10_output = "Predicted demand for {} after 10 years is {} copies:".format(title_input,
           predict_title(author_input,publisher_input,country_input,
               document_type_input,10,int(page_count_input),language_input)[0])
           return render_template("index.html",
                              my_input="Results for: Author: {}, Publisher: {},Country: {}, Document type: {}, Page count: {} and language: {}".format(author_input, publisher_input,country_input,document_type_input
                                ,page_count_input,language_input),
                              title_output = title_input,
                              output_1=yr1_output,
                              output_2=yr3_output,
                              output_3=yr10_output,
                              my_form_result="NotEmpty")


### Start the server
if __name__ == "__main__":
    app.run(debug=True)
