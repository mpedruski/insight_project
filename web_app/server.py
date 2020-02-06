from flask import Flask, render_template, request, Markup
from joblib import load
import numpy as np
import pandas as pd
import re
from pathlib import Path
import datetime
import logging
logging.basicConfig(level=logging.CRITICAL,format='%(asctime)s - %(levelname)s - %(message)s')


def predict_title(author, publisher, country, doc_type, year_offset, language):
    ''' [str, str, str, str, int, str] -> int
    Accepts an author key, a publisher key, a country key, a document-type key,
    a year from the present, and a language key, finds the corresponding value
    to each key (as well as the year from the present as an int) and passes
    these to the loaded regression model to return predicted output'''
    title_data = np.array([dictionaries[0].get(author,0), dictionaries[1].get(publisher,0),
        dictionaries[2].get(country,0),dictionaries[3][doc_type], year_offset,
        dictionaries[4][language]]).astype(int)
    prediction = regr_1.predict(title_data.reshape(1,-1)).astype(int)
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
    language_input = request.args.get('language')
    document_type_input = request.args.get('type_info')

    inputs = [title_input,author_input,publisher_input_initial,country_input
    ,language_input,document_type_input]

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
        yr1_pred = predict_title(author_input,publisher_input,country_input,
        document_type_input,1,language_input)[0]
        if yr1_pred == 1:
            yr1_output = Markup("Books with the characteristics of <i>{}</i> are predicted to have a demand for 1 copy after one year.".format(title_input))
        else:
            yr1_output = Markup("Books with the characteristics of <i>{}</i> are predicted to have a demand for {} copies after one year.".format(title_input,yr1_pred))
        yr3_pred = predict_title(author_input,publisher_input,country_input,
        document_type_input,3,language_input)[0]
        if yr3_pred == 1:
            yr3_output = Markup("Books with the characteristics of <i>{}</i> are predicted to have a demand for 1 copy after three years.".format(title_input))
        else:
            yr3_output = Markup("Books with the characteristics of <i>{}</i> are predicted to have a demand for {} copies after three years.".format(title_input,yr3_pred))
        return render_template("index.html",
            title_output = title_input,
            output_1=yr1_output,
            output_2=yr3_output,
            my_form_result="NotEmpty")

### Start the server
if __name__ == "__main__":

    ### Determine paths to datasets and load into pandas dataframe
    processed_folder = Path("../data/processed/")
    dictionaries_file = processed_folder / 'variable_dictionaries.joblib'
    model_data = processed_folder / 'decision_tree_parameters.joblib'

    ### Load in label data to categorize new title:
    dictionaries = load(dictionaries_file)
    ### Load saved model
    regr_1 = load(model_data)

    app.run(debug=True)
