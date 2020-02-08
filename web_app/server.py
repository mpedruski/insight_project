from flask import Flask, render_template, request, Markup
from joblib import load
import numpy as np
import re
from pathlib import Path
import logging
logging.basicConfig(level=logging.CRITICAL,format='%(asctime)s - %(levelname)s - %(message)s')


def predict_title(author, publisher, country, doc_type, years, language, title):
    ''' [str, str, str, str, [int], str] -> int
    Accepts an author key, a publisher key, a country key, a document-type key,
    a list of years from the present, and a language key, finds the corresponding value
    to each key (as well as the year from the present as an int) and passes
    these to the loaded regression model to return predicted output'''
    outputs = []
    for i in range(len(years)):
        # prediction = round(rf_model.predict(np.array([0,
        #     0, 996559, 0, 4218963, 3222549]).reshape(1,-1))[0]).astype(int)
        # prediction = round(rf_model.predict(np.array([dictionaries[0].get('Pedruski, Michael',0),
        #     dictionaries[1].get('Late Nite Press',0), dictionaries[2]['xxc'], 0,
        #     dictionaries[3]['LV_Documentaire A'], dictionaries[4]['eng']]).reshape(1,-1))[0]).astype(int)
        prediction = round(rf_model.predict(np.array([dictionaries[0].get(author,0),
            dictionaries[1].get(publisher,0), dictionaries[2][country], 0,
            dictionaries[3][doc_type], dictionaries[4][language]]).reshape(1,-1))[0]).astype(int)
        if prediction == 1:
            outputs.append(Markup("Predicted demand for <i>{}</i> is <b>1 copy<b>.".format(title)))
        else:
            outputs.append(Markup("Predicted demand for <i>{}</i> is <b>{}<b> copies.".format(title, prediction)))
    return outputs

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

def author_publisher_warning(author, publisher):
    if author not in dictionaries[0].keys():
        author_warning = "Your author wasn't found in the catalogue, and 0 lifetime borrows have been attributed to them."
    else:
        author_warning = ""
    if publisher not in dictionaries[1].keys():
        publisher_warning = "Your publisher wasn't found in the catalogue, and 0 lifetime borrows have been attributed to them."
    else:
        publisher_warning = ""
    if author not in dictionaries[0].keys() or publisher not in dictionaries[1].keys():
        general_warning = "If this is unexpected, check that your entry format meets the expected format."
    else:
        general_warning = ""
    warning_output = author_warning + " " + publisher_warning + " " + general_warning
    return warning_output

### Determine paths to datasets and load into pandas dataframe
processed_folder = Path("../data/processed/")
dictionaries_file = processed_folder / 'expanded_variable_dictionaries.joblib'
model_data = processed_folder / 'expanded_regression_tree_parameters.joblib'

### Load in label data to categorize new title:
dictionaries = load(dictionaries_file)
### Load saved model
rf_model = load(model_data)

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
        outputs = predict_title(author_input,publisher_input,country_input,
            document_type_input,[0],language_input,title_input)
        warning_output = author_publisher_warning(author_input, publisher_input)
        return render_template("index.html",
            title_output = title_input,
            output_0=outputs[0],
            # output_1=outputs[1],
            # output_2=outputs[2],
            output_3=warning_output,
            my_form_result="NotEmpty")

### Start the server
if __name__ == "__main__":

    app.run(debug=True)
