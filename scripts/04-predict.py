from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import OneHotEncoder
from joblib import load
import numpy as np
import pandas as pd
from pathlib import Path
import datetime
import logging

### Regression based on historical data

### Determine paths to datasets and load into pandas dataframe
# test_folder = Path("../data/test")
processed_folder = Path("../data/processed")
variable_data = processed_folder / 'cluster_labels_for_each_variable.csv'
model_data = processed_folder / 'SGD_model_parameters.joblib'
encoding_data = processed_folder / 'OHE_parameters.joblib'

df = pd.read_csv(variable_data)
### Load in label data to categorize new title:

author_labels = df.set_index('Author_names')['Author_labels'].to_dict()
publisher_labels = df.set_index('Publisher_names')['Publisher_labels'].to_dict()
country_labels = df.set_index('Country_names')['Country_labels'].to_dict()
type_labels = df.set_index('Type_names')['Type_labels'].to_dict()
### Load saved model
clf = load(model_data)
enc = load(encoding_data)
def predict_title(author, publisher, country, doc_type):
    title_data = np.array([author_labels[author],publisher_labels[publisher],
        country_labels[country],type_labels[doc_type]]).astype(int)
    onehotlabels_new = enc.transform(title_data.reshape(1,-1))
    prediction = clf.predict(onehotlabels_new)
    return prediction

res = predict_title('Derib,','Casterman,','quc','LV_Fiction J')
print(res)
