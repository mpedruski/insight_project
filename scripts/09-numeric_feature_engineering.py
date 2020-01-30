import pandas as pd
from pathlib import Path
import datetime
import re
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.linear_model import SGDRegressor
from joblib import dump

logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s')

def reestablish_labels(variable):
    '''Returns the labels that correspond to the individual titles after clustering
    for unique values of the variable'''
    to_find = np.array(df[variable])
    names = np.array(df.groupby(variable)[variable].first())
    counts = np.array(df.groupby(variable)['Lifetime'].sum())
    logging.debug('Beginning realignment of {}'.format(variable))
    label_column = []
    for i in to_find:
        index = np.where(names==i)[0][0]
        label_column.append(int(counts[index]))
    return label_column
    # print(author_lifetime)
    logging.debug('Ending realignment of {}'.format(variable))

### Determine paths to datasets and load into pandas dataframe

data_folder = Path("../data/cleaned")
output_folder = Path("../data/processed")
file_to_open = data_folder / 'by_titles.csv'
author_dictionary_file = output_folder / 'author_dictionary.joblib'
publisher_dictionary_file = output_folder / 'publisher_dictionary.joblib'
country_dictionary_file = output_folder / 'country_dictionary.joblib'
coded_variables_file = output_folder / 'coded_variables.joblib'

### Create dataframe
df = pd.read_csv(file_to_open,nrows=None)

### Labels for each title in the dataset for popularity of author, publisher,
### country

author_counts = np.array(df.groupby('Auteur')['Lifetime'].sum())
publisher_counts = np.array(df.groupby('Editeur')['Lifetime'].sum())
country_counts = np.array(df.groupby('Pays')['Lifetime'].sum())

logging.debug("Author counts: {}".format(len(author_counts)))

book_ID = df.groupby('ISN')['ISN'].first()
author_values = reestablish_labels('Auteur')
publisher_values = reestablish_labels('Editeur')
country_values = reestablish_labels('Pays')
coded_variables = [author_values,publisher_values,country_values]
author_names = df.groupby('Auteur')['Auteur'].first()
publisher_names = df.groupby('Editeur')['Editeur'].first()
country_names = df.groupby('Pays')['Pays'].first()

author_dict = dict(zip(author_names, author_counts))
publisher_dict = dict(zip(publisher_names, publisher_counts))
country_dict = dict(zip(country_names, country_counts))

## Save output to file
dump(author_dict, author_dictionary_file)
dump(publisher_dict, publisher_dictionary_file)
dump(country_dict, country_dictionary_file)
dump(coded_variables,coded_variables_file)
