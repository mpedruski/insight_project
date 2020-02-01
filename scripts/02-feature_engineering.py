import pandas as pd
from pathlib import Path
import datetime
import re
import logging
import numpy as np
from joblib import dump

np.random.seed(31415)
logging.basicConfig(level=logging.CRITICAL,format='%(asctime)s - %(levelname)s - %(message)s')

def names_list(variables):
    '''[str] -> [str]
    Accepts a list of column titles, and returns a list of lists names for each
    level of a variable'''
    names_list = []
    for variable in variables:
        logging.debug(variable)
        x = df.groupby(variable)[variable].first().values.tolist()
        logging.debug('List of names for a variable:{}'.format(x))
        names_list.append(x)
    logging.debug("List of names: {}".format(names_list))
    return names_list

def reestablish_labels(variables, names_list):
    '''Returns the labels that correspond to the individual titles after clustering
    for unique values of the variable'''
    labels_name_accumulator = []
    for i in range(len(variables)):
        logging.debug("Iterator: {}".format(i))
        label_column = []
        to_find = np.array(df[variables[i]])
        logging.debug('To find: {}'.format(to_find))
        labels = np.array(df.groupby(variables[i])['Lifetime'].sum())
        logging.debug("Labels: {}".format(labels))
        names = np.array(names_list[i])
        logging.debug("Names: {}".format(names))
        for j in to_find:
            logging.debug(j)
            index = np.where(names==j)[0][0]
            logging.debug(index)
            # logging.debug("Labels should have {} appended".format(labels[index]))
            label_column.append(int(labels[index]))
        logging.debug("{} label column length: {}".format(variables[i], len(label_column)))
        labels_name_accumulator.append(label_column)
    # logging.debug("Labels list: {}".format(labels_name_accumulator))
    return labels_name_accumulator

def year_offset(year):
    '''(int) -> list of ints
    Returns the difference in number of years between each title in the data
    and the chosen year'''
    label_column = []
    for i in df['Annee']:
        label_column.append(year-i)
    return label_column

def limited_categorical_variable_numericizer(variables):
    '''[str] -> [ints]
    Returns a numeric list of category labels for each title in the
    dataset based on its level of the supplied variable'''
    list_of_lists=[]
    for variable in variables:
        label_column = []
        x = df[variable].unique()
        for i in df[variable]:
            for j in range(len(x)):
                if i == x[j]:
                    label_column.append(j)
                    break
        list_of_lists.append(label_column)
    return list_of_lists

### Determine paths to datasets and load into pandas dataframe

data_folder = Path("../data/cleaned")
processed_folder = Path("../data/processed")
file_to_open = data_folder / 'by_titles.csv'
files_to_write=['cluster_labels_for_each_title.csv', 'variable_dictionaries.joblib']

df = pd.read_csv(file_to_open,nrows=100000)
logging.critical('Beginning analysis')

### Process author, publisher, and country data
logging.critical('Imputing lifetime borrows to authors, publishers, and countries')
### Return values for authors, publishers, and countries based on the lifetime number
### of borrows for them and create lists of names for each author, publisher,
### and country
variables_to_label = ['Auteur','Editeur','Pays']
names_list = names_list(variables_to_label)
titles_label_values = reestablish_labels(variables_to_label,names_list)

logging.critical('Manipulating year data and categorizing document type and language')
### Giving category numbers to variables that don't have enough levels to merit
### a finer approach
numerical_categories = limited_categorical_variable_numericizer(['Type-document','Langue'])
year_column = year_offset(2020)

logging.critical('Saving output')
### Export data set for each title
results = [titles_label_values[0],titles_label_values[1],titles_label_values[2],
    year_column,numerical_categories[0],numerical_categories[1],df['Nombre-pages'],df['Demanded']]
df1 = pd.DataFrame(results).transpose()
df1.columns = ["Auteur_labels","Editeur_labels","Pays_labels","Years_offset",
    "Document_type_labels","Language_type_labels","Nombre_pages","Demand"]
df1.to_csv(processed_folder / files_to_write[0])

### Save dictionaries to file for easy import into server
dictionaries = [dict(zip(names_list[0], df.groupby('Auteur')['Lifetime'].sum())),
    dict(zip(names_list[1], df.groupby('Editeur')['Lifetime'].sum())),
    dict(zip(names_list[2], df.groupby('Pays')['Lifetime'].sum())),
    dict(zip(df['Type-document'].unique(),range(len(df['Type-document'].unique())))),
    dict(zip(df['Langue'].unique(),range(len(df['Langue'].unique()))))]
dump(dictionaries, processed_folder / files_to_write[1])
