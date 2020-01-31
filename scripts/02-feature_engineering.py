import pandas as pd
from pathlib import Path
import datetime
import re
import logging
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
from joblib import dump

np.random.seed(31415)
logging.basicConfig(level=logging.CRITICAL,format='%(asctime)s - %(levelname)s - %(message)s')

def variable_cluster(variables):
    '''[str] -> [int]
    Accepts a list of column titles, and returns a list of lists containing integer
    output cluster labels for each instance of each variable, based on the number
    of lifetime borrows for elements of the variable'''
    labels_list = []
    for variable in variables:
        logging.debug('Variable for clustering:{}'.format(variable))
        x = df.groupby(variable)['Lifetime'].sum()
        logging.debug('Lifetime publications:{}'.format(x))
        x = np.array(x).reshape(-1,1)
        # The following bandwidth can be automatically detected using
        bandwidth = estimate_bandwidth(x, quantile=0.2, n_samples=500)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(x)
        labels = ms.labels_
        logging.debug('Labels:{}'.format(labels))
        labels_list.append(labels)
    logging.debug("List of labels: {}".format(labels_list))
    return labels_list

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

def reestablish_labels(variables, names_list, labels_list):
    '''Returns the labels that correspond to the individual titles after clustering
    for unique values of the variable'''
    labels_name_accumulator = []
    for i in range(len(variables)):
        logging.debug("Iterator: {}".format(i))
        label_column = []
        to_find = np.array(df[variables[i]])
        logging.debug('To find: {}'.format(to_find))
        labels = np.array(labels_list[i])
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
processed_folder = Path("../data/test")
# processed_folder = Path("../data/processed")
file_to_open = data_folder / 'by_titles.csv'
title_label_data = processed_folder / 'cluster_labels_for_each_title.csv'
author_dictionary_file = processed_folder / 'author_dictionary_clusters.joblib'
publisher_dictionary_file = processed_folder / 'publisher_dictionary_clusters.joblib'
country_dictionary_file = processed_folder / 'country_dictionary_clusters.joblib'
type_dictionary_file = processed_folder / 'type_dictionary_categories.joblib'
language_dictionary_file = processed_folder / 'language_dictionary_categories.joblib'

df = pd.read_csv(file_to_open,nrows=3000)
logging.critical('Beginning analysis')

### Process author, publisher, and country data
logging.critical('Counting number of ISBNs associated with a variable')

### Cluster authors, publishers, and countries based on the lifetime number
### of borrows for them and create lists of names for each author, publisher,
### and country
variables_to_cluster = ['Auteur','Editeur','Pays']
logging.critical('Beginning clustering')
labels_list = variable_cluster(variables_to_cluster)
names_list = names_list(variables_to_cluster)
logging.debug("Length of labels list {}, length of names list: {}".format(len(labels_list),len(names_list)))
logging.critical("Labels list [0]: {}".format(labels_list[0]))
titles_cluster_values = reestablish_labels(variables_to_cluster,names_list,labels_list)
### Giving category numbers to variables that don't have enough levels to merit
### clustering
logging.critical('Manipulating year data and categorizing document type and language')
year_column = year_offset(2020)
numerical_categories = limited_categorical_variable_numericizer(['Type-document','Langue'])
logging.critical('Saving output')

### Export data set for each title
results = [titles_cluster_values[0],titles_cluster_values[1],titles_cluster_values[2],
    year_column,numerical_categories[0],numerical_categories[1],df['Nombre-pages'],df['Demanded']]
df1 = pd.DataFrame(results)
df1 = df1.transpose()
df1.columns = ["Auteur_labels","Editeur_labels","Pays_labels","Years_offset",
    "Document_type_labels","Language_type_labels","Nombre_pages","Demand"]
df1.to_csv(title_label_data)

### Save dictionaries to file
dump(dict(zip(names_list[0], labels_list[0])), author_dictionary_file)
dump(dict(zip(names_list[1], labels_list[1])), publisher_dictionary_file)
dump(dict(zip(names_list[2], labels_list[2])), country_dictionary_file)
dump(dict(zip(df['Type-document'].unique(),range(len(df['Type-document'].unique())))), type_dictionary_file)
dump(dict(zip(df['Langue'].unique(),range(len(df['Langue'].unique())))), language_dictionary_file)
