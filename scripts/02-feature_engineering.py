import pandas as pd
from pathlib import Path
import datetime
import re
import logging
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs

logging.basicConfig(level=logging.CRITICAL,format='%(asctime)s - %(levelname)s - %(message)s')


def variable_cluster(variable):
    '''Returns cluster labels each instance of a variable, based on the number
    of titles of that instance in the dataset'''
    logging.debug("Numbers of titles associated with each level of {}: {}".format(variable, variable_count(variable)))
    x = np.array(variable_count(variable)).reshape(-1,1)
    # The following bandwidth can be automatically detected using
    bandwidth = estimate_bandwidth(x, quantile=0.2, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(x)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    logging.debug("number of estimated clusters for variable {}: {}".format(variable, n_clusters_))
    return(labels)

def variable_count(variable):
    '''Returns the total number of titles associated with a value of variable'''
    x = df[variable]
    unique_levels = df[variable].unique()
    count = [len(np.where(x==item)[0]) for item in unique_levels]
    return count

def reestablish_labels(variable,names,labels):
    '''Returns the labels that correspond to the individual titles after clustering
    for unique values of the variable'''
    label_column = []
    for i in df[variable]:
        index = np.where(names==i)[0]
        label_column.append(int(labels[index]))
    return label_column

def year_offset(year):
    '''(int) -> list of ints
    Returns the difference in number of years between each title in the data
    and the chosen year'''
    label_column = []
    for i in df['Annee']:
        label_column.append(year-i)
    return label_column

def limited_categorical_variable_numericizer(variable):
    '''(str) -> list of ints
    Returns a numeric list of category labels for each title in the
    dataset based on its level of the supplied variable'''
    label_column = []
    x = df[variable].unique()
    for i in df[variable]:
        for j in range(len(x)):
            if i == x[j]:
                label_column.append(j)
                break
    return label_column

### Determine paths to datasets and load into pandas dataframe
# test_folder = Path("../data/test")
data_folder = Path("../data/cleaned")
processed_folder = Path("../data/processed")
file_to_open = data_folder / 'by_titles.csv'
title_label_data = processed_folder / 'cluster_labels_for_each_title.csv'
variable_label_data = processed_folder / 'cluster_labels_for_each_variable.csv'

df = pd.read_csv(file_to_open)
logging.debug(df)

### Process author, publisher, and country data

author_count = variable_count('Auteur')
publisher_count = variable_count('Editeur')
country_count = variable_count('Pays')

country_names = df['Pays'].unique()
author_names = df['Auteur'].unique()
publisher_names = df['Editeur'].unique()

author_labels = variable_cluster('Auteur')
publisher_labels = variable_cluster('Editeur')
country_labels = variable_cluster('Pays')

title_author_labels = reestablish_labels('Auteur',author_names,author_labels)
title_publisher_labels = reestablish_labels('Editeur',publisher_names,publisher_labels)
title_country_labels = reestablish_labels('Pays',country_names,country_labels)

year_column = year_offset(2019)
document_type_column = limited_categorical_variable_numericizer('Type-document')
language_type_column = limited_categorical_variable_numericizer('Langue')

### Export data set for each title
results = [title_author_labels,title_publisher_labels,title_country_labels,year_column,
    document_type_column,language_type_column,df['Nombre-pages'],df['Demanded']]
df1 = pd.DataFrame(results)
df1 = df1.transpose()
df1.columns = ["Auteur_labels","Editeur_labels","Pays_labels","Years_offset",
    "Document_type_labels","Language_type_labels","Nombre_pages","Demand"]
df1.to_csv(title_label_data)

### Export data set for each variable
labels = [author_names,author_labels,publisher_names,publisher_labels,
    country_names,country_labels,df['Type-document'].unique(),
    range(len(df['Type-document'].unique()))]
df2 = pd.DataFrame(labels)
df2 = df2.transpose()
df2.columns = ["Author_names","Author_labels","Publisher_names","Publisher_labels",
    "Country_names","Country_labels","Type_names","Type_labels"]
df2.to_csv(variable_label_data)
