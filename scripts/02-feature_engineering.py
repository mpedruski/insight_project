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
    x = np.array(variable).reshape(-1,1)
    # The following bandwidth can be automatically detected using
    bandwidth = estimate_bandwidth(x, quantile=0.2, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(x)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    return(labels)

def reestablish_labels(variable,dataframe):
    '''Returns the labels that correspond to the individual titles after clustering
    for unique values of the variable'''
    label_column = []
    labels = np.array(dataframe['Label'])
    logging.debug("Labels: {}".format(labels))
    names = np.array(dataframe['Variable'])
    logging.debug("Names: {}".format(names))
    to_find = np.array(df[variable])
    logging.debug('To find: {}'.format(to_find))
    for i in to_find:
        logging.debug(i)
        index = np.where(names==i)[0][0]
        logging.debug(index)
        logging.debug("Labels should have {} appended".format(labels[index]))
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

df = pd.read_csv(file_to_open,nrows=10000)
logging.debug('Beginning analysis')

### Process author, publisher, and country data
logging.debug('Counting number of ISBNs associated with a variable')
### Identify how many ISBNs are associated with each author, publisher,
### or country
author_count = df.groupby('Auteur')['ISN'].count()
publisher_count = df.groupby('Editeur')['ISN'].count()
country_count = df.groupby('Pays')['ISN'].count()
author_names = df.groupby('Auteur')['Auteur'].first()
publisher_names = df.groupby('Editeur')['Editeur'].first()
country_names = df.groupby('Pays')['Pays'].first()
### Cluster authors, publishers, and countries based on the number of publications
### attributed to them
logging.debug('Beginning clustering')
author_labels = variable_cluster(author_count)
author_names_values = author_names.values
author_data = [author_names_values,author_labels]
dfa = pd.DataFrame(author_data).transpose()
dfa.columns = ['Variable','Label']
logging.debug("Author names and their labels = {} ".format(dfa))

publisher_labels = variable_cluster(publisher_count)
publisher_names_values = publisher_names.values
publisher_data = [publisher_names_values,publisher_labels]
dfp = pd.DataFrame(publisher_data).transpose()
dfp.columns = ['Variable','Label']

country_labels = variable_cluster(country_count)
country_names_values = country_names.values
country_data = [country_names_values,country_labels]
dfc = pd.DataFrame(country_data).transpose()
dfc.columns = ['Variable','Label']

### Attaching results of clustering (labels) to each ISBN in the catalogue
logging.debug('Linking cluster labels to names')
title_author_labels = reestablish_labels('Auteur',dfa)
logging.debug("Vector of labels for author = {} ".format(title_author_labels))
logging.debug("Vector of author names = {} ".format(df['Auteur']))
title_publisher_labels = reestablish_labels('Editeur',dfp)
title_country_labels = reestablish_labels('Pays',dfc)

### Giving category numbers to variables that don't have enough levels to merit
### clustering
logging.debug('Manipulating year data and categorizing document type and language')
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
variable_labels = [dfa['Variable'].values,dfa['Label'].values,dfp['Variable'].values,
    dfp['Label'].values,dfc['Variable'].values,dfc['Label'].values,
    df['Type-document'].unique(),range(len(df['Type-document'].unique()))]
df2 = pd.DataFrame(variable_labels)
df2 = df2.transpose()
df2.columns = ["Author_names","Author_labels","Publisher_names","Publisher_labels",
    "Country_names","Country_labels","Type_names","Type_labels"]
df2.to_csv(variable_label_data)
