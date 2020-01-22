import pandas as pd
from pathlib import Path
import datetime
import re
import logging
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs


logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s')

### Determine paths to datasets and load into pandas dataframe
test_folder = Path("../data/processed")
file_to_open = test_folder / 'by_titles.csv'
file_to_write = test_folder / 'cluster_labels.csv'

df = pd.read_csv(file_to_open)

def variable_cluster(variable):
    '''Returns cluster labels each instance of a variable, based on the number
    of titles of that instance in the dataset'''
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
title_country_labels = reestablish_labels('Editeur',country_names,country_labels)

results = [title_author_labels,title_publisher_labels,title_country_labels]
df1 = pd.DataFrame(results)
df1 = df1.transpose()
df1.columns = ["Auteur_labels","Editeur_labels","Pays_labels"]

df1.to_csv(file_to_write)
