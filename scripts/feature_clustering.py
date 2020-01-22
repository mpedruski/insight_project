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

### Variables for which clustering is needed
# need_clustering = ['Auteur','Editeur','Pays']
# count_outputs = [[],[],[]]
# names_outputs = [[],[],[]]

# for i in range(len(need_clustering)):
#     count_outputs[i] = variable_count(need_clustering[i])
#     name_outputs[i] = df[need_clustering].unique()
#
# df1 = pd.DataFrame(results)
# df1 = df1.transpose()

author_count = variable_count('Auteur')
publisher_count = variable_count('Editeur')
country_count = variable_count('Pays')

country_names = df['Pays'].unique()
author_names = df['Auteur'].unique()
pubisher_names = df['Editeur'].unique()

author_labels = variable_cluster('Auteur')
publisher_labels = variable_cluster('Editeur')
country_labels = variable_cluster('Pays')
