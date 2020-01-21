import pandas as pd
from pathlib import Path
import datetime
import re
import logging

logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s')

def available_copies(isn):
    documents = df1.loc[df1['ISN'] == isn]
    available_count = sum(documents['Statut-document'])
    return available_count

def total_copies(isn):
    documents = df1.loc[df1['ISN'] == isn]
    all_copies = len(documents)
    return all_copies

def aquire_trait(isn,trait):
    documents = df1.loc[df1['ISN'] == isn]
    aspect = documents[trait]
    candidates = {}
    for i in aspect:
        candidates.setdefault(i,0)
        candidates[i] = candidates[i] + 1
    if len(candidates)>0:
        accept_aspect = max(candidates,key=candidates.get)
    else:
        accept_aspect = None
    return accept_aspect

def determine_exclude(file, nrows):
    df = pd.read_csv(file,usecols=[6],nrows=nrows)
    ### Isolate categories of document that refer to books
    df['Type-document']=df['Type-document'].astype('category')
    cats = df['Type-document'].unique()
    book_cats = {cat for cat in cats if "LV" in cat}

    ### Which rows of the dataset don't refer to books
    exclude = []
    for i in range(len(df['Type-document'])):
        if df['Type-document'][i] not in book_cats:
            exclude.append(i+1)
    return exclude

def numericize_availability(isbns):
    ### Convert ISN = disponible to 1, otherwise leave as 0
    avails = []
    df_isn = df1['Statut-document']
    for i in range(len(df_isn)):
        if df_isn[i] == 'Disponible':
            avails.append(1)
        else:
            avails.append(0)
    return avails

def text_remove_from_numeric_data(uncleaned_list):
    logging.debug('Length of list pre formatting = {}'.format(len(uncleaned_list)))
    cleaned_list = []
    pattern = re.compile(r'[0-9]+')
    for uncleaned_item in uncleaned_list:
        pattern_output = pattern.search(str(uncleaned_item))
        if pattern_output is None:
            cleaned_list.append('nan')
        else:
            cleaned_list.append(pattern_output.group())
    logging.debug('Length of list post formatting = {}'.format(len(cleaned_list)))
    return cleaned_list

print(datetime.datetime.now())

### Determine paths to datasets
data_folder = Path("../data/raw")
test_folder = Path("../data/test")

file_to_open = data_folder / 'biblioMTL_cat_2020_01_09.csv'
file_to_write = test_folder / 'by_titles.csv'
test_file = test_folder / 'biblioMTL_small_test.csv'
# test_file = test_folder / 'biblioMTL_big_test.csv'

nrows = 300

### Determine which rows of dataset refer to books
exclude = determine_exclude(test_file,nrows)

### Clean and collate variables of interest on a per-title basis
df1 = pd.read_csv(test_file, header=0, usecols=[5,6,8,10,12,13,14,15,16,17,19],
    skiprows=exclude,nrows=(nrows-len(exclude)))
### Find list of unique ISNs in collection
isns = df1['ISN'].unique()
### Convert availability of titles to numeric values
df1['Statut-document'] = numericize_availability(isns)
### Clean year and page data
df1['Annee'] = text_remove_from_numeric_data(df1['Annee'])
df1['Nombre-pages'] = text_remove_from_numeric_data(df1['Nombre-pages'])

### Creating lists for variables to combine into a reduced dataframe
available_count = [available_copies(i) for i in isns]
total_count = [total_copies(i) for i in isns]
demand_count = []
for i in range(len(available_count)):
    demand_count.append(total_count[i]-available_count[i])
columns_to_build = ['Titre','Auteur','Editeur','Lieu','Pays','Annee','Nombre-pages','Langue','Type-document']
results = [[],[],[],[],[],[],[],[],[]]
for i in range(len(columns_to_build)):
    results[i] = [aquire_trait(j,columns_to_build[i]) for j in isns]

### After all processing is complete, clean ISBN data
isns = text_remove_from_numeric_data(isns)
### Build and join datasets based on list results, export to csv


df3 = pd.DataFrame(results)
df3 = df3.transpose()
df3.columns = columns_to_build
df4 = pd.DataFrame({'Total':total_count,'Available':available_count,'Demanded':demand_count,'ISBN':isns})
df5 = df3.join(df4)
df5 = df5.dropna()
df5.to_csv(file_to_write)
print(datetime.datetime.now())
