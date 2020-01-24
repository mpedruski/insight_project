import pandas as pd
from pathlib import Path
import datetime
import re
import logging

logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s')

def combine_numeric_data(isn, variable):
    documents = df.loc[df['ISN'] == isn]
    count = sum(documents[variable])
    return count

def total_copies(isn):
    documents = df.loc[df['ISN'] == isn]
    all_copies = len(documents)
    return all_copies

def aquire_trait(isn,trait):
    documents = df.loc[df['ISN'] == isn]
    try:
        aspect = documents[trait].iloc[0]
    except:
        aspect = None
    return aspect

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
    df_isn = df['Statut-document']
    for i in range(len(df_isn)):
        if df_isn[i] == 'Disponible':
            avails.append(1)
        else:
            avails.append(0)
    return avails

def text_remove_from_numeric_data(uncleaned_list):
    logging.debug('Text remove from numeric: Length of list pre formatting = {}'.format(len(uncleaned_list)))
    cleaned_list = []
    pattern = re.compile(r'[0-9]+')
    for uncleaned_item in uncleaned_list:
        pattern_output = pattern.search(str(uncleaned_item))
        if pattern_output is None:
            cleaned_list.append(None)
        else:
            cleaned_list.append(pattern_output.group())
    logging.debug('Length of list post formatting = {}'.format(len(cleaned_list)))
    return cleaned_list

print(datetime.datetime.now())

### Determine paths to datasets
data_folder = Path("../data/raw")
test_folder = Path("../data/test")
output_folder = Path("../data/cleaned")

file_to_open = data_folder / 'biblioMTL_cat_2020_01_09.csv'
file_to_write = test_folder / 'by_titles.csv'
# test_file = test_folder / 'biblioMTL_small_test.csv'
test_file = test_folder / 'biblioMTL_big_test.csv'

nrows = 10000

### Determine which rows of dataset refer to books
exclude = determine_exclude(file_to_open,nrows)

### Clean and collate variables of interest on a per-title basis
df = pd.read_csv(file_to_open, header=0, usecols=[3,5,6,8,10,12,13,14,15,16,17,19],
    skiprows=exclude,nrows=(nrows-len(exclude)))
### Find list of unique ISNs in collection
isns = df['ISN'].unique()
### Convert availability of titles to numeric values
df['Statut-document'] = numericize_availability(isns)
### Clean year and page data
df['Annee'] = text_remove_from_numeric_data(df['Annee'])
df['Nombre-pages'] = text_remove_from_numeric_data(df['Nombre-pages'])

### Creating lists for variables to combine into a reduced dataframe
logging.debug('Initiating available count')
available_count = [combine_numeric_data(i,'Statut-document') for i in isns]
logging.debug('Initiating lifetime count')
lifetime_count = [combine_numeric_data(i,'Nombre-prets-vie') for i in isns]
logging.debug('Initiating total count')
total_count = [total_copies(i) for i in isns]
logging.debug('Initiating demand count')
demand_count = []
for i in range(len(available_count)):
    demand_count.append(total_count[i]-available_count[i])
columns_to_build = ['Titre','Auteur','Editeur','Lieu','Pays','Annee','Nombre-pages','Langue','Type-document']
results = [[],[],[],[],[],[],[],[],[]]
logging.debug('Inputting uniform text for all items that share a ISBN')

for i in range(len(columns_to_build)):
    results[i] = [aquire_trait(j,columns_to_build[i]) for j in isns]

### After all processing is complete, clean ISBN data
isns = text_remove_from_numeric_data(isns)
### Build and join datasets based on list results, export to csv


df1 = pd.DataFrame(results)
logging.debug('Beginning transpose')
df1 = df1.transpose()
logging.debug('Done transpose')
df1.columns = columns_to_build
df2 = pd.DataFrame({'Total':total_count,'Available':available_count,'Demanded':demand_count,'Lifetime':lifetime_count,'ISBN':isns})
df1 = df1.join(df2)
df1 = df1.dropna()
df1.to_csv(file_to_write)
print(datetime.datetime.now())
