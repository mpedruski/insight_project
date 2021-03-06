import pandas as pd
from pathlib import Path
import numpy as np
import datetime
import re
import logging

logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s')

def aquire_trait(trait):
    ''' (str, str) -> str
    Accepts an ISBN value and a variable of interest, and returns the value
    of the variable of interest for the first item with that ISBN, if a value
    exists.'''
    try:
        aspect = df.groupby('ISN')[trait].first()
    except:
        aspect = None
    return aspect

def determine_exclude(file, nrows):
    ''' (file, int) -> [int]
    Accepts a CSV file name and an integer, reading the first 'integer' rows
    of the CSV into a datagrame, analyzes the type of item in the row,
    and if the type is not a book appends the items index number to a list
    of titles to be excluded from later analysis.'''
    df = pd.read_csv(file,usecols=[5,6],nrows=nrows)
    ### Isolate categories of document that refer to books
    cats = df['Type-document'].unique()
    book_cats = {cat for cat in cats if "LV" in cat}
    ### Isolate categories of document that are available or in use, not
    ### lost or pulled from the collection (and thus irrelevant)
    use_cats = {'Disponible','Prêté','En transit','Consultation sur place',
        'En traitement','Recherché'}

    exclude_books = np.where(~df['Type-document'].isin(book_cats)==True)[0]+1
    logging.debug('Not a book: {}'.format(exclude_books))
    exclude_use = np.where(~df['Statut-document'].isin(use_cats)==True)[0]+1
    logging.debug('Neither available or demanded: {}'.format(exclude_use))
    exclude = np.unique(np.concatenate([exclude_books, exclude_use]))
    logging.debug('Items to exclude: {}'.format(exclude))
    return exclude

def numericize_availability(isbns):
    '''[str] -> [int]
    Iterates through all the items in a pandas df, returning the value 1
    if the item is available, and 0 if it is not.'''
    ### Convert ISN = disponible to 1, otherwise leave as 0
    logging.debug(df['Statut-document'].unique())
    avails = df['Statut-document'].isin(['Disponible'])
    return avails

def text_remove_from_numeric_data(uncleaned_list):
    '''[str] -> [int]
    Accepts a list of strings that are fundamentally numeric data with string
    formatting, and returns a list of numeric items only'''
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

def author_formatting(uncleaned_list):
    '''[str] -> [int]
    Accepts a list of items that contain unformatted author names, and isolates
    author first name, last name, and middle initial without surrounding
    punctuation/formatting.'''
    logging.debug('Author formatting: Length of list pre formatting = {}'.format(len(uncleaned_list)))
    cleaned_list = []
    pattern = re.compile(r'[a-zA-ZÀ-ÿ].*[a-zA-ZÀ-ÿ](, [a-zA-ZÀ-ÿ].*[a-zA-ZÀ-ÿ])?([a-zA-ZÀ-ÿ])?')
    for uncleaned_item in uncleaned_list:
        if isinstance(uncleaned_item, str):
            pattern_output = pattern.search(str(uncleaned_item))
            if pattern_output is None:
                cleaned_list.append(None)
            else:
                cleaned_list.append(pattern_output.group().replace(".",""))
        else:
            cleaned_list.append(None)
    logging.debug('Length of list post formatting = {}'.format(len(cleaned_list)))
    return cleaned_list

def location_formatting(uncleaned_list):
    '''[str] -> [int]
    Accepts a list of items that contain unformatted author names, and isolates
    author first name, last name, and middle initial without surrounding
    punctuation/formatting.'''
    logging.debug('Location formatting: Length of list pre formatting = {}'.format(len(uncleaned_list)))
    cleaned_list = []
    pattern = re.compile(r'[a-zA-ZÀ-ÿ].*[a-zA-ZÀ-ÿ]')
    for uncleaned_item in uncleaned_list:
        if isinstance(uncleaned_item, str):
            pattern_output = pattern.search(str(uncleaned_item))
            if pattern_output is None:
                cleaned_list.append(None)
            else:
                cleaned_list.append(pattern_output.group())
        else:
            cleaned_list.append(None)
    logging.debug('Length of list post formatting = {}'.format(len(cleaned_list)))
    return cleaned_list

def publisher_formatting(uncleaned_list):
    '''[str] -> [int]
    Accepts a list of publisher names, removes common words, and then uses REGEX
    to remove punctuation to create uniform author names'''
    logging.debug('Publisher formatting: Length of list pre formatting = {}'.format(len(uncleaned_list)))
    cleaned_list = []
    stop_words = ['publishers','books','book','co.','ltée','canada', ' ']
    for uncleaned_item in uncleaned_list:
        if isinstance(uncleaned_item, str):
            uncleaned_item=uncleaned_item.lower()
            for j in stop_words:
                uncleaned_item = uncleaned_item.replace(j,'')
            pattern = re.compile(r'[a-zA-ZÀ-ÿ].*[a-zA-ZÀ-ÿ]')
            pattern_output = pattern.search(str(uncleaned_item))
            if pattern_output is None:
                cleaned_list.append(None)
            else:
                cleaned_list.append(pattern_output.group())
        else:
            cleaned_list.append(None)
    logging.debug('Length of list post formatting = {}'.format(len(cleaned_list)))
    return cleaned_list

if __name__ == "__main__":

    print(datetime.datetime.now())
    data_folder = Path("../data/raw")
    output_folder = Path("../data/cleaned")
    file_to_open = data_folder / 'biblioMTL_cat_2020_01_09.csv'
    file_to_write = output_folder / 'by_titles.csv'

    nrows = None

    ### Determine which rows of dataset refer to books
    exclude = determine_exclude(file_to_open,nrows)

    ### Clean and collate variables of interest on a per-ISBN basis
    df = pd.read_csv(file_to_open, header=0, usecols=[3,5,6,8,10,12,13,14,15,16,17,19],
        skiprows=exclude,nrows=nrows)

    ### Numericize ISBNs to make for faster handling and to prevent duplicates
    ### and to feed to other variables for their grouping
    df['ISN'] = text_remove_from_numeric_data(df['ISN'])
    isns = df['ISN'].unique()
    ### Convert availability of items to numeric values
    df['Statut-document'] = numericize_availability(isns)

    ### Count availability, lifetime borrows, and total copie availability by ISBN
    available_count = df.groupby('ISN')['Statut-document'].sum()
    lifetime_count = df.groupby('ISN')['Nombre-prets-vie'].sum()
    total_count = df.groupby('ISN')['ISN'].count()

    ### Clean year and page data by removing text
    df['Annee'] = text_remove_from_numeric_data(df['Annee'])
    df['Nombre-pages'] = text_remove_from_numeric_data(df['Nombre-pages'])

    ### Clean author, place, and publisher data by removing unnecessary formatting
    df['Auteur'] = author_formatting(df['Auteur'])
    df['Lieu'] = location_formatting(df['Lieu'])
    df['Editeur'] = publisher_formatting(df['Editeur'])

    ### Caculated demand for ISBNs by subtracting number available from total number
    demand_count = []
    for i in range(len(available_count)):
        demand_count.append(total_count[i]-available_count[i])

    logging.debug('Inputting uniform text for all items that share a ISBN')
    columns_to_build = ['Titre','Auteur','Editeur','Pays','Annee','Nombre-pages','Langue','Type-document','Lieu']
    results = [aquire_trait(i) for i in columns_to_build]

    ### Join datasets, remove rows for which data is missing, and export to csv
    df1 = pd.DataFrame(results)
    logging.debug('Beginning transpose')
    df1 = df1.transpose()
    logging.debug('Done transpose')
    df1.columns = columns_to_build
    df2 = pd.DataFrame({'Total':total_count,'Available':available_count,'Demanded':demand_count,'Lifetime':lifetime_count})
    df1 = df1.join(df2)
    df1 = df1.dropna()
    df1.to_csv(file_to_write)
    print(datetime.datetime.now())
