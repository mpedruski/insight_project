### Load dependencies and dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from pathlib import Path
from scipy import stats
import numpy as np

data_folder = Path("../data/test")
file_to_open = data_folder / 'by_titles.csv'

### Create dataframe
df = pd.read_csv(file_to_open)

### Count current demand for all titles by an author
countries = df['Pays'].unique()

country_demands = []
for country in countries:
    documents = df.loc[df['Pays'] == country]
    country_demands.append(sum(documents['Demanded']))

print(len(countries))
