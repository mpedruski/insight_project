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
authors = df['Auteur'].unique()

author_demands = []
for author in authors:
    documents = df.loc[df['Auteur'] == author]
    author_demands.append(sum(documents['Demanded']))

# Subsample 400 datapoints for plotting

author_demands = author_demands[:900]
data = np.asarray(author_demands).reshape(30,30)

### Make histogram of current demand
ax = sns.heatmap(data, cbar_kws={'label': "Total demands for author's works"})
plt.axis('off')
ax.set_title('Sample of 900 authors')
fig = ax.get_figure()
fig.savefig('Current_demand_heatmap.png')
