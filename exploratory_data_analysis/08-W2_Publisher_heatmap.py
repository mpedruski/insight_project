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
publishers = df['Editeur'].unique()

publisher_demands = []
for publisher in publishers:
    documents = df.loc[df['Editeur'] == publisher]
    publisher_demands.append(sum(documents['Demanded']))

print(len(publisher_demands))

# Subsample 400 datapoints for plotting

publisher_demands = publisher_demands[:400]
data = np.asarray(publisher_demands).reshape(20,20)

### Make histogram of current demand
ax = sns.heatmap(data, cbar_kws={'label': "Total demands for publisher's books"})
plt.axis('off')
ax.set_title('Sample of 400 publishers')
fig = ax.get_figure()
fig.savefig('Publisher_demand_heatmap.png')
