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

### Number of books plotted
print(df.shape)

### Make new dataframe from x and y, and plot
ax = sns.catplot(y='Demanded',x='Type-document',kind = 'box',data=df)
ax.set_axis_labels('Document type','Demanded copies per title')
ax.set_xticklabels(rotation=30,ha='right')
plt.title("Demands per title for 2489 books")
ax.savefig('Document-type_boxplots.png')
