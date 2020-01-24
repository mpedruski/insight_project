import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

### Set path to csv that combines all data by title and load
data_folder = Path("../data/processed")
file_to_open = data_folder / 'by_titles.csv'
file_to_write = data_folder / 'need_demands.csv'
df = pd.read_csv(file_to_open)

### How many documents have 0 availability and thus need web augmentation?
x = df['Available']
y = np.where(x==0)
zero_availability = list(y[0])
### Return results where there are 0 availability
df = df.iloc[zero_availability]
df_output = pd.concat([df['Titre'],df['Auteur'],df['ISBN']],axis=1)
df_output.to_csv(file_to_write)
