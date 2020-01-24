### Load dependencies and dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from pathlib import Path
from scipy import stats
import numpy as np

data_folder = Path("../data/cleaned")
file_to_open = data_folder / 'by_titles.csv'

### Create dataframe
df = pd.read_csv(file_to_open)

fiz = sns.regplot(x="Annee", y="Demanded", data=df)
fiz.set(xlabel='Year of publication', ylabel='Current demand')
plt.xlim(1970, 2000)
fiz = fiz.get_figure()
fiz.savefig('Year_demanded.png')
