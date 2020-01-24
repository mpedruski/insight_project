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

fiz = sns.scatterplot(x = df["Total"], y = df["Demanded"])
fiz.set(xlabel='Total number of copies', ylabel='Current demand')
fiz.plot([0, 0], [80, 80], linewidth=2)
fiz = fiz.get_figure()
fiz.savefig('Available_demanded.png')
