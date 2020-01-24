### Load dependencies and dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

data_folder = Path("../data/cleaned")
file_to_open = data_folder / 'by_titles.csv'

### Create dataframe
df = pd.read_csv(file_to_open)

# ### Make histogram of lifetime loan distribution
# ax = sns.distplot(df["Lifetime"],kde=False)
# fig = ax.get_figure()
# fig.savefig('Lifetime_loans_histogram.png')

### Make histogram of current demand
ax = sns.distplot(df["Demanded"],kde=False)
fig = ax.get_figure()
fig.savefig('Current_demand_histogram.png')
#
# ### Make histogram of total copy count
# ax = sns.distplot(df["Total"],kde=False)
# fig = ax.get_figure()
# fig.savefig('Available_copies_histogram.png')
