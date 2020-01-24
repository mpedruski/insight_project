### Load dependencies and dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

data_folder = Path("../data/raw")
file_to_open = data_folder / 'biblioMTL_cat_2020_01_09.csv'

### Create dataframe and sort for lifetime loans
df = pd.read_csv(file_to_open,usecols=[3,6,8,10])
### Convert statut-document type to category to save on memory
df['Type-document']=df['Type-document'].astype('category')

df1 = df.sort_values(by="Nombre-prets-vie")[-500:]
df1.to_csv(r'Top_500_Library_books.csv')
