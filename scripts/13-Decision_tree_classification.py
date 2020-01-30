from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix

from joblib import dump
import numpy as np
import pandas as pd
from pathlib import Path
import datetime
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s')

### Regression based on historical data

### Determine paths to datasets and load into pandas dataframe
# test_folder = Path("../data/test")
processed_folder = Path("../data/processed")
file_to_open = processed_folder / 'cluster_labels_for_each_title.csv'
model_parameters = processed_folder / 'SGD_model_parameters.joblib'
encoding_parameters = processed_folder / 'OHE_parameters.joblib'

df = pd.read_csv(file_to_open)


### Find length of majority and minority class data
minority_class_len = len(df[df['Demand']==1])
majority_class_indices = df[df['Demand']==0].index

### Generate a list of majority class indices to retain, based on how many minority
### class data their are
random_majority_indices = np.random.choice(majority_class_indices, minority_class_len,
    replace = False)
### Make list of minority class indices and combine with majority class indices
### then define dataset as the rows of the dataset from those indices
minority_class_indices = df[df['Demand']>0].index
under_sample_indices = np.concatenate([minority_class_indices,random_majority_indices])
df = df.loc[under_sample_indices]


### Select variable to predict as well as features to be used
y = df['Demand']
x = df[['Auteur_labels','Editeur_labels','Pays_labels','Document_type_labels',
    'Years_offset','Nombre_pages','Language_type_labels']]
### Split dataset into training and testing components
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# logging.debug('Numeric features {}'.format(onehotlabels_train))

## Classify features on variable to predict, and test against
### test set
regr_1 = DecisionTreeClassifier(max_depth=7, min_samples_leaf=10)
regr_1_model = regr_1.fit(x_train, y_train)
y_pred = regr_1.predict(x_test)
print(regr_1.score(x_test,y_test))
print(confusion_matrix(y_test, y_pred))
print(sum(confusion_matrix(y_test, y_pred)))
# scores = np.zeros((7,19))
# for i in range(3,10):
#     for j in range(1,20):
#         regr_1 = DecisionTreeClassifier(max_depth=j, min_samples_leaf=i)
#         regr_1_model = regr_1.fit(x_train, y_train)
#         # y_1 = regr_1.predict(x_test)
#         scores[i-3,j-1]=regr_1.score(x_test,y_test)
#         # print(R2)
#     # print(y_1.min(),y_1.max())
# print(np.amax(scores))
# parameters = np.unravel_index(scores.argmax(), scores.shape)
# print(parameters)
# plt.figure()
# regr_1 = DecisionTreeClassifier(max_depth=parameters[1]+1, min_samples_leaf=parameters[0]+3)
# regr_1.fit(x_train, y_train)
# print(regr_1.score(x_test,y_test))
# plot_tree(regr_1, filled=True)
# plt.savefig('Regression_tree_output.pdf',format='pdf')
