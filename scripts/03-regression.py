from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import OneHotEncoder
from joblib import dump
import numpy as np
import pandas as pd
from pathlib import Path
import datetime
import logging

### Regression based on historical data

### Determine paths to datasets and load into pandas dataframe
# test_folder = Path("../data/test")
processed_folder = Path("../data/processed")
file_to_open = processed_folder / 'cluster_labels_for_each_title.csv'
model_parameters = processed_folder / 'SGD_model_parameters.joblib'
encoding_parameters = processed_folder / 'OHE_parameters.joblib'

df = pd.read_csv(file_to_open)

### Select variable to predict as well as features to be used
y = df['Demand']
x = df[['Auteur_labels','Editeur_labels','Pays_labels','Document_type_labels']]

### Split dataset into training and testing components
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

### One hot encode categorical features
enc = OneHotEncoder(sparse=False)
enc.fit(x)
onehotlabels_train = enc.transform(x_train)
onehotlabels_test = enc.transform(x_test)
### Regress onehot encoded features on variable to predict, and test against
### test set
clf = SGDRegressor(max_iter=1000, tol=1e-3)
clf_model = clf.fit(onehotlabels_train,y_train)
R2 = clf_model.score(onehotlabels_test,y_test)

### Save model parameters
dump(clf_model, model_parameters)
dump(enc, encoding_parameters)
