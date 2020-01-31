from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from joblib import dump
import numpy as np
import pandas as pd
from pathlib import Path
import datetime
import logging

np.random.seed(31415)
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
x = df[['Auteur_labels','Editeur_labels','Pays_labels','Document_type_labels']]

### Split dataset into training and testing components
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

### One hot encode categorical features
enc = OneHotEncoder(sparse=False)
enc.fit(x[['Auteur_labels','Editeur_labels','Pays_labels','Document_type_labels']])
onehotlabels_train = enc.transform(x_train[['Auteur_labels','Editeur_labels','Pays_labels','Document_type_labels']])
onehotlabels_test = enc.transform(x_test[['Auteur_labels','Editeur_labels','Pays_labels','Document_type_labels']])
# logging.debug('Numeric features {}'.format(onehotlabels_train))
n_classes = len(np.unique(df['Demand'].values))

### Regress onehot encoded features on variable to predict, and test against
### test set
clf = SGDClassifier(max_iter=10000, tol=1e-3)
clf_model = clf.fit(onehotlabels_train,y_train)
R2 = clf_model.score(onehotlabels_test,y_test)
print(R2)
y_pred = clf.predict(onehotlabels_test)
conf_mat = confusion_matrix(y_test, y_pred, labels = range(n_classes))

s = []
print(conf_mat.shape)
for i in range(len(conf_mat)):
    if i == 0:
        s.append(conf_mat[i,i]+conf_mat[i,i+1])
    elif i == conf_mat.shape[0]-1:
        s.append(conf_mat[i,i]+conf_mat[i, i-1])
    else:
        s.append(conf_mat[i,i]+conf_mat[i,i-1]+conf_mat[i,i+1])
logging.debug('How many predictions are in neighbourhood: {}'.format(s))
total_sum = sum(sum(conf_mat))
logging.debug('Accuracy of predctions in the neighbourhood: {}'.format(sum(s)/total_sum))
logging.debug('Number of actual 0 cases by value {}'.format(sum(conf_mat[0,:])))
logging.debug('Number of predicted 0 cases by value {}'.format(sum(conf_mat[:,0])))
ind_sums = np.sum(conf_mat,axis=1)
logging.debug('Number of prediced cases predicted to be x {}'.format(ind_sums))


print(clf.classes_)
# print(clf.feature_importances_)

### Confirm that the model is doing better than random
random_prob_test_array = []
for i in y_test:
    random_prob_test_array.append(np.random.choice(y_test))
random_prob_test_array = np.array(random_prob_test_array)
count = len(np.where(random_prob_test_array==y_test)[0])
logging.debug('Accuracy if the model was predicting at random based on available ratios: {}'.format(count/total_sum))
expanded_count = []
true_vals = np.array(y_test)
print(true_vals)
for i in range(random_prob_test_array.shape[0]):
    # print(i)
    if true_vals[i] == random_prob_test_array[i]:
        expanded_count.append(random_prob_test_array[i])
    elif true_vals[i] == random_prob_test_array[i] + 1:
        expanded_count.append(random_prob_test_array[i])
    elif true_vals[i] == random_prob_test_array[i] + -1:
        expanded_count.append(random_prob_test_array[i])

logging.debug('Accuracy if the model was predicting at random (expanded ): {}'.format(len(expanded_count)/total_sum))

# print(clf.predict(enc.transform(np.array([0,0,0,0]).reshape(1,-1))))
# print(clf.predict(np.array([[0,0,0,0],[1,0,1,0]])))

### Save model parameters
dump(clf_model, model_parameters)
dump(enc, encoding_parameters)
