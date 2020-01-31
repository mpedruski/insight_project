from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from joblib import dump, load
import numpy as np
import pandas as pd
from pathlib import Path
import datetime
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s')
np.random.seed(31415)

### Regression based on historical data

### Determine paths to datasets and load into pandas dataframe
processed_folder = Path("../data/processed")
file_to_open = processed_folder / 'cluster_labels_for_each_title.csv'
coded_variables_file = processed_folder / 'coded_variables.joblib'
model_parameters = processed_folder / 'decision_tree_parameters.joblib'

### Loading response dataset
df = pd.read_csv(file_to_open)
### Loading codings of author, publisher, and country based on lifetime borrows
var = load(coded_variables_file)
df['Auteur_values'] = var[0]
df['Editeur_values'] = var[1]
df['Pays_values'] = var[2]

### Find length of majority and minority class data
minority_class_len = len(df[df['Demand']==2])
majority_class_1_indices = df[df['Demand']==0].index
majority_class__2_indices = df[df['Demand']==1].index

### Generate a list of majority class indices to retain, based on how many minority
### class data their are
random_majority_1_indices = np.random.choice(majority_class_1_indices, minority_class_len,
    replace = False)
random_majority_2_indices = np.random.choice(majority_class__2_indices, minority_class_len,
        replace = False)
### Make list of minority class indices and combine with majority class indices
### then define dataset as the rows of the dataset from those indices
minority_class_indices = df[df['Demand']>1].index
under_sample_indices = np.concatenate([minority_class_indices,random_majority_1_indices,
    random_majority_2_indices])
df = df.loc[under_sample_indices]

### Select variable to predict as well as features to be used
y = df['Demand']
x = df[['Auteur_values','Editeur_values','Pays_values','Document_type_labels',
    'Years_offset','Language_type_labels']]
### Split dataset into training and testing components
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

## Classify features on variable to predict, and test against
### test set
regr_1 = RandomForestClassifier(max_depth=16, min_samples_leaf=8, bootstrap = False)
regr_1_model = regr_1.fit(x_train, y_train)
y_pred = regr_1.predict(x_test)

### Validate the model
### Straight up precision just says how many items were exactly right
print(regr_1.score(x_test,y_test))
### Confusion matrix shows many items are misclassified as closely related values
### (i.e. 4 as 3 or 5, which is an acceptable margin of error for my problem).
### Accounting for this wider margin, the accuracy is close to 0.7
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)
s = []
print(conf_mat.shape)
for i in range(len(conf_mat)):
    if i == 0:
        s.append(conf_mat[i,i]+conf_mat[i,i+1])
    elif i == conf_mat.shape[0]-1:
        s.append(conf_mat[i,i]+conf_mat[i, i-1])
    else:
        s.append(conf_mat[i,i]+conf_mat[i,i-1]+conf_mat[i,i+1])
total_sum = sum(sum(conf_mat))
logging.debug('Accuracy of predctions in the neighbourhood: {}'.format(sum(s)/total_sum))

logging.debug('List of possible classes: {}'.format(regr_1.classes_))
logging.debug('Feature importance for forest model: {}'.format(regr_1.feature_importances_))

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

### Save output of model

dump(regr_1, model_parameters)
