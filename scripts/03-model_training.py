from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
import numpy as np
import pandas as pd
from pathlib import Path
import datetime
import logging
import matplotlib.pyplot as plt

def major_classes_undersampler(data, variable, class_to_match):
    '''(DataFrame, str, int) -> DataFrame
    Accepts a dataframe as an argument, and a variable name that will be used
    to reduce the size of the dataframe by undersampling majority class data'''
    minority_class_len = len(data[data[variable]==class_to_match])
    logging.debug("Demand classes: {}".format(data.groupby(variable)[variable].count()))
    ### classes_to_undersample should be used with caution, it assumes that indices
    ### map directly to demand categories, which is true for my data, but my not be for all
    classes_to_undersample = np.where(data.groupby(variable)[variable].count()>minority_class_len)[0]
    logging.debug("Classes to undersample: {}".format(classes_to_undersample))
    indices_to_keep = []
    for i in classes_to_undersample:
        indices_to_keep.append(np.random.choice(data[data['Demand']==i].index.tolist(),
            minority_class_len, replace = False))
    minority_class_indices = data[data['Demand']>=class_to_match].index
    under_sample_indices = np.concatenate([minority_class_indices,indices_to_keep[0],indices_to_keep[1]])
    logging.debug("indices to keep: {}".format(under_sample_indices))
    return data.loc[under_sample_indices]

def train_model(data, y_var, x_vars):
    ''' (dataframe, str, [str]) -> model
    Returns a fitted random forest model from an input dataframe'''
    y = data[y_var]
    x = data[x_vars]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    regr = RandomForestClassifier(max_depth=16, min_samples_leaf=8, bootstrap = False)
    regr_model = regr.fit(x_train, y_train)
    return regr_model, x_test, y_test

def model_accuracy(regr, x_test, y_test):
    '''(model, array, array) -> None
    Takes in a fitted classification model and prints output reports directly
    to the terminal.
    '''
    y_pred = regr.predict(x_test)
    print("Model score: {}".format(regr_1.score(x_test,y_test)))
    conf_mat = confusion_matrix(y_test, y_pred)
    print("Confusion matrix: {}".format(conf_mat))
    s = []
    for i in range(len(conf_mat)):
        if i == 0:
            s.append(conf_mat[i,i]+conf_mat[i,i+1])
        elif i == conf_mat.shape[0]-1:
            s.append(conf_mat[i,i]+conf_mat[i, i-1])
        else:
            s.append(conf_mat[i,i]+conf_mat[i,i-1]+conf_mat[i,i+1])
    total_sum = sum(sum(conf_mat))
    print('Full classification report: {}'.format(classification_report(y_test,y_pred)))
    ### Straight up precision just says how many items were exactly right
    ### Confusion matrix shows many items are misclassified as closely related values
    ### (i.e. 4 as 3 or 5, which is an acceptable margin of error for my problem).
    print('Accuracy of predctions in the neighbourhood: {}'.format(sum(s)/total_sum))
    print('Feature importance for forest model: {}'.format(regr_1.feature_importances_))
    conf_mat_reduced = []

def random_accuracy(regr,y_test):
    '''(model, array, array) -> None
    Takes in a fitted classification model and prints output accuracy reports for
    random selection of y_test items.
    '''
    total_sum = len(y_test)
    random_prob_test_array = []
    for i in y_test:
        random_prob_test_array.append(np.random.choice(y_test))
    random_prob_test_array = np.array(random_prob_test_array)
    count = len(np.where(random_prob_test_array==y_test)[0])
    print('Accuracy if the model was predicting at random based on available ratios: {}'.format(count/total_sum))
    print('Full classification report for random data: {}'.format(classification_report(y_test,random_prob_test_array)))
    expanded_count = []
    true_vals = np.array(y_test)
    for i in range(random_prob_test_array.shape[0]):
        if true_vals[i] == random_prob_test_array[i]:
            expanded_count.append(random_prob_test_array[i])
        elif true_vals[i] == random_prob_test_array[i] + 1:
            expanded_count.append(random_prob_test_array[i])
        elif true_vals[i] == random_prob_test_array[i] + -1:
            expanded_count.append(random_prob_test_array[i])
    print('Accuracy if the model was predicting at random (expanded ): {}'.format(len(expanded_count)/total_sum))

### Classification of demand based on historical data
if __name__ == "__main__":

    logging.basicConfig(level=logging.CRITICAL,format='%(asctime)s - %(levelname)s - %(message)s')
    np.random.seed(31415)

    ### Determine paths to datasets and load into pandas dataframe
    processed_folder = Path("../data/processed")
    file_to_open = processed_folder / 'feature_labels_for_each_title.csv'
    model_parameters = processed_folder / 'decision_tree_parameters.joblib'

    ### Loading response dataset
    df = pd.read_csv(file_to_open)
    df = major_classes_undersampler(df,'Demand',2)
    ### Training model
    predictor_variables = ['Auteur_labels', 'Editeur_labels', 'Pays_labels',
           'Years_offset', 'Document_type_labels', 'Language_type_labels']
    regr_1, x_test, y_test = train_model(df,'Demand',predictor_variables)

    ### Assess model accuracy
    model_accuracy(regr_1,x_test, y_test)
    ### Confirm that the model is doing better than random
    random_accuracy(regr_1,y_test)

    plt.figure()
    plt.bar([0,1,2,3,4,5],regr_1.feature_importances_)
    plt.savefig('../feature_importances.png')

    ### Save output of model

    dump(regr_1, model_parameters)
