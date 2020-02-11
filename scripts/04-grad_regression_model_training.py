from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from joblib import dump, load
import numpy as np
import pandas as pd
from pathlib import Path
import datetime
import logging
import seaborn as sns
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
    rf = GradientBoostingRegressor(n_estimators = 1000, random_state = 42, learning_rate=0.1,)
    rf_model = rf.fit(x_train, y_train)
    return rf_model, x_test, y_test

def model_accuracy(rf_model, x_test, y_test):
    '''(model, array, array) -> None
    Takes in a fitted classification model and prints output reports directly
    to the terminal.
    '''
    y_pred = rf_model.predict(x_test)
    print("Model R2: {}".format(rf_model.score(x_test,y_test)))
    print("Model RMSE: {}".format(mean_squared_error(y_test, y_pred)))
    sns.scatterplot(y_test,y_pred)
    plt.savefig('../test_regression_forest_scatterplot.png')

### Classification of demand based on historical data
if __name__ == "__main__":

    logging.basicConfig(level=logging.CRITICAL,format='%(asctime)s - %(levelname)s - %(message)s')
    np.random.seed(31415)

    ### Determine paths to datasets and load into pandas dataframe
    processed_folder = Path("../data/processed")
    file_to_open = processed_folder / 'feature_labels_for_each_title.csv'
    model_parameters = processed_folder / 'grad_boost_parameters.joblib'

    ### Loading response dataset
    df = pd.read_csv(file_to_open)
    df = major_classes_undersampler(df,'Demand',2)
    ### Training model
    predictor_variables = ['Auteur_labels', 'Editeur_labels', 'Pays_labels',
           'Years_offset', 'Document_type_labels', 'Language_type_labels']
    gb_model, x_test, y_test = train_model(df,'Demand',predictor_variables)

    ### Assess model accuracy
    model_accuracy(gb_model,x_test, y_test)

    plt.figure()
    plt.bar([0,1,2,3,4,5],gb_model.feature_importances_)
    plt.savefig('../regression_tree_feature_importances.png')

    ### Save output of model

    # dump(rf_model, model_parameters)
