from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
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
    # ### To compare how model predictions improve (or not) on the current relationship
    # ### between supply and demand I include Total demand in the predictor variables
    # ### so that it will be partitioned by train_test_split, but it isn't inteded
    # ### as a model feature. I remove it from the x_train and x_test sets before
    # ### model fitting, but return the Total attributed to test cases, so it can
    # ### be passed to model validation as a predictor to compare against my model.
    x_test_demand = x_test[x_vars[-1]]
    x_train = x_train[x_vars[0:-1]]
    x_test = x_test[x_vars[0:-1]]
    rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
    rf_model = rf.fit(x_train, y_train)
    return rf_model, x_test, y_test, x_test_demand

def model_accuracy(rf_model, x_test, y_test):
    '''(model, array, array) -> None
    Takes in a fitted classification model and prints output reports directly
    to the terminal.
    '''
    y_pred = rf_model.predict(x_test)
    print("Model R2: {}".format(rf_model.score(x_test,y_test)))
    print("Model RMSE: {}".format(mean_squared_error(y_test, y_pred)))
    rounded = [round(i) for i in y_pred]
    r2 = 1 - ((y_test - rounded) ** 2).sum()/((y_test - y_test.mean()) ** 2).sum()
    print("Rounded R2: {}.".format(r2))
    print("Rounded model NRMSE: {}".format(mean_squared_error(y_test, rounded, squared = False)/np.mean(y_test)))
    sns.set()
    sns.scatterplot(rounded, y_test)
    plt.plot([0, 90], [0,90], linewidth=2)
    plt.savefig('../test_regression_forest_scatterplot.png')

def current_accuracy(x_test_demand, y_test):
    print(len(x_test_demand))
    print(len(y_test))
    curr_r2 = 1 - ((y_test - x_test_demand) ** 2).sum()/((y_test - y_test.mean()) ** 2).sum()
    print("R2 of current collection supply vs. demand: {}.".format(curr_r2))
    print("Current NRMSE: {}".format(mean_squared_error(y_test, x_test_demand, squared = False)/np.mean(y_test)))

def model_test_cases(rf_model):
    '''(model, array, array) -> None
    Takes in a fitted classification model and prints output for hard coded test cases.
    '''
    print("JK Rowling, Scholastic, US, LVJ, Eng:{}".format(rf_model.predict(np.array([23401,
        680251, 2378036, 0, 14010164, 3222549]).reshape(1,-1))))
    print("Marcel Proust, Gallimard, France, LVA, Fre:{}".format(rf_model.predict(np.array([2878,
        525805, 17528526, 0, 8752779, 27562080]).reshape(1,-1))))
    print("Michael Pedruski, Late Night, Can, NFA, Eng:{}".format(rf_model.predict(np.array([0,
        0, 996559, 0, 4218963, 3222549]).reshape(1,-1))))

### Classification of demand based on historical data
if __name__ == "__main__":

    logging.basicConfig(level=logging.CRITICAL,format='%(asctime)s - %(levelname)s - %(message)s')
    np.random.seed(31415)

    ### Determine paths to datasets and load into pandas dataframe
    processed_folder = Path("../data/processed")
    file_to_open = processed_folder / 'expanded_feature_labels_for_each_title.csv'
    model_parameters = processed_folder / 'expanded_regression_tree_parameters.joblib'

    ### Loading response dataset
    df = pd.read_csv(file_to_open)
    df = major_classes_undersampler(df,'Demand',2)
    ### Training model (see note in train_model about use of 'Total')
    predictor_variables = ['Auteur_labels', 'Editeur_labels', 'Pays_labels',
           'Years_offset', 'Document_type_labels', 'Language_type_labels','Total']
    logging.debug(predictor_variables)
    rf_model, x_test, y_test, x_test_demand = train_model(df,'Demand',predictor_variables)

    ### Assess model accuracy
    model_accuracy(rf_model,x_test, y_test)
    current_accuracy(x_test_demand,y_test)
    model_test_cases(rf_model)

    plt.figure()
    plt.bar([0,1,2,3,4,5],rf_model.feature_importances_)
    plt.savefig('../regression_tree_feature_importances.png')

    ### Save output of model

    # dump(rf_model, model_parameters)
