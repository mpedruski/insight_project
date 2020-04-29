from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from joblib import dump, load
import numpy as np
import pandas as pd
from pathlib import Path
import datetime
import logging
import seaborn as sns
import matplotlib.pyplot as plt

def major_classes_undersampler(data, class_to_match):
    '''(Pandas_series, str, int) -> DataFrame
    Accepts a Pandas Series as an argument, and returns indices to undersamples the
    dataframe so that no demand class is greater than the class_to_match'''
    minority_class_len = len(data[data==class_to_match])
    classes_to_undersample, classes_to_retain_intact = [],[]
    for i in np.unique(data):
        if len(data[data==i])> minority_class_len:
            classes_to_undersample.append(i)
        if len(data[data==i]) <= minority_class_len:
            classes_to_retain_intact.append(i)
    logging.debug("Classes to undersample: {}".format(classes_to_undersample))
    logging.debug("Classes to retain: {}".format(classes_to_retain_intact))
    indices_to_keep = []
    for i in classes_to_undersample:
        indices_to_keep.append(np.random.choice(data[data==i].index,
            minority_class_len, replace = False))
    for i in classes_to_retain_intact:
        indices_to_keep.append(data[data==i].index)
    indices_to_keep = np.concatenate(indices_to_keep)
    logging.debug("indices to keep: {}".format(indices_to_keep))
    return indices_to_keep

def data_split(data,split):
    '''(dataframe, float) -> (training dataframe, test dataframe)
    Accepts as arguments a dataframe to be split into training and test components,
    and a decimal that reflects what proportion of the data will be allocated
    to the training dataframe. Both train and test dataframes are returned.'''

    msk = np.full(len(data),False)
    test_ind = np.random.choice(np.arange(0,len(data)),int(len(data)*split),replace=False)
    msk[test_ind] = True
    train = data[msk]
    test = data[~msk]
    return train, test

def cross_validate(x_train, y_train):
    ''' (dataframe,dataframe)
    Prints model scores for models fitted on different levels of undersampled data
    after accepting predictor data (x_train) and response data (y_train)'''
    print("Original length of data: {}".format(len(y_train)))
    ### Compare score when matching class 0 (no undersampling), 1, or 2
    for i in [0,1,2]:
        ind_to_keep = np.sort(major_classes_undersampler(y_train,i))
        print("Number of data points retained: {}".format(len(ind_to_keep)))
        logging.debug(ind_to_keep)
        y_train_red = y_train[ind_to_keep]
        logging.debug(y_train)
        x_train_red = x_train[x_train['ID'].isin(ind_to_keep)]
        logging.debug(x_train)
        x_train_red = x_train_red.drop('ID',axis=1)
        logging.debug(x_train_red)
        rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
        rf.fit(x_train_red, y_train_red)
        print("Model R2, undersampled to {},: {}".format(i,rf.score(x_cv.drop('ID',axis=1),y_cv)))

def train_model(x_train, y_train):
    ''' (dataframe, dataframe) -> model
    Returns a fitted random forest model from input training dataframes'''
    x_train = x_train.drop('ID',axis=1)
    print(x_train.head(3))
    rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
    rf.fit(x_train, y_train)
    return rf

def model_accuracy(rf_model, x_test, y_test):
    '''(model, array, array) -> None
    Takes in a fitted classification model and prints output reports directly
    to the terminal.
    '''
    x_test = x_test.drop('ID',axis =1)
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

def current_accuracy_test_data(x_test_demand, y_test):
    '''([int],[int]) -> None
    Accepts a list of supply for titles (x_test_demand) and a list of actual
    demand for these titles (y_test), and returns measures of goodness of fit
    for a linear model using demand ~ supply (assesses ability of supply to
    predict demand for the books in the test set)'''
    lm = linear_model.LinearRegression()
    lm.fit(np.array(x_test_demand).reshape(-1,1),y_test)
    print('R2 of linear model between supply and demand (test set): {}'.format(lm.score(np.array(x_test_demand).reshape(-1,1),y_test)))
    print("nRMSE of linear model between supply and demand (test set): {}".format(mean_squared_error(x_test_demand, y_test, squared = False)/np.mean(y_test)))
    sns.set()
    sns.scatterplot(x_test_demand, y_test)
    plt.plot([0, 90], [0,90], linewidth=2)
    plt.savefig('../current_demand_supply.png')

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

def global_accuracy(rf_model):
    '''Takes in raw data as well as the fitted model to show model predictions
    for the entire catalogue'''
    df = pd.read_csv(file_to_open)
    ### Make predictions for all data points in catalogue
    predictor_variables = ['Auteur_labels', 'Editeur_labels', 'Pays_labels',
       'Years_offset', 'Document_type_labels', 'Language_type_labels']
    x = df[predictor_variables]
    y_true = df['Demand']
    y_pred = rf_model.predict(x)
    rounded = [round(i) for i in y_pred]
    sns.set()
    sns.scatterplot(rounded, y_true)
    plt.plot([0, 200], [0,200], linewidth=2)
    plt.savefig('../global_regression_forest_scatterplot.png')

### Classification of demand based on historical data
if __name__ == "__main__":

    logging.basicConfig(level=logging.CRITICAL,format='%(asctime)s - %(levelname)s - %(message)s')
    np.random.seed(31415)

    ### Determine paths to datasets and load into pandas dataframe
    processed_folder = Path("../data/processed")
    file_to_open = processed_folder / 'expanded_feature_labels_for_each_title.csv'
    model_parameters = processed_folder / 'expanded_regression_tree_parameters_post_cv.joblib'

    ### Loading response dataset
    df = pd.read_csv(file_to_open)
    df.columns = ['ID','Auteur_labels', 'Editeur_labels', 'Pays_labels',
               'Years_offset', 'Document_type_labels', 'Language_type_labels',
               'Total','Demand']

    ### Split model into train and test sets, then split train data into train and CV
    ### Training model (see note in train_model about use of 'Total')
    predictor_variables = ['ID','Auteur_labels', 'Editeur_labels', 'Pays_labels',
           'Years_offset', 'Document_type_labels', 'Language_type_labels']
    train_cv, test = data_split(df,0.7)
    train, cv = data_split(train_cv,0.8)
    x_train_cv, x_train, x_cv, x_test = train_cv[predictor_variables], train[predictor_variables], cv[predictor_variables], test[predictor_variables]
    y_train_cv, y_train, y_cv, y_test = train_cv['Demand'], train['Demand'],cv['Demand'],test['Demand']

    ### Cross validate for different levels of undersampling
    # cross_validate(x_train,y_train)
    ### No undersampling gives highest CV score, so train model without undersampling
    rf = train_model(x_train_cv,y_train_cv)

    ### Assess model accuracy
    model_accuracy(rf,x_test, y_test)

    ### Assess accuracy of demand vs supply
    x_test_demand = test['Total']
    current_accuracy_test_data(x_test_demand,y_test)
    model_test_cases(rf)
    global_accuracy(rf)

    print(rf.feature_importances_)

    sns.set()
    plt.figure()
    # plt.bar([0,1,2,3,4,5],np.sort(rf.feature_importances_)[::-1])
    plt.bar([0,1,2,3,4,5],rf.feature_importances_)
    plt.savefig('../regression_tree_feature_importances.png')

    ### Save output of model

    # dump(rf, model_parameters)
