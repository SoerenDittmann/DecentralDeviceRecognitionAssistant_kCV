#%% Importing Packages
import pandas as pd
import numpy as np
import time
from pickle import load
import glob
#import matplotlib.pyplot as plt
import copy
import scipy
#from sklearn.metrics import plot_confusion_matrix
#import seaborn as sn


import tensorflow
import winsound

#import git
import json
import os

#Import Sklearn Packages
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#Import Sktime
#from sktime.classification.compose import TimeSeriesForestClassifier
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier, ElasticEnsemble, ProximityForest
from sktime.classification.dictionary_based import BOSSEnsemble
from sktime.classification.hybrid import HIVECOTEV1
from sktime.classification.hybrid import HIVECOTEV2
from sktime.classification.sklearn._rotation_forest import RotationForest
from sktime.transformations.panel.summarize import RandomIntervalFeatureExtractor
from sktime.utils.slope_and_trend import _slope

#Import own Packages
from Classification.custom_classifiers.classifiers import RISErejectOption_entropy, custom_fbeta
from Classification.custom_classifiers.utils import build_sktime_data, calc_accuracy, data_stats
from Classification.data_handling.basics import read_in_data, handling_data, map_to_plaintext_labels
#from Classification.custom_classifiers.train_model import rise_training

#%% General
# Orga: Define ring
duration = 3000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)

# Technical: Define max. time series length
series_length = 1000 #Time series length for predictions

#%% 1. Load OS Data
##############################################################################

#%% 1.1GVL and Load OS Data

#First load spydata df into workspace: 20210530_Diss_Data_DataFile.spydata
#Required manual import
try:
    sensor_dic
except NameError:
    print("Please import OS Data")

#load working directory
fileDirectory = os.path.dirname("__file__")

#%% Prepare data

#%% Data preprocessing
#build Dataframe in correct format for sktime. Sensor length set to 1000 in build_sktime_data
keys_sorted, all_data = build_sktime_data(sensor_dic, series_length)
X, y = all_data['X'], all_data['y']


#%%Restructure database for individual binary class predictions

#Initiate dict with [sensorname]{[sensorname], [other]} as tag structure
database = {}

selected_keys = ['vibration_sensor']

#Iterate over all the available sensor types
for el in selected_keys:
    sensor_type = el
    #copy complete dict once again under tag for the sensor type
    database[sensor_type] = copy.deepcopy(sensor_dic)
    #Declare a list of sensor types that are stored under new tag other and can be removed
    del_list = []
    data = pd.DataFrame()

    
    for key in keys_sorted:    
        if (key != sensor_type):
            if 'other' not in database[sensor_type]:
                #case 1: add new key 'other' to dict
                database[sensor_type]['other'] = copy.deepcopy(database[sensor_type][key])
                #print('Sensortype initialisierung: '+sensor_type)
                #print('Key initialisierung: '+key)
                #for dataframes in database[sensor_type][key]:
                #    print('Df initialisierung: '+dataframes)

            else:
                #case 2: other is initialised and the dataframe needs to be joined to the existing df under other 
                for df in database[sensor_type][key]:
                    
                    if df in database[sensor_type]['other']:
                        
                        #print('Key if: '+key)
                        #print('Df if: '+df)
                            
                        data = database[sensor_type][key][df]
                        data = pd.DataFrame(data)
                        database[sensor_type]['other'][df] = pd.DataFrame(database[sensor_type]['other'][df]).join(data)
                        data = pd.DataFrame()
                    
                    #case 3: other initialized and dfs to be added not existing
                    
                    else:
                        database[sensor_type]['other'][df] = pd.DataFrame(database[sensor_type][key][df])
                        #print('Key else: '+key)
                        #print('Df else: '+df)
                        data_neu = database[sensor_type][key][df]
                        

            #fill del_list    
            del_list.append(key)

        
    #del keys that are now stores in other
    for e in del_list:
        database[sensor_type].pop(e, None)
        
#%%Define data base for repeated cv
#Implement HIVE COTE BASE for Vibration data
keys_sorted, all_data = build_sktime_data(database['vibration_sensor'], series_length)

X, y = all_data['X'], all_data['y']        



#%%HIVE COTE2 binary classification with reject option
##############################################################################


#%% Define Grid
#Extensive params grid
param_grid= {'stc_params': ({
        "estimator": RotationForest(n_estimators=3),
        "n_shapelet_samples": 500, 
        "max_shapelets": 20,
        "batch_size": 100,
    },
    {
        "estimator": RotationForest(n_estimators=2),
        "n_shapelet_samples": 100, 
        "max_shapelets": 5,
        "batch_size": 50,
    },
    {
        "estimator": RotationForest(n_estimators=2),
        "n_shapelet_samples": 300, 
        "max_shapelets": 10,
        "batch_size": 75,
    }),
    'drcif_params': ({"n_estimators": 10}, 
                     {"n_estimators": 5}, 
                     {"n_estimators": 8}),
    'arsenal_params': ({"num_kernels": 50, "n_estimators": 2},
                       {"num_kernels": 100, "n_estimators": 5},
                       {"num_kernels": 75, "n_estimators": 10}),
    'tde_params': ({
        "n_parameter_samples": 25,
        "max_ensemble_size": 5,
        "randomly_selected_params": 10
    },
    {
        "n_parameter_samples": 50,
        "max_ensemble_size": 10,
        "randomly_selected_params": 20
    },
    {
        "n_parameter_samples": 12,
        "max_ensemble_size": 3,
        "randomly_selected_params": 5,
    }),
    'threshold': [0, 0.2, 0.4, 0.6, 0.8]    
    }

#%% Define HIVE COTE 2.0 estimator with a rejection option
#For that the model incl. some functions need to be overwritten
#See e.g. https://scikit-learn.org/stable/developers/develop.html#parameters-and-init
#Accessed 25.10.2023

#overwrite HIVE COTE 2.0 class
class HC2withReject(HIVECOTEV2):
    def __init__(
        self,
        threshold=0.5,
        stc_params=None,
        drcif_params=None,
        arsenal_params=None,
        tde_params=None,
        time_limit_in_minutes=0,
        save_component_probas=False,
        verbose=0,
        n_jobs=1,
        random_state=None,
        **kwargs
    ):
        super().__init__(
            stc_params=stc_params,
            drcif_params=drcif_params,
            arsenal_params=arsenal_params,
            tde_params=tde_params,
            time_limit_in_minutes=time_limit_in_minutes,
            save_component_probas=save_component_probas,
            verbose=verbose,
            n_jobs=n_jobs,
            random_state=random_state,
            **kwargs
        )
        self.threshold = threshold


    
    #define customized predict class für hc2 classifier
    def predict(self, X):
        predictions = super().predict(X)
        #assign predictions lower than threshold wit class 9 (rejected)
        predictions = [9 if pred < self.threshold else pred for pred in predictions]
        return predictions
    
    #define get params for troubleshoot
    def get_params(self, deep=True):
        params = super().get_params()
        params.update({
            'threshold': self.threshold,
            'stc_params': self.stc_params,
            'drcif_params': self.drcif_params,
            'arsenal_params': self.arsenal_params,
            'tde_params': self.tde_params,
            'time_limit_in_minutes': self.time_limit_in_minutes,
            'save_component_probas': self.save_component_probas,
            'verbose': self.verbose,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state
        })
        return params
    
#%% Define customized scorer with subsequent GridSearchCV to account for rejections

def custom_precision (y_true, y_pred, threshold=0.2):
    predictions_incl_rej = [9 if pred < threshold else pred for pred in y_pred]
    #Regard only predictions that were not rejected to avoid a bias towards 
    #rejections
    filtered_indices_scorer = [i for i, pred in enumerate(predictions_incl_rej) if pred != 9]
    y_true_filtered_scorer = [y_true[i] for i in filtered_indices_scorer]
    y_pred_filtered_scorer = [y_pred[i] for i in filtered_indices_scorer]   
    return precision_score(y_true_filtered_scorer, y_pred_filtered_scorer, labels=[0,1], pos_label=1)

#Define customized scorer
custom_scorer = make_scorer(custom_precision, greater_is_better=True)

#%%Define Nested Cross Validation
#HIVE COTE2 reject option 

#generate random integers to fix random_seed for each model in the inner loop of the nested CV
n_outer_splits = 3
n_inner_splits = 3


#------------------------Nested-CV HIVECOTE V2-----------------------------------------------

#reset iteration variable i
i = 0

# configure the cross-validation procedure
cv_outer = StratifiedKFold(n_splits=n_outer_splits, shuffle=True, random_state=1)

# define lists to store results
res_cm = []
res_acc = []
res_precision = []
res_rejection_ratio = []
res_best_model = []
res_best_params = []
res_y_pred = []
res_y_test = []
res_y_true_filtered_outer = []
res_y_pred_filtered_outer = []


for train_ix, test_ix in cv_outer.split(X,y):
    # split data
    X_train, X_test = all_data['X'][train_ix], all_data['X'][test_ix]
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train, y_test = all_data['y'][train_ix], all_data['y'][test_ix]
    # configure the cross-validation procedure
    cv_inner = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=1)
    # define the model
    model = HC2withReject(random_state=123, time_limit_in_minutes=5)
    # search space defined above
    # define search
    search = GridSearchCV(model, param_grid, scoring=custom_scorer, cv=cv_inner, refit=True)
    # execute search
    result = search.fit(X_train, y_train)
    # get the best performing model fit on the whole training set
    best_model = result.best_estimator_
    y_pred = best_model.predict(X_test)
    
    #Account for rejected predictions
    filtered_indices_outer = [i for i, pred in enumerate(y_pred) if pred != 9]
    y_true_filtered_outer = [y_test[i] for i in filtered_indices_outer]
    y_pred_filtered_outer = [y_pred[i] for i in filtered_indices_outer]
    
    #Store data results from outer folds
    res_y_pred.append(y_pred)
    res_y_test.append(y_test)
    res_y_true_filtered_outer.append(y_true_filtered_outer)
    res_y_pred_filtered_outer.append(y_pred_filtered_outer)
    
    
    
    #Store results
    #In the confusion matrix, we want to see the complete pred and complete test
    res_cm.append(confusion_matrix(y_test, y_pred, labels=[0, 1, 9]))
    '''
    Below once again the assumptions that rejected predictions do not count
    into reported acc or precision
    '''
    res_acc.append(accuracy_score(y_true_filtered_outer, y_pred_filtered_outer))
    res_precision.append(precision_score(y_true_filtered_outer, y_pred_filtered_outer, labels=[0, 1], pos_label=1))
    res_rejection_ratio.append(y_pred.count(9) / len(y_pred))
    res_best_model.append(best_model)
    res_best_params.append(result.best_params_)
    
    #report progress
    print(i)
    #iterate
    i += 1

#notify about end of Nested CV
winsound.Beep(freq, duration)
#%% TODO: Ich muss im custom scorer genauso wie unten die rejected klasses
#ignorieren, sonst habe ich einen Bias bzgl. rejection

'''
DELETE
filtered_indices = [i for i, pred in enumerate(res_y_pred[0]) if pred != 9]
y_true_filtered_test = [res_y_true_filtered[0][i] for i in filtered_indices]
y_pred_filtered_test = [res_y_pred[0][i] for i in filtered_indices]

# Calculate the accuracy score for the filtered lists
accuracy = accuracy_score(y_true_filtered_test, y_pred_filtered_test)
'''
