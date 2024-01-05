'''
Define own k-fold cross validation function to assess model stability.
Code snippets for measure_KPIs and seeding of the repeated CV based on: 
Mathieu. M.-P., Enabling scalability in digital twin data acquisition: 
A machine learning driven device recognition assistant
Cited in thesis as:[Mat-21]
'''

#%%Import packages
import pandas as pd
import numpy as np
import time
from pickle import load
import glob
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import copy
import scipy
#from sklearn.metrics import plot_confusion_matrix
#import seaborn as sn
#import tensorflow
import winsound

#Import Sklearn Packages
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score
#Evaluation packages
from sklearn.model_selection import RepeatedStratifiedKFold

#Import Sktime
#from sktime.classification.compose import TimeSeriesForestClassifier
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier, ElasticEnsemble, ProximityForest
from sktime.classification.dictionary_based import BOSSEnsemble
from sktime.classification.hybrid import HIVECOTEV2
from sktime.transformations.panel.shapelet_transform import (RandomShapeletTransform)
from sktime.transformations.panel.summarize import RandomIntervalFeatureExtractor
from sktime.classification.sklearn._rotation_forest import RotationForest
from sktime.utils.slope_and_trend import _slope

#Import own Packages
from Classification.custom_classifiers.classifiers import RISErejectOption_entropy, custom_fbeta
from Classification.custom_classifiers.utils import build_sktime_data, calc_accuracy, data_stats
#from Classification.data_handling.basics import read_in_data, handling_data, map_to_plaintext_labels



#%% Prepare data bases for binary class predictions

#Required manual import of sensor dic
try:
    sensor_dic
except NameError:
    print("Please import OS Data")
    
#%% GVL
series_length = 1000


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

#%% Define best model from nested kCV

hc2 = HIVECOTEV2(
    stc_params={
        "estimator": RotationForest(n_estimators=2),
        "n_shapelet_samples": 100, 
        "max_shapelets": 5,
        "batch_size": 50,
    },
    drcif_params={"n_estimators": 10},
    arsenal_params={"num_kernels": 100, "n_estimators": 5},
    tde_params={
        "n_parameter_samples": 12,
        "max_ensemble_size": 3,
        "randomly_selected_params": 5,
    },
    random_state=42,
    time_limit_in_minutes=5
    )





#%% def function to calc metrics for repeated cv below

def measure_KPIs(classifier, X_train, y_train, X_test, y_test):
        
        #train & predict
        classifier.fit(X_train, y_train)
        preds = classifier.predict(X_test)
        
        #calculate cm, accuracy and precision
        cm = confusion_matrix(y_test, preds)
        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average='macro')
                
        model_name = str(classifier)
        
        return cm, acc, precision, model_name, preds



        
#%%HIVE-COTE BASE VIBRATION kCV

n_splits = 3
n_repeats = 3
number_of_algorithms = 1


res_cm = []
res_acc = []
res_precision = []
res_best_model = []
res_best_params = []
res_y_pred = []
res_y_test = []

rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state = 42)

#produce series of random but reproducible seeds to use during loops
rng = np.random.default_rng(seed=42)
loop_random_seeds = rng.integers(low=0, high=1000000, size = n_splits*n_repeats*number_of_algorithms)  


#Define loop for repeated kCV
i = 0

for train_index, test_index in rskf.split(X, y):

    #Define storage location for split results
    results_table = []
    
    #Define data for split
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    #Convert to DataFrame
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    
    
    #Train model and calc metrics
    cm, acc, precision, model_name, preds = measure_KPIs(hc2, X_train, y_train, X_test, y_test)
    
    res_cm.append(cm)
    res_acc.append(acc)
    res_precision.append(precision)
    res_y_pred.append(preds)
    res_y_test.append(y_test)
    
    i = i+1

#------End of for-loop--------


'''
Results of the base HIVE COTE model over repeated cv
Mean precision: 85%
Mean accuracy: 87%


'''

#%% Create exemplary plot 

y_test_plot = res_y_test[8]
y_pred_plot = res_y_pred[8]


#Merge y_test (real class) and y_pred (predicted class)
y_test_plot = copy.deepcopy(y_test_plot)
y_test_plot = pd.DataFrame(y_test_plot)
y_pred_plot = pd.DataFrame(y_pred_plot)


y_pred_plot= y_pred_plot.assign(real=y_test_plot["y"].values)

#Res is the df storing all cases, where algorithm predicted 1 but true class is 0
res_rows_pred1_rl0 = y_pred_plot.loc[(y_pred_plot[0] == 1) & (y_pred_plot['real'] == 0)]
res_rows_pred0_rl1 = y_pred_plot.loc[(y_pred_plot[0] == 0) & (y_pred_plot['real'] == 1)]

#Get list of inidizes in X and y based on the comparison above
res_rows_pred1_rl0_Xy = test_index[res_rows_pred1_rl0.index]
res_rows_pred0_rl1_Xy = test_index[res_rows_pred0_rl1.index]



#%% Image of the results

#x = np.arange(1,801)
fig, ax = plt.subplots(nrows=89, ncols=1, figsize=(40, 80))


#NOTE: Highlighted misclassified time series:
#blue: Train data
#green: Correctly predicted Test Data
#red: Test Data: 1 pred, but act 0. Index 48,50,53
#orange: Test Data: 0 pred, but act 1 Index


for i in range(0, X.shape[0]):
    
    if i in train_index:
        color = 'blue'
        #print(i)
    elif i in res_rows_pred0_rl1_Xy:
        color = 'orange'
    elif i in res_rows_pred1_rl0_Xy:
        color = 'red'
    else: #test data
        color = 'green'
    #Plot only chosen examples:
        #31/32 right
        #63/73 false positive
        #86/64 not recognized
    if i in [31,32,86,64,63,73]:
        ax[i].plot(X[i], color = color, label = [X[i].name, y[i], i])
        ax[i].legend(loc="upper right")

plt.show() 



#Smoothening for better Plot quality
def smoothing_ts(ts, window_size=8):
    return ts.rolling(window=window_size).mean()

smoothed_1 = smoothing_ts(X[31])
smoothed_2 = smoothing_ts(X[32])

smoothed_3 = smoothing_ts(X[64])
smoothed_4 = smoothing_ts(X[86])

smoothed_5 = smoothing_ts(X[88])
smoothed_6 = smoothing_ts(X[63])





#%% Reorder plot
plt.rcParams['font.family'] ='serif'

DIN_width = 8.27
DIN_hight = 11.69

plot_hight = DIN_hight*0.7

ratio = DIN_width / plot_hight

fig, ax = plt.subplots(nrows=6, ncols=1, figsize=(DIN_width, plot_hight), gridspec_kw={'height_ratios':[1]*6})
plt.tight_layout()

ax[0].plot(smoothed_1, color = '#6C8EBF', label = ['Position: High storage system'])
ax[0].legend(loc="upper right")
ax[1].plot(smoothed_2, color = '#82B366', label = ['Position: High storage system'])
ax[1].legend(loc="upper right")


ax[2].plot(smoothed_3, color = '#6C8EBF', label = ['Gyroskop: SensiML'])
ax[2].legend(loc="upper right")
ax[3].plot(smoothed_4, color = '#D6B656', label = ['Vibration: Milling NASA'])
ax[3].legend(loc="upper right")



ax[4].plot(smoothed_5, color = '#6C8EBF', label = ['Vibration: FEMTO'])
ax[4].legend(loc="upper right")
ax[5].plot(smoothed_6, color = '#B85450', label = ['Gyroskop: SensiML'])
ax[5].legend(loc="upper right")



plt_colors = ['#6C8EBF', '#82B366', '#D6B656', '#B85450', '#666666']
lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in plt_colors]
labels = ['Traingsdaten', 'Testdaten: Richtig klassifiziert', 'Testdaten: "Andere" vorhergesagt, aber Vibration','Testdaten: Vibration vorhergesagt, aber "Andere"','Potentielle Kandidaten für Shapelets']
#plt.legend(lines, labels)



lgd = fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.745, 0),
          fancybox=True, ncol=1,labelspacing=.8)

plt.savefig('C:/Users/USER/OneDrive - Technische Universität Berlin/Desktop/D/Mongrafie/Grafiken/FinaleGrafiken/6.5/HIVECOTE2_Binary_results_subplots.svg', bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=2000)            
plt.show()