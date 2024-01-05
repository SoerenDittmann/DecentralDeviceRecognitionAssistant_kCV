import keras
import numpy as np
import tensorflow as tf
from scipy.stats import entropy


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, fbeta_score, confusion_matrix

from sktime.classification.interval_based import RandomIntervalSpectralEnsemble


def custom_f1(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average = 'micro', labels = np.unique(y_true))
    return f1

def custom_fbeta(y_true, y_pred, beta):
    f1 = fbeta_score(y_true, y_pred, average = 'micro', beta=beta, labels = np.unique(y_true))
    return f1

#funktionsweise implementieren was passiert wenn same probability
#implementiere weiteren parameter_ distanz, der den Abstand zum zweit wahrscheinlichsten optimiert

def decision_func_entropy(probas, threshold):
    
    #Create boolean mask from max values - all zeros but 1 where max_value is located
    max_val_bool = np.zeros(probas.shape, dtype=bool) 
    max_val_bool[np.arange(len(probas)), probas.argmax(axis=1)] = 1
           
    #calculate columnwise entropy of all entries
    entr = entropy(probas,axis=1, base=2)/np.log2(probas.shape[1])
    entr = entr > threshold
    entr=entr[:,None]
    
    #extend array to include potential rejection option if entropy is higher than threshold (*2 for prioritization)
    ext_matrix = np.concatenate((max_val_bool,entr*2),axis=1)
    
    return ext_matrix.argmax(axis=1) #shape=(#point, 1) 

def decision_func_rel_entropy(probas, threshold, class_distribution):
    
    #Create boolean mask from max values - all zeros but 1 where max_value is located
    max_val_bool = np.zeros(probas.shape, dtype=bool) 
    max_val_bool[np.arange(len(probas)), probas.argmax(axis=1)] = 1
       
    class_distribution = np.tile(class_distribution,(probas.shape[0],1))
    
    #calculate columnwise entropy of all entries
    entr = entropy(probas, qk = class_distribution,axis=1, base=2)/np.log2(probas.shape[1])
    entr = entr > threshold
    entr=entr[:,None]
    
    #extend array to include potential rejection option if entropy is higher than threshold (*2 for prioritization)
    ext_matrix = np.concatenate((max_val_bool,entr*2),axis=1)
    
    return ext_matrix.argmax(axis=1) #shape=(#point, 1) 


def decision_func_probability(probas, threshold):
    
    #Delete all non-maximal values of a column
    #---First create boolean mask from max values
    max_val_bool = np.zeros(probas.shape, dtype=bool) 
    max_val_bool[np.arange(len(probas)), probas.argmax(axis=1)] = 1
    #---Select only those max values from the input array keeping shape constant
    max_vals = probas*max_val_bool
    
    #compare probability to threshold and prioritize with multiplication
    mask = 2*(max_vals > threshold)
    
    #extend array to include potential rejection option
    ext_mask = np.pad(mask,((0,0),(0,1)),mode='constant',constant_values=1) #shape=(#point, #classes)
    
    return ext_mask.argmax(axis=1) #shape=(#point, 1) 
    
class RISErejectOption_entropy(RandomIntervalSpectralEnsemble):
    
    def __init__(
        self,
        n_estimators=200,
        min_interval=16,
        max_interval=0,#manually added by SD
        acf_lag=100,
        acf_min_values=4,
        n_jobs=None,
        random_state=None,
        threshold = 1
    ):
        super(RandomIntervalSpectralEnsemble, self).__init__(
            base_estimator=DecisionTreeClassifier(random_state=random_state),
            n_estimators=n_estimators,
        )
        self.n_estimators = n_estimators
        self.min_interval = min_interval
        self.max_interval = max_interval#manually added by SD
        self.acf_lag = acf_lag
        self.acf_min_values = acf_min_values
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.threshold = threshold

        # We need to add is-fitted state when inheriting from scikit-learn
        self._is_fitted = False

    def predict(self, X):
        """
        Find predictions for all cases in X. Built on top of `predict_proba`.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_instances, n_columns]
            The input samples. If a Pandas data frame is passed it must have a
            single column (i.e., univariate classification). RISE has no
            bespoke method for multivariate classification as yet.
        Returns
        -------
        y : array of shape = [n_instances]
            The predicted classes.
        """
        yPred = decision_func_entropy(self.predict_proba(X),self.threshold)       
        return yPred
    
class RISErejectOption_rel_entropy(RandomIntervalSpectralEnsemble): #calculate entropy relative to class_distribution
    
    def __init__(
        self,
        n_estimators=200,
        min_interval=16,
        acf_lag=100,
        acf_min_values=4,
        n_jobs=None,
        random_state=None,
        threshold = 1,
        class_distribution = None
    ):
        super(RandomIntervalSpectralEnsemble, self).__init__(
            base_estimator=DecisionTreeClassifier(random_state=random_state),
            n_estimators=n_estimators,
        )
        self.n_estimators = n_estimators
        self.min_interval = min_interval
        self.acf_lag = acf_lag
        self.acf_min_values = acf_min_values
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.threshold = threshold
        #create class_distribution variable to save baseline
        #could be implemented into .fit function
        self.class_distribution = class_distribution

        # We need to add is-fitted state when inheriting from scikit-learn
        self._is_fitted = False

    def predict(self, X):
        """
        Find predictions for all cases in X. Built on top of `predict_proba`.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_instances, n_columns]
            The input samples. If a Pandas data frame is passed it must have a
            single column (i.e., univariate classification). RISE has no
            bespoke method for multivariate classification as yet.
        Returns
        -------
        y : array of shape = [n_instances]
            The predicted classes.
        """
        yPred = decision_func_rel_entropy(self.predict_proba(X),self.threshold,self.class_distribution)   
        return yPred
    
class RISErejectOption_proba(RandomIntervalSpectralEnsemble): #threshold is not applied to entropy but to probability of highest class
    
    def __init__(
        self,
        n_estimators=200,
        min_interval=16,
        acf_lag=100,
        acf_min_values=4,
        n_jobs=None,
        random_state=None,
        threshold = 0
    ):
        super(RandomIntervalSpectralEnsemble, self).__init__(
            base_estimator=DecisionTreeClassifier(random_state=random_state),
            n_estimators=n_estimators,
        )
        self.n_estimators = n_estimators
        self.min_interval = min_interval
        self.acf_lag = acf_lag
        self.acf_min_values = acf_min_values
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.threshold = threshold

        # We need to add is-fitted state when inheriting from scikit-learn
        self._is_fitted = False

    def predict(self, X):
        """
        Find predictions for all cases in X. Built on top of `predict_proba`.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_instances, n_columns]
            The input samples. If a Pandas data frame is passed it must have a
            single column (i.e., univariate classification). RISE has no
            bespoke method for multivariate classification as yet.
        Returns
        -------
        y : array of shape = [n_instances]
            The predicted classes.
        """
        yPred = decision_func_probability(self.predict_proba(X),self.threshold)       
        return yPred
    
class Classifier_RESNET:

    def __init__(self, output_directory, input_shape, nb_classes, custom_metric, threshold = 1, verbose=False, build=True, load_weights=False):
        self.output_directory = output_directory
        self.threshold = threshold
        self.custom_metric = custom_metric##
        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            if load_weights == True:
                self.model.load_weights(self.output_directory
                                        .replace('resnet_augment', 'resnet')
                                        .replace('TSC_itr_augment_x_10', 'TSC_itr_10')
                                        + '/model_init.hdf5')
            else:
                self.model.save_weights(self.output_directory + 'model_init.hdf5')
        return

    def build_model(self, input_shape, nb_classes):
        n_feature_maps = 64

        input_layer = keras.layers.Input(input_shape)

        # BLOCK 1

        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)

        # BLOCK 2

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)

        # BLOCK 3

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = keras.layers.BatchNormalization()(output_block_2)

        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation('relu')(output_block_3)

        # FINAL

        gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy']) #Did not work for me to change to different metric

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)

        self.callbacks = [reduce_lr, model_checkpoint]

        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true, epochs):
        if not tf.test.is_gpu_available:
            print('error')
            exit()
            
        # x_val and y_val are only used to monitor the test loss and NOT for training
        batch_size = 64
        nb_epochs = epochs #default was 1500

        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)

        self.model.save(self.output_directory + 'last_model.hdf5')

        y_pred = self.predict(x_val, y_true, x_train, y_train, y_val,
                              return_df_metrics=False) #y_pred are containing the probabilities of all possible classes

        # save predictions
        np.save(self.output_directory + 'y_pred.npy', y_pred)

        #check whether a custom metric is wanted
        if self.custom_metric:
            #take threshold into consideration
            y_pred = decision_func_entropy(y_pred, self.threshold)
            #return custom metrics f1-Score micro
            df_metrics = custom_fbeta(y_true, y_pred, beta=0.5)
        
        else:
            y_pred = np.argmax(y_pred, axis=1)       
            cm = confusion_matrix(y_true, y_pred)
            df_metrics = cm.diagonal().sum()/len(y_true)


        keras.backend.clear_session()

        return df_metrics

    def predict(self, x_test, y_true, x_train, y_train, y_test, return_df_metrics=True):
        model_path = self.output_directory + 'best_model.hdf5'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test)     
        if return_df_metrics:
            #check whether a custom metric is wanted
            if self.custom_metric:
                #take threshold into consideration
                y_pred = decision_func_entropy(y_pred, self.threshold)
                #return custom metrics f1-score micro
                df_metrics = custom_fbeta(y_true, y_pred, beta=0.5)
            else:
                y_pred = np.argmax(y_pred, axis=1)       
                cm = confusion_matrix(y_true, y_pred)
                df_metrics = cm.diagonal().sum()/len(y_true)
            return df_metrics
        else:
            return y_pred
