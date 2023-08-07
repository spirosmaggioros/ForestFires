from sklearn.metrics import classification_report, mean_absolute_error , mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, KFold, cross_val_score
from sklearn import svm
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
import pandas as pd
from sklearn.svm import SVR
#import tensorflow as tf
import matplotlib.pyplot as plt
#from tensorflow.keras import Model
#from tensorflow.keras import Sequential
#from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler , MinMaxScaler
#from tensorflow.keras.layers import Dense, Dropout
#from tensorflow.keras.losses import MeanSquaredLogarithmicError

def find_parameters():
    n_estimators = [int(x) for x in np.linspace(start=10 , stop=500, num=10)]
    max_depth = [int(x) for x in np.linspace(10 , 100 , num=10)]
    min_samples = [2 ,5 ,10]
    min_samples_leaf = [1 , 2 ,4]
    bootstrap=[True , False]

    random_grid = {
            'n_estimators':n_estimators,
            'max_depth':max_depth,
            'min_samples_split':min_samples,
            'min_samples_leaf':min_samples_leaf,
            'bootstrap':bootstrap,
            }
    
    return random_grid

def random_forest_algorithm(data):
    #print(df.head())
    #spliting the dataset
    scaler = StandardScaler()
    X = np.array(data[['temp' , 'wind' , 'RH' , 'rain']])
    X = scaler.fit_transform(X)
    y = np.array(data[['FFMC']])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=10, bootstrap=True, max_depth=40,min_samples_leaf=4,min_samples_split=2)
    rf.fit(X_train , y_train.ravel())
    
    y_pred = rf.predict(X_test)

    plt.figure()
    plt.plot(y_pred , label='predictions' , marker='o')
    plt.plot(y_test , label='original')
    plt.legend()
    plt.show()
    error = abs(y_pred - y_test)
    mape = np.mean(100 * (error / y_test))
    acc = 100 - mape
    print("ACCURACY : " + str(round(acc, 2)))

    #y_test = scaler.inverse_transform(y_test)
    #y_pred = scaler.inverse_transform(y_pred)
    return rf

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

def find_random_forest(data):
    scaler = StandardScaler()
    X = np.array(data[['temp' , 'RH' , 'rain' , 'wind']])
    X = scaler.fit_transform(X)
    y = np.array(data[['FFMC']])
    rf = RandomForestRegressor()
    grid_search = GridSearchCV(estimator = rf , param_grid = find_parameters(),
                               cv = 5, n_jobs = -1 , verbose=2)
    X_train , X_test , y_train , y_test = train_test_split(X , y ,train_size=0.2 , random_state=42)
    
    grid_search.fit(X_train , y_train.ravel())
    best_params = grid_search.best_params_
    print(best_params)
    best_estimator = grid_search.best_estimator_
    grid_accuracy = evaluate(best_estimator , X_test , y_test)
   


def KNNRegressor(data , n_neighbors = 9):
    X = data[['temp' , 'RH' , 'wind' , 'rain']].values.tolist()
    y = data[['FFMC' , 'DMC' , 'DC' , 'ISI']].values.tolist()
    #X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X , y)
    #y_pred = knn.predict(X_test) 
    #visual_original = []
    #visual_predicted = []

    #for i in range(len(y_pred)):
    #    visual_original.append(y_test[i][2])

    #for i in range(len(y_pred)):
    #    visual_predicted.append(y_pred[i][2])

    #fig = plt.figure()
    #plt.plot(visual_original , label='original')
    #plt.plot(visual_predicted , label='predicted')
    #plt.legend()
    #plt.show()

    return knn
    
#def neural_network_for_fire_regression(data):
#    X = data[['temp', 'RH', 'wind', 'rain']].values.tolist()
#    y = data[['FFMC', 'DMC', 'DC', 'ISI']].values.tolist()
#    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)
#    #scaling functionfrom sklearn.svm import SVR
#    def scale_datasets(x_train, x_test):
#
#          """
#          Standard Scale test and train data
#          Z - Score normalization
#          """
#          # we can also use the minmax scaler or
#          #even robust
#          standard_scaler = Standfrom sklearn.svm import SVRardScaler()
#          x_train_scaled = pd.DataFrame(
#              standard_scaler.fit_transform(x_train),
#              columns=x_train.columns
#          )
#          x_test_scaled = pd.DataFrame(
#              standard_scaler.transform(x_test),
#              columns = x_test.columns
#          )
#          return x_train_scaled, x_test_scaled
#    x_train_scaled, x_test_scaled = scale_datasets(X_train, X_test)
#    #Scaling the data would result in faster convergence to the global optimal value for loss function optimization functions.
#    #Here we use Sklearn’s StandardScaler class which performs z-score normalization.
#    #The z-score normalization subtracts each data from its mean and divides it by the standard deviation of the data.
#    hidden_units1 = 160
#    hidden_units2 = 480
#    hidden_units3 = 256
#    learning_rate = 0.01
#    # Creating model using the Sequential in tensorflow
#    def build_model_using_sequential():
#      model = Sequential([
#        Dense(hidden_units1, kernel_initializer='normal', activation='relu'),
#        Dropout(0.2),
#        Dense(hidden_units2, kernel_initializer='normal', activation='relu'),
#        Dropout(0.2),
#        Dense(hidden_units3, kernel_initializer='normal', activation='relu'),
#        Dense(1, kernel_initializer='normal', activation='linear')
#      ])
#      return model
#    # build the model
#    model = build_model_using_sequential()
#    # loss function
#    msle = MeanSquaredLogarithmicError()
#    model.compile(
#        loss=msle,
#        optimizer=Adam(learning_rate=learning_rate),
#        metrics=[msle]
#    )
#    # train the model
#    history = model.fit(
#        x_train_scaled.values,
#        y_train.values,
#        epochs=10,
#        batch_size=64,
#        validation_split=0.2
#    )
#    def plot_history(history, key):
#      plt.plot(history.history[key])
#      plt.plot(history.history['val_'+key])
#      plt.xlabel("Epochs")
#      plt.ylabel(key)
#      plt.legend([key, 'val_'+key])
#      plt.show()
#   # Plot the history
#   plot_history(history, 'mean_squared_logarithmic_error')
#    return model, model.predict(x_test_scaled), y_test


def svm_algorithm(data , kernel='linear', random_state=42):
    #print(data.head())
    X = data[['temp' , 'RH' , 'wind' , 'rain']].values.tolist()
    y = data['FFMC'].values.tolist()
    X_train , X_test , y_train , y_test = train_test_split(X ,y, test_size=0.2 , random_state=random_state)

    clf = SVR(kernel=kernel)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    #plt.figure()
    #plt.plot(y_pred , label='predictions')
    #plt.plot(y_test , label='original') 
    #plt.legend()
    #plt.show()
    return y_pred , y_test
 
def dbscan_algorithm(data , epsilon=3.5 , min_samples = 10):
    data.drop(columns=['area','month','day', 'X' , 'Y' , 'temp' , 'RH' , 'wind','rain'] , inplace=True)
    #data['area'] = np.log(1 + data['area'])
    print(data)
    clustering = DBSCAN(eps=epsilon , min_samples=min_samples).fit(data)
    print("\n==============\nDBSCAN output\n==============\n")
    print(f"There are {np.unique(clustering.labels_)} clusters.")
    data['cluster'] = clustering.labels_
    return data

def KNN_algorithm(data ,to_predict, n_neighbors=5 , random_state=42):
    X = data[['FFMC' , 'DMC' , 'DC', 'ISI' , 'temp', 'RH', 'wind' , 'rain']].values.tolist()
    y = data[[to_predict]].values.tolist()

    X_train , X_test , y_train , y_test = train_test_split(X, y, test_size = 0.2, random_state=random_state)

    knn = KNeighborsClassifier(n_neighbors = n_neighbors).fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return y_pred , y_test

def initialize_centroids(k , data):
    centroids = data.sample(k)
    return centroids

def kmeans_algorithm(data , n_clusters=4 , random_state=0, n_init="auto"):
    kmeans = KMeans(n_clusters=n_clusters,init='k-means++', random_state=random_state).fit_predict(data)
    return kmeans


