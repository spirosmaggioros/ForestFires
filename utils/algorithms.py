from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import svm
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.ensemble import  RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import MeanSquaredLogarithmicError


def random_forest_algorithm(df):
    #print(df.head())
    #spliting the dataset
    X = df.drop(["danger", "month", "day"], axis=1)
    y = df.pop('danger')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Store the accuracies for different estimator values
    accuracies = []
    estimator_values = [10, 20, 50, 100, 150, 200, 250, 300]
    # Train and evaluate the model for each estimator value
    for n_estimators in estimator_values:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    # Plot the results
    plt.plot(estimator_values, accuracies, marker='o')
    plt.xlabel("Estimator Values")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Best parameters for the RFC")
    #using the best model and graph it
    rf = RandomForestClassifier(n_estimators=150, max_depth=3,
                                max_features=3,
                                min_samples_leaf=4, random_state=42)
    rf.fit(X_train, y_train)
    
    fig = plt.figure(figsize=(15, 10))
    plot_tree(rf.estimators_[0],filled=True, impurity=True,
              rounded=True)
    plt.title("THE FIRST DECISION TREE")
    plt.show()
    
    return 

def KNNRegressor(data , n_neighbors = 5):
    X = data[['temp' , 'RH' , 'wind' , 'rain']].values.tolist()
    y = data[['FFMC' , 'DMC' , 'DC' , 'ISI']].values.tolist()
    #X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X , y)
    #y_pred = knn.predict(X_test) 
    #visual_original = []
    #visual_predicted = []

    #for i in range(len(y_pred)):
    #    visual_original.append(y_test[i][1])

    #for i in range(len(y_pred)):
    #    visual_predicted.append(y_pred[i][1])

    #fig = plt.figure()
    #plt.plot(visual_original , label='original')
    #plt.plot(visual_predicted , label='predicted')
    #plt.legend()
    #plt.show()

    return knn
    
def neural_network_for_fire_regression(data):
    X = data[['temp', 'RH', 'wind', 'rain']].values.tolist()
    y = data[['FFMC', 'DMC', 'DC', 'ISI']].values.tolist()
    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)
    #scaling function
    def scale_datasets(x_train, x_test):

          """
          Standard Scale test and train data
          Z - Score normalization
          """
          # we can also use the minmax scaler or
          #even robust
          standard_scaler = StandardScaler()
          x_train_scaled = pd.DataFrame(
              standard_scaler.fit_transform(x_train),
              columns=x_train.columns
          )
          x_test_scaled = pd.DataFrame(
              standard_scaler.transform(x_test),
              columns = x_test.columns
          )
          return x_train_scaled, x_test_scaled
    x_train_scaled, x_test_scaled = scale_datasets(X_train, X_test)
    #Scaling the data would result in faster convergence to the global optimal value for loss function optimization functions.
    #Here we use Sklearn’s StandardScaler class which performs z-score normalization.
    #The z-score normalization subtracts each data from its mean and divides it by the standard deviation of the data.
    hidden_units1 = 160
    hidden_units2 = 480
    hidden_units3 = 256
    learning_rate = 0.01
    # Creating model using the Sequential in tensorflow
    def build_model_using_sequential():
      model = Sequential([
        Dense(hidden_units1, kernel_initializer='normal', activation='relu'),
        Dropout(0.2),
        Dense(hidden_units2, kernel_initializer='normal', activation='relu'),
        Dropout(0.2),
        Dense(hidden_units3, kernel_initializer='normal', activation='relu'),
        Dense(1, kernel_initializer='normal', activation='linear')
      ])
      return model
    # build the model
    model = build_model_using_sequential()
    # loss function
    msle = MeanSquaredLogarithmicError()
    model.compile(
        loss=msle,
        optimizer=Adam(learning_rate=learning_rate),
        metrics=[msle]
    )
    # train the model
    history = model.fit(
        x_train_scaled.values,
        y_train.values,
        epochs=10,
        batch_size=64,
        validation_split=0.2
    )
    def plot_history(history, key):
      plt.plot(history.history[key])
      plt.plot(history.history['val_'+key])
      plt.xlabel("Epochs")
      plt.ylabel(key)
      plt.legend([key, 'val_'+key])
      plt.show()
    # Plot the history
    plot_history(history, 'mean_squared_logarithmic_error')
    return model, model.predict(x_test_scaled), y_test


def svm_algorithm(data , kernel='linear', random_state=42):
    #print(data.head())
    data['area'] = np.log(1 + data['area'])
    X = data[['FFMC', 'DMC','DC' , 'ISI', 'temp' , 'RH' , 'wind' , 'rain']].values.tolist()
    y = data['danger'].values.tolist()
    X_train , X_test , y_train , y_test = train_test_split(X ,y, test_size=0.3 , random_state=random_state)

    clf = svm.SVC(kernel=kernel)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
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


