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


