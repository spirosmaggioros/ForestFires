from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.ensemble import  RandomForestClassifier , RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.svm import SVR
import requests
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def random_forest_algorithm(df , predict):
    #print(df.head())
    #spliting the dataset
    scaler = StandardScaler()
    X = df[['temp' ,'RH', 'wind' , 'rain']].values.tolist()
    X = scaler.fit_transform(X)
    y = df[[predict]].values.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=490 , random_state=42)
    model.fit(X_train , np.ravel(y_train))
    y_pred = model.predict(X_test)
    accuracy = mean_absolute_error(y_test , np.ravel(y_pred))
    print("Accuracy is " + str(accuracy))

    # Store the accuracies for different estimator values
    #ccuracies = []
    #estimators = np.arange(10 ,1000 , 10)
    #Train and evaluate the model for each estimator value
    #for n_estimators in estimators:
    #    print(n_estimators)
    #    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    #    model.fit(X_train, np.ravel(y_train))
    #    y_pred = model.predict(X_test)
    #    accuracy = mean_absolute_error(y_test , np.ravel(y_pred)) 
    #    accuracies.append(accuracy)

     #Plot the results
    #plt.plot(estimators, accuracies, marker='o')
    #plt.xlabel("Estimator Values")
    #plt.ylabel("Accuracy")
    #plt.legend()
    #plt.title("Best parameters for the RFC")
    #plt.show()
    #using the best model and graph it
    #rf = RandomForestRegressor(n_estimators=3, random_state=42)
    #rf.fit(X_train, np.ravel(y_train))
    #y_pred = rf.predict(X_test)
    plt.plot(y_test , label="testing")
    plt.plot(y_pred , label="predictions")
    plt.legend()
    plt.show()
    #print(mean_absolute_error(y_test , y_pred))
    return model

def KNNRegressor(data, predicted_value = 'FFMC'):
    X = data[['temp' , 'RH' , 'wind' , 'rain']].values.tolist()
    y = data[[predicted_value]].values.tolist()
    param = 1
    if predicted_value == 'FFMC':
        param = 44
    elif predicted_value == 'DMC' or predicted_value == 'DC':
        param = 2
    else:
        param = 3
    knn = KNeighborsRegressor(n_neighbors=param , n_jobs=-1)
    knn.fit(X , y)
    #knn = KNeighborsRegressor(n_neighbors=5)
    #knn.fit(X_train , y_train)
    #preds = knn.predict(X_test)
    #plt.plot(preds)
    #plt.plot(y_test)
    #plt.show()
    return knn

def nn(data , to_predict): 
    X = data[['temp' , 'RH' , 'wind' , 'rain']].values.tolist()
    y = data[to_predict].values.tolist()
    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)
    model = keras.Sequential([
        layers.Dense(units=4, activation='relu' , input_shape=[4]),
        layers.Dense(units=4, activation='relu'),
        layers.Dense(units=4 , activation='relu'),
        layers.Dense(units=4 , activation='relu'),
        layers.Dense(units=1)
        ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
        )

    history = model.fit(
        X_train , y_train,
        validation_data = (X_test , y_test),
        batch_size = 32,
        epochs=100,
        )

    history_df = pd.DataFrame(history.history)
    history_df['loss'].plot()
    plt.show()
    return model

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

def dbscan_algorithm(data , epsilon=1.5 , min_samples = 9):
    X = data[['temp' , 'rain' , 'wind' , 'RH' , 'area']]
    clustering = DBSCAN(eps=epsilon , min_samples=min_samples).fit(X)
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

def agglomerative_clustering(data , n_clusters = 5 , affinity='euclidean' , linkage='single'):
    X = data[['temp' , 'rain' , 'wind' , 'RH']].iloc[0:300].values
    model = AgglomerativeClustering(n_clusters=n_clusters , affinity=affinity , linkage=linkage)
    model.fit(X)
    labels = model.labels_
    plt.scatter(X[labels==0, 0], X[labels==0, 1],  marker='o', color='red')
    plt.scatter(X[labels==1, 0], X[labels==1, 1],marker='o', color='blue')
    plt.scatter(X[labels==2, 0], X[labels==2, 1], marker='o', color='green')
    plt.scatter(X[labels==3, 0], X[labels==3, 1], marker='o', color='purple')
    plt.scatter(X[labels==4, 0], X[labels==4, 1],  marker='o', color='orange')
    plt.show()


