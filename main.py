from sklearn import svm
from sklearn.cluster import DBSCAN, kmeans_plusplus
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from data_visual import get_layout
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from dash import Dash
from sklearn import metrics
from utils.algorithms import KNN_algorithm, dbscan_algorithm, svm_algorithm, kmeans_algorithm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

app = Dash(__name__, suppress_callback_exceptions=True)

data = pd.read_csv("forestfires.csv")


#y_pred , y_test = svm_algorithm(data, kernel="linear")
#accuracy , recall = metrics.accuracy_score(y_test , y_pred) , metrics.recall_score(y_test,y_pred)
#print(f"Precision and recall for SVM algorithm is {accuracy} and {recall}")
#data['area'] = np.log(1 + data['area'])
print(data)
data['area'] = np.log(1 + data['area'])
X = data.iloc[: , [4 , 5, 6]].values
print(X)
y_kmeans = kmeans_algorithm(X)
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=100, c='blue', label ='Cluster 1')
#ax.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=100, c='yellow', label ='Cluster 2')
#ax.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=100, c='red', label ='Cluster 3')
#ax.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s=100, c='black', label ='Cluster 4')

#plt.show()
fig = go.Figure(layout=go.Layout(showlegend=True,template=None,scene=dict(xaxis = dict(title='FFMC'),yaxis = dict(title='DMC'),zaxis = dict(title='ISI')),title=go.layout.Title(text="3D Cluster After Preprocessing for cluster: " , font=dict(family="Arial",size=20,color='#000000'))))
fig.add_trace(
    go.Scatter3d(
    x = X[y_kmeans == 0 , 0],
    y = X[y_kmeans == 0 , 1],
    z = X[y_kmeans == 0 , 2],
    mode='markers',
    marker=dict(color='blue', size=6)
    )
)
fig.add_trace(
    go.Scatter3d(
   x = X[y_kmeans == 1 , 0],
   y = X[y_kmeans == 1 , 1],
   z = X[y_kmeans == 1 , 2],
   mode='markers',
   marker=dict(color='yellow', size=6))
    )
fig.add_trace(
   go.Scatter3d(
    x = X[y_kmeans == 2 , 0],
   y = X[y_kmeans == 2 , 1],
   z = X[y_kmeans == 2 , 2],
   mode='markers',
   marker=dict(color='red', size=6))
    )
fig.add_trace(
   go.Scatter3d(
   x = X[y_kmeans == 3 , 0],
   y = X[y_kmeans == 3 , 1],
   z = X[y_kmeans == 3 , 2],
   mode='markers',
   marker=dict(color='black', size=6)
    )
)

app.layout = get_layout(fig)
app.run_server()
