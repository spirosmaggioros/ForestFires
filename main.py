from sklearn import svm
from sklearn.cluster import DBSCAN, kmeans_plusplus
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from data_visual import get_layout
from sklearn.datasets import load_digits
from dash import Dash
from sklearn import metrics
from utils.algorithms import KNN_algorithm, dbscan_algorithm, svm_algorithm, kmeans_algorithm
from utils.data_preprocess import process_data_for_clustering

app = Dash(__name__, suppress_callback_exceptions=True)

data = pd.read_csv("forestfires.csv")

data = process_data_for_clustering(data)
y_pred, y_test = KNN_algorithm(data)
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.accuracy_score(y_test, y_pred))

colors = {1:'blue' , 2:'yellow' , 3:'red' , 4:'darkred'}

fig = go.Figure(layout=go.Layout(showlegend=True,template=None,scene=dict(xaxis = dict(title='RH'),yaxis = dict(title='wind'),zaxis = dict(title='temp')),title=go.layout.Title(text="3D Cluster After Preprocessing for cluster: " , font=dict(family="Arial",size=20,color='#000000'))))
fig.add_trace(
    go.Scatter3d(
    x = data['RH'],
    y = data['wind'],
    z = data['temp'],
    mode='markers',
    marker=dict(color=data['danger'].map(colors) , size=6),
    )
)

app.layout = get_layout(fig)
app.run_server()
