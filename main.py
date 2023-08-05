from sklearn import svm
from sklearn.cluster import DBSCAN, kmeans_plusplus
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from visualization.visual import get_layout
from sklearn.datasets import load_digits
from dash import Dash
from sklearn import metrics
from utils.algorithms import KNN_algorithm, dbscan_algorithm, svm_algorithm, kmeans_algorithm, random_forest_algorithm , KNNRegressor
from utils.data_preprocess import process_data_for_clustering

app = Dash(__name__, suppress_callback_exceptions=True)

data = pd.read_csv("forestfires.csv")

data = process_data_for_clustering(data)
meteorological_data = pd.read_csv('naxos_data.csv')
knn = KNNRegressor(data)
predictions = knn.predict(meteorological_data[['temp' , 'RH' , 'wind' , 'rain']].values.tolist())
meteorological_data['FFMC'] = predictions[: , 0]
meteorological_data['DMC'] = predictions[: , 1]
meteorological_data['DC'] = predictions[: , 2]
meteorological_data['ISI'] = predictions[: , 3]

meteorological_data = process_data_for_clustering(meteorological_data, include_area = False)

#y_pred, y_test = KNN_algorithm(data , to_predict='area')
#print(metrics.confusion_matrix(y_test, y_pred))
#print(metrics.accuracy_score(y_test, y_pred))

colors = {1:'blue' , 2:'yellow' , 3:'red' , 4:'darkred'}

fig = px.line(meteorological_data , x="Date" , y="danger" , title="danger levels over the next 5 days")
app.layout = get_layout(fig)
app.run_server()
