from sklearn import svm
from sklearn.cluster import DBSCAN, kmeans_plusplus
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from visualization.visual import get_layout
from sklearn.datasets import load_digits
from dash import Dash ,  Output, Input
from sklearn import metrics
from utils.algorithms import KNN_algorithm, dbscan_algorithm, svm_algorithm, kmeans_algorithm, random_forest_algorithm , KNNRegressor
from utils.data_preprocess import process_data_for_clustering

app = Dash(__name__, suppress_callback_exceptions=True)

dropdown_figures = ['Barplot' , 'Scatterplot' , 'Lineplot']

def fill_data(meteorological_data , predictions):
    meteorological_data['FFMC'] = predictions[: , 0]
    meteorological_data['DMC'] = predictions[: , 1]
    meteorological_data['DC'] = predictions[: , 2]
    meteorological_data['ISI'] = predictions[: , 3]
    return meteorological_data


data = pd.read_csv("forestfires.csv")
    
data = process_data_for_clustering(data)
meteorological_data = pd.read_csv('naxos_data.csv')

knn = KNNRegressor(data)
predictions = knn.predict(meteorological_data[['temp' , 'RH' , 'wind' , 'rain']].values.tolist())
meteorological_data = fill_data(meteorological_data , predictions)
meteorological_data = process_data_for_clustering(meteorological_data, include_area = False)

fig = px.bar(meteorological_data , x=meteorological_data['Date'].tolist() ,y="danger" , title="Danger level")


@app.callback(
    Output('graph1' , 'figure' , allow_duplicate=True),
    Input('dropdown-figure' , 'value'),
    prevent_initial_call=True
)
def update_data(figure_value):
    global meteorological_data
    if figure_value == "Barplot": 
        fig = px.bar(meteorological_data , x=meteorological_data['Date'].tolist(), y="danger", title="Danger level for: " )
    elif figure_value == "Scatterplot":
        fig = px.scatter(meteorological_data , x="Date" , y="temp" ,color='danger')
    else:
        fig = px.line(meteorological_data, x="Date" , y="danger")
    return fig



colors = {1:'blue' , 2:'yellow' , 3:'red' , 4:'darkred'}

app.layout = get_layout(fig ,dropdown_figures)
app.run_server(threaded=True)
