from sklearn import svm
from sklearn.cluster import DBSCAN, kmeans_plusplus
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from visualization.visual import get_layout , set_area_of_interest
from sklearn.datasets import load_digits
from dash import Dash ,  Output, Input
from sklearn import metrics
from utils.algorithms import nn,KNN_algorithm, dbscan_algorithm, svm_algorithm, kmeans_algorithm, random_forest_algorithm , KNNRegressor , agglomerative_clustering
from utils.data_preprocess import process_data_for_clustering , preprocess_forest_data , fill_forest_data
import matplotlib.pyplot as plt
import seaborn as sns

app = Dash(__name__, suppress_callback_exceptions=True)

dropdown_figures = ['Barplot' , 'Scatterplot' , 'Lineplot']

def fill_data(meteorological_data , FFMC , DMC , DC , ISI):
    meteorological_data['FFMC'] = FFMC
    meteorological_data['DMC'] = DMC
    meteorological_data['DC'] = DC
    meteorological_data['ISI'] = ISI
    return meteorological_data


data = pd.read_csv("data/forestfires.csv")
forest_data = pd.read_csv("NEW_DATA.csv")
forest_data = preprocess_forest_data(forest_data)
#forest_data = fill_forest_data(forest_data)
#knn = KNNRegressor(forest_data)
#rf = random_forest_algorithm(forest_data)

data = process_data_for_clustering(data)


FFMC = KNNRegressor(data , 'FFMC')
DMC = KNNRegressor(data , 'DMC')
DC = KNNRegressor(data , 'DC')
ISI = KNNRegressor(data , 'ISI')

meteorological_data = pd.read_csv('data/naxos_data.csv')
ffmc = FFMC.predict(forest_data[['temp' , 'RH' , 'wind', 'rain']].values.tolist())
dmc = DMC.predict(forest_data[['temp' , 'RH' , 'wind', 'rain']].values.tolist())
dc = DC.predict(forest_data[['temp' , 'RH' , 'wind', 'rain']].values.tolist())
isi = ISI.predict(forest_data[['temp' , 'RH' , 'wind', 'rain']].values.tolist())
forest = fill_data(forest_data , ffmc , dmc , dc , isi)
forest_data = process_data_for_clustering(forest_data, include_area = False , include_date=True)

heatmap = sns.heatmap(forest_data[['temp' , 'RH' , 'wind' , 'rain' , 'FFMC' , 'DMC' , 'DC' , 'ISI']].corr()[['DMC']].sort_values(by='FFMC' , ascending=False) , vmin=-1 , vmax = 1 , annot=True , cmap='BrBG')
plt.show()


fig = set_area_of_interest(forest_data[0:10] , 1)
fig2 = set_area_of_interest(forest_data[0:10] , 2)
fig3 = set_area_of_interest(forest_data[0:10] , 3)
fig4 = set_area_of_interest(forest_data[0:10] , 4)
fig5 = set_area_of_interest(forest_data[0:10] , 5)


@app.callback(
    Output('datatable-after-gap' , 'data', allow_duplicate=True),
    Output('graph1' , 'figure' , allow_duplicate=True),
    Output('graph2' , 'figure' , allow_duplicate=True),
    Output('graph3' , 'figure' , allow_duplicate=True),
    Output('graph4' , 'figure', allow_duplicate=True),
    Output('graph5' , 'figure' , allow_duplicate=True),
    [Input('dropdown-figure' , 'value'),
    Input('dropdown-dates' , 'value')],
    prevent_initial_call=True
)

def update_data(figure_value , date_value):
    global forest_data
    fig2 = px.pie(forest_data.loc[forest_data['Date'] == date_value] , values='danger' ,names='danger')
    fig3 = px.scatter_matrix(forest_data.loc[forest_data['Date'] == date_value] , dimensions=['FFMC' , 'DMC' , 'DC' ,'ISI'], color='danger')
    fig4 = px.parallel_coordinates(forest_data.loc[forest_data['Date'] == date_value] , color='danger' , labels={'FFMC' : 'FFMC' , 'DMC':'DMC' , 'DC':'DC' , 'ISI':'ISI',},
                                      color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=2)
    fig5 = px.density_mapbox(forest_data.loc[forest_data['Date'] == date_value] , lat='latitude' , lon='longitude' , z='danger' , radius=10 , center=dict(lat=38, lon=24) , zoom=5 , mapbox_style="stamen-terrain")
    if figure_value == "Barplot": 
        fig = px.bar(forest_data.loc[forest_data['Date'] == date_value] , x=forest_data['Date'].tolist(), y="danger", title="Danger level for: " )
    elif figure_value == "Scatterplot":
        fig = px.scatter(forest_data.loc[forest_data['Date'] == date_value] , x="Date" , y="temp" ,color='danger')
    else:
        fig = px.line(forest_data.loc[forest_data['Date'] == date_value], x="Date" , y="danger")

    return forest_data.loc[forest_data['Date'] == date_value].to_dict('records') , fig , fig2 , fig3 , fig4 , fig5



colors = {1:'blue' , 2:'yellow' , 3:'red' , 4:'darkred'}
dropdown_dates = forest_data['Date'].unique()
app.layout = get_layout(fig ,fig2,fig3,fig4,fig5, dropdown_figures, dropdown_dates , forest_data[0:10])
app.run_server(threaded=True)

