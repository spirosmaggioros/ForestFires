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
