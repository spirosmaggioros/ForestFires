import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
import requests
import json

data = pd.read_csv('data/forest_data.csv')
response = requests.get('https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Rodopi/2018-01-05/2018-01-05?unitGroup=metric&include=days&key=UB4LE3EDV4RS24C69Y8WLWPMW&contentType=csv')
answers = pd.read_csv(response)
print(answers)
