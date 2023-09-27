import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import requests
import datetime as dt
from datetime import datetime, timedelta
import time
pd.options.mode.chained_assignment = None  # default='warn'

cases = ['Low' , 'Moderate' , 'High' , 'Very High']

def add_danger_column(data):
    low , moderate , high , very_high = 0 , 0, 0, 0
    if data['FFMC'] < 86.1:
        low += 1
    elif 86.1 <= data['FFMC'] <= 89.2:
        moderate += 1
    elif 89.2 <= data['FFMC'] <= 93.0:
        high += 1
    elif data['FFMC'] >= 93.0:
        very_high += 1

    if data['DMC'] < 27.9:
        low += 1
    elif 27.9 <= data['DMC'] <= 53.1:
        moderate += 1
    elif 53.1 <= data['DMC'] <= 140.7:
        high += 1
    else:
        very_high += 1

    if data['DC'] < 334.1:
        low += 1
    elif 334.1 <= data['DC'] <= 450.6:
        moderate += 1
    elif 450.6 <= data['DC'] < 794.4:
        high += 1
    else:
        very_high += 1

    if data['ISI'] < 5.0:
        low += 1
    elif 5.0 <= data['ISI'] < 7.5:
        moderate += 1
    elif 7.5 <= data['ISI'] <= 13.4:
        high += 1
    else:
        very_high += 1
    
    
    if low == max(low , moderate, high ,very_high):
        return 1
    elif moderate == max(low , moderate, high ,very_high):
        return 2
    elif high == max(low , moderate, high ,very_high):
        return 3
    else:
        return 4

def convert_int_to_str(data):
    if data['danger'] < 0:
        return 'Very Low'
    if 0 <= data['danger'] <= 1:
        return 'Low'
    if 1 <= data['danger'] <= 2:
        return 'Moderate'
    if 2 <= data['danger'] <= 3:
        return 'High'
    if 3 <= data['danger'] <= 4:
        return 'Very High'
    if data['danger'] >= 4:
        return 'Extreme'

def process_data_for_clustering(data , include_area=True , include_date=False):
    if include_area==True:
        data['area'] = np.log(1 + data['area'])
    if include_date==True:
        data['Date'] = data['start_time'].str[0:10]
    #data.drop(columns=['X' , 'Y' , 'month' , 'day'] , inplace=True)
    data['danger'] = data.apply(add_danger_column, axis=1)
    #imputer = IterativeImputer(estimator=linear_model.BayesianRidge(), sample_posterior=True,random_state=0)
    #data = pd.DataFrame(imputer.fit_transform(data),columns = data.columns, index=data.index)
    return data



def preprocess_forest_data(data):
    data.dropna(subset=['start_time' , 'end_time' , 'temp'] , inplace=True)
    return data

def fill_forest_data(data):
    response = requests.Session()
    longitude  = []
    latitude = []
    temp = []
    RH = []
    wind = []
    rain = []
    conditions = []
    description = []
    for index , row in data.iterrows():
        print(index)
        address = str(row['address'])
        address = address.replace(" " , "-") 
        start_time = row['start_time'][0:10]
        hour = row['start_time'][11:19]
        ans = response.get('https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/' + address + '-Greece'  + '/'
                                + start_time +'T'+hour+'?unitGroup=metric&include=days&key=UB4LE3EDV4RS24C69Y8WLWPMW&contentType=json&include=current&elements=tempmax,humidity,windspeed,precip,conditions,description', headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"})

        if ans.ok == False:
            longitude.append(None)
            latitude.append(None)
            temp.append(None)
            RH.append(None)
            wind.append(None)
            rain.append(None)
            conditions.append(None)
            description.append(None)
            continue
        add = ans.json()
        longitude.append(add['longitude'])
        latitude.append(add['latitude'])
        temp.append(add['days'][0].get('tempmax'))
        RH.append(add['days'][0].get('humidity'))
        wind.append(add['days'][0].get('windspeed'))
        rain.append(add['days'][0].get('precip'))
        conditions.append(add['days'][0].get('conditions'))
        description.append(add['days'][0].get('description'))

    data['longitude'] = longitude
    data['latitude'] = latitude
    data['temp'] = temp
    data['RH'] = RH
    data['wind'] = wind
    data['rain'] = rain
    data['conditions'] = conditions
    data['description'] = description
    data.to_csv('NEW_DATA.csv')
    return data
