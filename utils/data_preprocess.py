import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import requests
from datetime import datetime, timedelta

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

def process_data_for_clustering(data , include_area=True):
    if include_area==True:
        data['area'] = np.log(1 + data['area'])
    #data.drop(columns=['X' , 'Y' , 'month' , 'day'] , inplace=True)
    data['danger'] = data.apply(add_danger_column, axis=1)
    #imputer = IterativeImputer(estimator=linear_model.BayesianRidge(), sample_posterior=True,random_state=0)
    #data = pd.DataFrame(imputer.fit_transform(data),columns = data.columns, index=data.index)
    return data

def preprocess_forest_data(data):
    data.dropna(subset=['address'] , inplace=True)
    data['area'] = data[['agricultural_area_burned' , 'crop_residue_area_burned' , 'dumping_ground_area_burned' , 'forest_area_burned' , 'grove_area_burned' ,'low_vegetation_area_burned', 'swamp_area_burned','woodland_area_burned']].sum(axis=1)
    data.drop(columns=['airplanes_cl215' , 'airplanes_cl415' , 'airplanes_gru' , 'airplanes_pzl' , 'army' , 'firefighters' , 'fire_station' , 'fire_trucks' , 'helicopters' , 'local_authorities_vehicles' , 'location' , 'machinery', 'other_firefighters' , 'machinery' , 'prefecture' , 'volunteers' , 'water_tank_trucks' , 'wildland_crew'] , inplace=True)
    data.drop(columns=['agricultural_area_burned' , 'crop_residue_area_burned' , 'dumping_ground_area_burned' ,'forest_area_burned' , 'grove_area_burned' ,'low_vegetation_area_burned', 'swamp_area_burned','woodland_area_burned'] , inplace=True)
    return data

def fill_forest_data(data):
    response = requests.Session()
    for index , row in data.iterrows():
        print(index)
        address = str(row['address'])
        address = address.replace(" " , "-") 
        start_time = row['start_time'][0:10]
        hour = row['start_time'][11:19]
        response = requests.get('https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/' + address + '-Greece'  + '/' 
                                + start_time +'T'+hour+'?unitGroup=metric&include=days&key=UB4LE3EDV4RS24C69Y8WLWPMW&contentType=json&include=current&elements=tempmax,humidity,windspeed,precip,conditions,description', headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"})
        if response.ok == False:
            data.drop(index = data.index[index] , axis = 0 , inplace = True)
            continue
        add = response.json()
        row['longitude'] = add['longitude']
        row['latitude'] = add['latitude']
        row['temp'] = add['days'][0].get('tempmax')
        row['RH'] = add['days'][0].get('humidity')
        row['wind'] = add['days'][0].get('windspeed')
        row['rain'] = add['days'][0].get('precip')
        row['conditions'] = add['days'][0].get('conditions')
        row['description'] = add['days'][0].get('description')
   
    data.to_csv('NEW_DATA.csv')
    return data
