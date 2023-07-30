import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

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

def process_data_for_clustering(data):
    data['area'] = np.log(1 + data['area'])
    #data.drop(columns=['X' , 'Y' , 'month' , 'day'] , inplace=True)
    data['danger'] = data.apply(add_danger_column, axis=1)
    #imputer = IterativeImputer(estimator=linear_model.BayesianRidge(), sample_posterior=True,random_state=0)
    #data = pd.DataFrame(imputer.fit_transform(data),columns = data.columns, index=data.index)
    return data
    

