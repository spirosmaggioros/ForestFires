import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
def add_danger_column(data):
    if data['FFMC'] < 86.1 and data['DMC'] < 27.9 and data['DC'] < 334.1 and data['ISI'] < 5.0:
        return 1
    if 86.1 <= data['FFMC'] <= 89.2 and 27.9 <= data['DMC'] <= 53.1 and 334.1 <= data['DC'] <= 450.6 and 5.0 <= data['ISI'] < 7.5: 
        return 2
    if 89.2 <= data['FFMC'] <= 93.0 and 53.1 <= data['DMC'] <= 140.7 and 450.6 <= data['DC'] < 794.4 and 7.5 <= data['ISI'] <= 13.4:
        return 3
    if data['FFMC'] >= 93.0 and data['DMC'] >= 140.7 and data['DC'] >= 794.4 and data['ISI'] >= 13.4:
        return 4
    else:
        return np.nan

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
    data.drop(columns=['X' , 'Y' , 'month' , 'day'] , inplace=True)
    data['danger'] = data.apply(add_danger_column, axis=1)
    imputer = IterativeImputer(estimator=linear_model.BayesianRidge(), sample_posterior=True,random_state=0)
    data = pd.DataFrame(imputer.fit_transform(data),columns = data.columns, index=data.index)
    data['danger'] = data.apply(convert_int_to_str, axis=1)
    return data
    

