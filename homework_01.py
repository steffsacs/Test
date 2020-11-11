import json
import requests

import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2, linregress
from scipy.optimize import minimize
from numpy import linalg as LA

#import our own files and reload
import FunctionsC530S
importlib.reload(FunctionsC530S)
import stream_classes8
importlib.reload(stream_classes8)

'''
HOMEWORK 1.

For each ric in a given list of rics, compute the values requested in a
given list of risk metrics and return a dataframe.

The risk metrics can change in number and order, but they always are a
subset of the following:
risk_metrics = ['ric',\
                'mean',\
                'std_dev',\
                'skewness',\
                'kurtosis',\
                'var_95',\
                'cvar_95',\
                'sharpe',\
                'jarque_bera',\
                'p_value',\
                'is_normal']

We will compare the resulting dataframe with our own, and based on the
number of correct entries your note will be given: 87% of correct entries
means your homework will be noted as 8.7.

To ensure the code is modulable enough, we will run a few different lists of
rics and lists of risk metrics. Each run will give a note, and your final
note will be the average of all these runs.

You can submit the homework as many times as you like until the deadline. The
final note will be the maximum note of all your submits.

After the deadline, we will send our solution.

You can test your code on your computer before sending it to github.
To test your code, you can run the ' pytest ' command on your computer or select
run test option in spyder and select test_homework01.py .
Remember to install pytest using $pip install pytest or $conda install pytest

'''
rics = ['BBVA.MC','SAN.MC']  #MT.AS','SAN.MC', 'BBVA.MC','REP.MC', 'VWS.CO', 'EQNR.OL', 'MXNUSD=X'
risk_metrics = ['ric',\
                'mean',\
                'std_dev',\
                'skewness',\
                'kurtosis',\
                'var_95',\
                'cvar_95',\
                'sharpe',\
                'jarque_bera',\
                'p_value',\
                'is_normal']
def create_dataframe_risk_metrics(rics, risk_metrics):
    #  NO OLVIDES CAMBIAR TU NUMERO DE CUENTA EN EL ARCHIVO test_homework01.py
    columns = risk_metrics
    nrows = len(rics) #Number of rows
    df = pd.DataFrame(columns = columns, index = range(nrows))
    for i in range(len(rics)):
        ric = rics[i]
       # risk_metrics = risk_metrics[i]
        jb = stream_classes8.jarque_bera_test(ric)     
        jb.load_timeseries() # llamamos al constructor
        jb.compute()
        #for i in range(len(risk_metrics)): #Eliminar
        if 'ric' in risk_metrics:
            df.iloc[i, df.columns.get_loc('ric')] = jb.ric
        if 'mean' in risk_metrics:
            df.iloc[i, df.columns.get_loc('mean')] = jb.mean
        if 'std_dev' in risk_metrics:
             df.iloc[i, df.columns.get_loc('std_dev')] = jb.std
        if 'skewness' in risk_metrics:
             df.iloc[i, df.columns.get_loc('skewness')] = jb.skew
        if 'kurtosis' in risk_metrics:
             df.iloc[i, df.columns.get_loc('kurtosis')] = jb.kurt
        if 'var_95' in risk_metrics:
             df.iloc[i, df.columns.get_loc('var_95')] = jb.VaR_95
        if 'cvar_95' in risk_metrics:
             df.iloc[i, df.columns.get_loc('cvar_95')] = jb.CVaR_95
        if 'sharpe' in risk_metrics:
             df.iloc[i, df.columns.get_loc('sharpe')] = jb.sharpe
        if 'jarque_bera' in risk_metrics:
             df.iloc[i, df.columns.get_loc('jarque_bera')] = jb.jarque_bera
        if 'p_value' in risk_metrics:
             df.iloc[i, df.columns.get_loc('p_value')] = jb.p_value
        if 'is_normal' in risk_metrics:
             df.iloc[i, df.columns.get_loc('is_normal')] = jb.is_normal     
        #Remember that you have to return a DataFrame.
    return df


# 

# NO MODIFICAR
def settings(numero_cuenta):
    r = requests.post('http://meva.sytes.net/ulmo/dataHW1.php',  {'numero_cuenta':numero_cuenta} )
    data = json.loads(r.text)
    rics = data['rics']
    risk_metrics = data['risk_metrics']
    df_student = create_dataframe_risk_metrics(rics, risk_metrics)
    print('--------------------------------------------------------')
    data_json = df_student.to_json()
    payload = {'json_data': data_json, 'numero_cuenta': numero_cuenta}
    r = requests.post('http://meva.sytes.net/ulmo/evaluateHW1.php', json=payload)
    print(r.text)

    return(r.text)
