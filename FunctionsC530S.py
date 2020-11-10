# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 22:59:31 2020

@author: ASUS
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2, linregress
from scipy.optimize import minimize
from numpy import linalg as LA 

# import our own files and reload
import stream_classes8
importlib.reload(stream_classes8)

# Aqui vamos a guardar funciones estáticas, con def definimos funciones
def load_time_series(ric, file_extension = 'csv'): # we will assume that the extension is csv
    # Get market data, load data, we must use \\
    path = 'C:\\Users\ASUS\\Documents\\9no semestre\\Seminario de Finanzas\\Market Data\\' + ric + '.' + file_extension
    if file_extension == 'csv' :
        table_raw = pd.read_csv(path) #default csv
    else:
        table_raw = pd.read_excel(path)
    # Create table of returns
    t = pd.DataFrame() #Empty data fram
    t['date'] = pd.to_datetime(table_raw ['Date'], dayfirst=True) #Set day as the first element of the date
    t['close'] = table_raw['Close'] # We are ceating a new column in the t data frame
    t.sort_values(by= 'date', ascending=True)
    t['close_previous'] = t['close'].shift(1)  # Recorre la celda un renglón, Step down the cell
    t['return_close'] = t['close']/t['close_previous']-1
    t = t.dropna() # To delete na
    t= t.reset_index(drop=True) # We are ordering to reset the index from 0-
    # Input for Jarque-Bera test
    x = t['return_close'].values  # returns as array
    x_str = 'Real returns' + ric # label e.g. ric
    
    return x, x_str, t # The things we want the function to return

def plot_time_series_price(t, ric):
    # Plot timeseries of prices
    plt.figure()
    plt.plot(t['date'], t['close'])
    plt.title('Time series real prices' + ric)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()
    # El return es void es decir no regresa variables
    
def plot_histogram(x, x_str, plot_str, bins=100): # si no le pongo nada la gráfica sale con 100 bins
    #Plot histogram
    plt.figure() #genero la figura
    plt.hist(x,bins) # muestro la figura en el histograma
    plt.title('Histogram '+ x_str)
    plt.xlabel( plot_str)
    plt.show() # we print it or show it, it must go at the end of the characteristics so thar it appears all the elements together

def synchronise_timeseries( ric, benchmark, file_extension= 'csv'):
    # Loading data from csv or Excel file
    x, str1, t1 = load_time_series(ric, file_extension) # creamos t1
    x, str2, t2 = load_time_series(benchmark, file_extension)
     # Synchronize timestamps
    # si imprimimos t1['date'] obtenemos una serie con las fechas
    # para poder comparar las convertimos a listas
    timestamp1 = list(t1['date'].values) # convertimos a un arreglo
    timestamp2 = list(t2['date'].values)
    timestamps = list(set(timestamp1) & set(timestamp2)) # cada timestamp es un conjunto y realizamos intersección
    # Synchronized time series for xq or ric
    t1_sync = t1[t1['date'].isin(timestamps)] # Filtramos solo las fechas que esten en timestamps
    t1_sync.sort_values(by = 'date', ascending=True)
    t1_sync = t1_sync.reset_index(drop=True)
    # Synchronized time series for xq or ric
    t2_sync = t2[t2['date'].isin(timestamps)] # Filtramos solo las fechas que esten en timestamps
    t2_sync.sort_values(by = 'date', ascending=True)
    t2_sync = t2_sync.reset_index(drop=True)
    # Table of returns for ric and benchmark
    t = pd.DataFrame()
    t['date'] = t1_sync['date'] #Guardamos fechas
    t['price_1'] = t1_sync['close'] # Guardamos precio t1 de close en columna llamada price_1
    t['price_2'] = t2_sync['close']
    t['return_1'] = t1_sync['return_close']
    t['return_2'] = t2_sync['return_close']
    
    # Compute vectors of returns
    returns_ric = t['return_1'].values #y #Convertimos a arreglos de floats
    returns_benchmark = t['return_2'].values 
   
    return returns_benchmark, returns_ric, t     #x, y, t


def compute_beta(ric, benchmark, bool_print=False):
    capm = stream_classes8.capm_manager(ric, benchmark)
    capm.load_timeseries()
    capm.compute()
    if bool_print:
        print('------')
        print(capm)
    beta = capm.beta
    return beta

def cost_function_beta_delta(x, delta, beta_usd, betas, epsilon=0.0):
    f_delta = (sum(x).item() + delta)**2 # Tomamos a x y lo sumamos y lo convertimos en escalar
    f_beta = (np.transpose(betas).dot(x).item() + beta_usd)**2 # Nos da matriz de 1x1
    f_penalty = epsilon * sum(x**2).item() #Para cuando nos dan valores muy grandes en optimal_hedge
    f = f_delta + f_beta + f_penalty
    return f