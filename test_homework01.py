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

import stream_functions
importlib.reload(stream_functions)
import stream_classes
importlib.reload(stream_classes)


import homework_01 as hw;

'''
NO OLVIDES PONER TU NUMERO DE CUENTA! Debe ir como Integer
'''
numero_cuenta = '313603620'

def test_1():
    print(".-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-")
    result = hw.settings(numero_cuenta)[0:4]
    print("En el primer test obtuviste", result)
    assert float(result) == 10.0

def test_2():
    print(".-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-")
    result = hw.settings(numero_cuenta)[0:4]
    print("En el segundo test obtuviste", result)
    assert float(result) == 10.0


def test_3():
    print(".-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-")
    result = hw.settings(numero_cuenta)[0:4]
    print("En el tercer test obtuviste", result)
    assert float(result) == 10.0
