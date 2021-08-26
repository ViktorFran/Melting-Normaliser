#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 20:31:03 2021

@author: Takuma Kume
"""
from matplotlib import pyplot as plt
import lmfit
from numpy import exp, sqrt, cbrt
import pyperclip

#-----------------------------------------------------------------------------
# Model Functions
#-----------------------------------------------------------------------------
R = 8.31446 # J/K*mol
cK = 273.15 # Conv Celcius to Kelvin
def biMolecularDuplex(x, N, a, D, b, H, Tm):
    return N+a*x+((b-a)*x+D-N)*(0.25*exp(H/R*(1/(cK+Tm)-1/(cK+x)))*((8*exp(H/R*(1/(cK+x)-1/(cK+Tm)))+1)**(0.5)-1))

def tetraMolecular(x, N, a, D, b, H, Tm):
    C = 8
    K = exp((H/R)*((1/(Tm+cK))-(1/(x+cK))))
    u = cbrt(sqrt(3)*sqrt(256*(C**3)*(K**3)+27*(C**2)*(K**4))+9*C*(K**2))
    u2 = cbrt(18)*C
    u3 = 4*cbrt(2/3)*K
    u4 = ((-8*(-4*C-K))/C)-32
    u5 = ((u/u2)-(u3/u))
    v = (1/2)*sqrt(u5)
    w = (1/2)*sqrt((u4)/(4*sqrt(u5))-u5)
    return N+a*x+((b-a)*x+D-N)*(v - w + 1)
    # return (v - w + 1)

def monoMolecular(x, N, a, D, b, H, Tm):
    return N + a*x + ((b-a)*x+D-N)/(1+ exp((H/R)*(1/(Tm+cK)-1/(x+cK))))

while True:
    # =============================================================================
    # Select the model you want to use
    
    selectedModel = lmfit.Model(monoMolecular)
    
    # =============================================================================
    
    # print(selectedModel.param_names)
    # print(selectedModel.independent_vars)
    
    print()
    print("Input Data: ")
    data = input()
    
    # Data cleaning to X Y variables 
    x = []
    y = []
    for line in data.splitlines():
        line_element = line.split()
        if len(line_element) == 2:
            x.append(float(line_element[0]))
            y.append(float(line_element[1]))
    
    # ----------------------------------------------------------------------------
    # add(name, value=None, vary=True, min=- inf, max=inf, expr=None, brute_step=None)
    #    Add a Parameter:
    #         name (str) – Name of parameter. Must match [a-z_][a-z0-9_]* and cannot be a Python reserved word.
    #         value (float, optional) – Numerical Parameter value, typically the initial value.
    #         vary (bool, optional) – Whether the Parameter is varied during a fit (default is True).
    #         min (float, optional) – Lower bound for value (default is -numpy.inf, no lower bound).
    #         max (float, optional) – Upper bound for value (default is numpy.inf, no upper bound).
    #         expr (str, optional) – Mathematical expression used to constrain the value during the fit (default is None).
    #         brute_step (float, optional) – Step size for grid points in the brute method (default is None).
    # ----------------------------------------------------------------------------
    
    HGuess = 200000
    TmGuess = 60
    goodFit = False
    while not goodFit:
        params = selectedModel.make_params()
        params.add('N', 0.1)
        params.add('a', 0.001)
        params.add('D', 0.1)
        params.add('b', 0.001)
        params.add('H', HGuess)
        params.add('Tm', TmGuess)
    
    # ----------------------------------------------------------------------------
    #  Model.fit(data, params=None, weights=None, method='leastsq', iter_cb=None, scale_covar=True, verbose=False, fit_kws=None, nan_policy=None, calc_covar=True, max_nfev=None, **kwargs)
    #     Fit the model to the data using the supplied Parameters.
    #     Parameters
    #             data (array_like) – Array of data to be fit.
    #             params (Parameters, optional) – Parameters to use in fit (default is None).
    #             weights (array_like, optional) – Weights to use for the calculation of the fit residual (default is None). Must have the same size as data.
    #             method (str, optional) – Name of fitting method to use (default is ‘leastsq’).
    #             iter_cb (callable, optional) – Callback function to call at each iteration (default is None).
    #             scale_covar (bool, optional) – Whether to automatically scale the covariance matrix when calculating uncertainties (default is True).
    #             verbose (bool, optional) – Whether to print a message when a new parameter is added because of a hint (default is True).
    #             fit_kws (dict, optional) – Options to pass to the minimizer being used.
    #             nan_policy ({'raise', 'propagate', 'omit'}, optional) – What to do when encountering NaNs when fitting Model.
    #             calc_covar (bool, optional) – Whether to calculate the covariance matrix (default is True) for solvers other than ‘leastsq’ and ‘least_squares’. Requires the numdifftools package to be installed.
    #             max_nfev (int or None, optional) – Maximum number of function evaluations (default is None). The default value depends on the fitting method.
    #             **kwargs (optional) – Arguments to pass to the model function, possibly overriding parameters.
    # ----------------------------------------------------------------------------
        
        fitted = selectedModel.fit(data=y, params=params, x=x, method='leastsq')
        print()
        print("Fit Success?: " + str(fitted.success))
        print(fitted.fit_report(show_correl=False))
    
    # ============================================================================
    # Uncomment below to show fit curve (make sure it is indented 8 spaces)    
    
        plt.scatter(x, y, marker=".")
        plt.plot(x, fitted.best_fit, linestyle='-', color='red')
        plt.show()
    
    # ============================================================================    
        
        isitgood = input("Good Fit? Y/n ") or "Y"
        if isitgood in {"Y", "Yes", "y", "yes"}:
            goodFit = True
        else:
            HGuess = float(input("H Guess?: "))
            TmGuess = float(input("Tm Guess?: "))
    
    #print(fitted.best_values)
    fit_N = fitted.best_values['N']
    fit_a = fitted.best_values['a']
    fit_D = fitted.best_values['D']
    fit_b = fitted.best_values['b']
    
    for i in range(len(y)):
        yi = y[i]
        xi = x[i]
        y[i] = (yi - fit_N - (fit_a * xi)) / (((fit_b - fit_a) * xi) + fit_D - fit_N)
    
    # ============================================================================
    # Uncomment below to show normalised curve (make sure it is indented 4 spaces)
    
    plt.scatter(x, y, marker=".")
    plt.show()
    
    # ============================================================================ 
    
    print()
    print("Normalised Values: ")
    print()    
    result = ""
    for i in range(len(y)):
        resultline = str(x[i]) + "\u0009" + str(y[i]) + '\n'
        result = result + resultline
        
    print(result)
    pyperclip.copy(result)
    
