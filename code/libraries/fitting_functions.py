import numpy as np
from scipy.optimize import curve_fit

import functions as funcs

## fit a second order polynomial to a data set
def fit_polynomial(x, y, order):
    popt, pcov = curve_fit(funcs.get_poly_function(order), x, y)
    
    return popt, pcov