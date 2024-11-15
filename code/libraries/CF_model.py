import numpy as np
from scipy.optimize import curve_fit


####  CLASS THAT DEFINES A MODEL BASED ON CURVE_FIT ####
## This allows to better understand the parameters of the regression model than the sklearn libraries

class CF_model():
    
    ## Constructor
    def __init__(self, degree = 3):
        ## store the polynomial degree
        self.degree = degree
        # SELECT FITTING FUNCTION HERE
        
        
        ## store the fitting parameters
        self.popt = None
        self.pcov = None
        
        # PASS THE NUMBER OF PARAMETERS
        
        
    #### functions to fit ####
    
    ## quadratic polynomial function
    def quad_func(self, x, a, b, c):
        if(len(x.shape) == 1): return a*x**2 + b*x + c
        elif(len(x.shape) == 2):
            x_sum = np.nansum(x, axis = 0)
            return a*x_sum**2 + b*x_sum + c
        else: return None
    
    ## cubed polynomial function
    def cube_func(self, x, a, b, c, d):
        if(len(x.shape) == 1): return a*x**3 + b*x**2 + c*x + d
        elif(len(x.shape) == 2):
            x_sum = np.nansum(x, axis = 0)
            return a*x_sum**3 + b*x_sum**2 + c*x_sum + d
        else: return None
        
    
    
    ## fit the model
    ## input shape: x -> (k, M); y -> (M)
    def fit(self, x, y):
        if self.degree == 2: self.popt, self.pcov = curve_fit(self.quad_func, x, y)
        elif self.degree == 3: self.popt, self.pcov = curve_fit(self.cube_func, x, y)
        else: raise NameError('Could not fit a polynomial of the requested degree')
        
    
    ## make prediction for the model
    ## input shape: x -> (k, M)
    def predict(self, x):
        if self.popt is None: raise AttributeError("No model parameters have been fitted.") 
        elif self.degree == 2: return self.quad_func(x, *self.popt)
        elif self.degree == 3: return self.cube_func(x, *self.popt)
        else: raise NameError('Could not fit a polynomial of the requested degree')
    
        