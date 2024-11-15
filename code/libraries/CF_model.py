import numpy as np
from scipy.optimize import curve_fit


####  CLASS THAT DEFINES A MODEL BASED ON CURVE_FIT ####
## This allows to better understand the parameters of the regression model than the sklearn libraries

class CF_model():
    
    ## Constructor
    def __init__(self, degree = 3):
        ## Create a dictionary of the polynomial functions and verify the input
        self.func_dict = {2: self.__quad_func, 3: self.__cube_func}
        if degree not in self.func_dict: raise KeyError('A polynomial function with this degree is not available.')
        
        ## store the polynomial degree
        self.degree = degree
        self.cf_func = self.func_dict[degree]
        
        ## store the fitting parameters
        self.popt = None
        self.pcov = None
        
        # PASS THE NUMBER OF PARAMETERS
        
        
        
        
    #### functions to fit ####
    
    ## quadratic polynomial function
    def __quad_func(self, x, a, b, c):
        if(len(x.shape) == 1): return a*x**2 + b*x + c
        elif(len(x.shape) == 2):
            x_sum = np.nansum(x, axis = 0)
            return a*x_sum**2 + b*x_sum + c
        else: return None
    
    ## cubed polynomial function
    def __cube_func(self, x, a, b, c, d):
        if(len(x.shape) == 1): return a*x**3 + b*x**2 + c*x + d
        elif(len(x.shape) == 2):
            x_sum = np.nansum(x, axis = 0)
            return a*x_sum**3 + b*x_sum**2 + c*x_sum + d
        else: return None
        
        
    
    #### Public functions of the class
    
    ## fit the model
    ## input shape: x -> (k, M); y -> (M)
    def fit(self, x, y):
        self.popt, self.pcov = curve_fit(self.cf_func, x, y)
        
    
    ## make prediction for the model
    ## input shape: x -> (k, M)
    def predict(self, x):
        if self.popt is None: raise AttributeError("No model parameters have been fitted.") 
        else: return self.cf_func(x, *self.popt)
    
        