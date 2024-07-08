import numpy as np
import pandas as pd
import astropy.io.fits as pyfits

from sklearn.svm import SVR

import os

# IMPORT RADEX_FITTING INSTEAD
from RADEX_fitting import RADEX_Fitting
import plotting_functions as plfuncs

#####
# DEFINES THE REGRESSOR CLASS
# -> Currently: SVR regression with Polynomial kernel
#####



#### definition of global values and parameters ####

## Parameters for SVR regression
high_svr_regressor_grid_num = 10000 ## Value to issue warning of larger grid

kernel_svr = "poly"
C_svr = 100
gamma_svr = "auto"
degree_svr = 2
epsilon_svr = 0.1
coef0_svr = 1
#######################################




#### definition of the Regressor class ####
## -> Holds list of models, e.g. to fit to multiple Nmol or Tkin ranges
class Regressor(RADEX_Fitting):
    
    ## constructor
    def __init__(self):
        RADEX_Fitting.__init__(self)
        self.models = []
        self.fitted_quantities = []
        self.Nmols = []
     
    
    
    ###########
    # PUBLIC FUNCTIONS TO CALL FIT YOUR DATA
    ###########
    
    
    
    ## empty the model list of the Regressor
    # def empty_models():
    
    
    
    ## A function to create an SVR model and simultaneously map the fitted properties
    def map_from_dens_SVRregression(self, grid_path, im_list_mol, Tkin = None, Nmol = None, plot_verify_fitting = True, test_perc = 30.):
        ## create the SVR regression model
        self.create_dens_SVRregression_model_for_molecule(grid_path, im_list_mol, Tkin, Nmol, plot_verify_fitting, test_perc)
        
        ## construct the map
        output_list = self.predict_map(im_list_mol)
        
        return output_list
    
    
    
    ## create a model prediction based on a provided grid path
    ## Currently available quantities: "log$_{10}$(n$_{H2}$)"
    ## grid_path: path to the directory
    ## Nmol and Tkin must be a list of floats
    def create_dens_SVRregression_model_for_molecule(self, grid_path, im_list_mol, Tkin = None, Nmol = None, plot_verify_fitting = True, test_perc = 30.):
        ## sort the Nmol input from small to large values and store the sorted array
        self.Nmols = sorted(Nmol)
        
        ## get the training and testing data after verification that fitting can be done
        Xs_train, Ys_train, Xs_test, Ys_test = self._get_train_test_data(grid_path, im_list_mol, Tkin, self.Nmols, test_perc)
        
        ## verify if the training data is not too large to be used to create an SVR regression model
        if(Ys_train.shape[1] > high_svr_regressor_grid_num):
            print("The number of data points on the grid is large, the regression might be time consuming. If you want to avoid this, make sure that your grid is smaller than: {num}".format(num = high_svr_regressor_grid_num))
        
        ## create the kernel model(s)
        for xs, ys in zip(Xs_train, Ys_train):
            svr = SVR(kernel = kernel_svr, C = C_svr, gamma = gamma_svr, degree = degree_svr, epsilon = epsilon_svr, coef0 = coef0_svr)
            self.models.append(svr.fit(xs, ys))
            self.fitted_quantities.append("log$_{10}$(n$_{H2}$)")
        
        ## verify the regression result 
        if(plot_verify_fitting):
            self.__verify_regression(Xs_test, Ys_test)
        else:
            print("----> Skipping the regression verification")
    
    
    
    ## return predictions over the full map for the constructed regression model
    ## N_map: should be column densities predicted for the given molecule
    def predict_map(self, Xs, N_map = None):
        ## verify that only one model exists if no column density map is provided
        if(N_map is None and len(self.models) > 1):
            raise TypeError('It is not possible to use multiple regression models if no column density map is provided.')
        
        ## Set all pixels to nan where there is no data for all provided transitions in the Image_List 
        Xs = Xs.get_uniform_nans()
        
        ## verify the size of the provided hdus
        Xs.verify_consistent_shapes()
        
        ## get the header and data sizes for the hdus in the hdu_list
        header, len_x, len_y = Xs[0].header, len(Xs[0].data[0]), len(Xs[0].data)
        
        ## get the column density intervals for the different models
        N_intervals = self.__get_N_intervals()
        
        ## make the prediction for each model
        outputs = []
        indices = []
        for idx, model in enumerate(self.models):
        
            ## extract the data from the HDUs into a numpy array to fit
            input_data = []
            inds_col = None
            for j, x in enumerate(Xs):
                ## get the input data within the given column density range
                input_data.append(x.data[(~np.isnan(x.data)) & (N_intervals[idx] < N_map) & (N_map <= N_intervals[idx+1])])
                
                ## store the corresponding indices (only once)
                if(j == 0):
                    inds_col = np.where((~np.isnan(x.data)) & (N_intervals[idx] < N_map) & (N_map <= N_intervals[idx+1]))
            
            ## transpose to the correct data format
            input_data = np.array(input_data).transpose() 
        
            ## make predictions for each model and store the indices
            outputs.append(model.predict(input_data)) 
            indices.append(inds_col)
        
        ## store the results into an HDU list to return
        output_list = self.__create_HDU_list_from_outputs(outputs, indices, len_x, len_y, header) # PROBLEM HERE: MERGE EVERYTHING TOGETHER: IN LOOP?
            
        return output_list
    
    
    
    #def compare_observed_and_training_data(self, im_list_mol, grid_path, Tkin = None, Nmol = None):
    #    # PLOT COMPARISON
    
    
    
    ##########
    # PRIVATE FUNCTIONS
    ##########   
    
    
    ## return a list with the column density intervals for each model
    def __get_N_intervals(self):
        intervals = [float('-inf')]
        
        ## loop over the column densities (self.Nmols) of the models to add intervals
        for idx in range(1, len(self.Nmols)):
            intervals.append(np.nanmean([self.Nmols[idx-1], self.Nmols[idx]]))
            
        ## complete the intervals
        intervals.append(float('inf'))
        
        return intervals
    
    
    
    ## quantify and verify the deviations for a model fit to a given physical quantity
    def __inspect_deviations_fit(self, y_in, y_fit, param, col_dens):
        ## Calculate the relative deviation for each test point
        rel_dev_arr = []
        for a, b in zip(y_in, y_fit):
            rel_dev_arr.append(np.abs(a-b)/a)
        
        ## Get the min and max of the test Y data
        y_min, y_max = np.nanmin(y_in), np.nanmax(y_in)
        
        ## plot the fit verification plots
        plfuncs.plot_fit_verification(y_in, y_fit, rel_dev_arr, [y_min, y_max], 
                                      "input ({param})".format(param = param), 
                                      "fit ({param})".format(param = param), 
                                      "relative deviation",
                                      title_input = "{quan} = {col_dens}x{fact} ({unit})".format(col_dens = col_dens/1e14,
                                                                                                fact = '10$^{14}$',
                                                                                                 unit = 'cm$^{-2}$',
                                                                                                 quan = 'N$_{H_{2}}$'
                                                                                                )
                                     )
    
    
    
    ## verify the regression result 
    def __verify_regression(self, Xs_test, Ys_test):
        ## Verify that the test data exists
        if(Xs_test is None or Ys_test is None):
            print("WARNING: The testing data is empty, skipping the regression verification.")
        ## verify that the number of models match the number of Xs inputs
        elif(Xs_test.shape[0] != len(self.models)):
            raise ValueError("The number of models does not match the number of test data sets")
            
        else:
            ## get the predictions for the training data
            ver_ys = []
            for xs, model in zip(Xs_test, self.models):
                ver_ys.append(model.predict(xs))
            
            ## compare the fit to the input
            for y_in, y_fit, param, col_dens in zip(Ys_test, ver_ys, self.fitted_quantities, self.Nmols):
                self.__inspect_deviations_fit(y_in, y_fit, param, col_dens)
    
    
    
    ## store the outputs into an HDU list to return
    def __create_HDU_list_from_outputs(self, outputs, indices, len_x, len_y, header):
        output_list = []
        
        ## initialize storage array
        new_arr = np.zeros((len_y, len_x), dtype = float)
        new_arr[new_arr == 0.] = np.nan
        
        ## update the header
        header['BUNIT'] = self.fitted_quantities[0]
        ## temporary verification
        for quantity in self.fitted_quantities:
            if(quantity != self.fitted_quantities[0]):
                print('BE CAREFULL WITH THE HEADER OF THE OUTPUT FITS FILE, IT MIGHT NOT CONTAIN THE RIGHT UNITS')
        
        for output, inds in zip(outputs, indices):
                
            ## store numpy ndarray for the PrimaryHDU
            for i, (x, y) in enumerate(zip(inds[0], inds[1])):
                new_arr[x][y] = output[i]
        
        ## store the output PrimaryHDU in
        output_list.append(pyfits.PrimaryHDU(new_arr, header))
            
        ## to HDUList format
        output_list = pyfits.HDUList(output_list) 
            
        return output_list