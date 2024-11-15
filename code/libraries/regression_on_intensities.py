import numpy as np
import pandas as pd
import astropy.io.fits as pyfits

from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge

import os

import matplotlib.pyplot as plt
import seaborn as sns

# IMPORT RADEX_FITTING INSTEAD
from RADEX_fitting import RADEX_Fitting
import plotting_functions as plfuncs
from CF_model import CF_model







#####
# DEFINES THE REGRESSOR CLASS
# -> Currently: SVR regression with Polynomial kernel
#####



#### definition of global values and parameters ####

## Parameters for SVR regression
high_svr_regressor_grid_num = 10000 ## Value to issue warning of larger grid

kernel_t = "poly"
degree_t = 2
coef0_t = 1
C_t = 100.
gamma_t = "auto"

epsilon_svr = 0.1

SEED = 9843
#######################################




#### definition of the Regressor class ####
## -> Holds list of models, e.g. to fit to multiple Nmol or Tkin ranges
class Regressor(RADEX_Fitting):
    
    ## constructor
    def __init__(self, method = 'SVR'):
        RADEX_Fitting.__init__(self)
        self.models = []
        self.fitted_quantities = []
        self.Nmols = []
        self.model_data_x = []
        
        ## verify the input
        if method != 'SVR' and method != 'KR' and method != 'CF': raise AttributeError('{m} is not a supported model type'.format(m = method))
        self.method = method
     
    
    
    ###########
    # PUBLIC FUNCTIONS TO FIT YOUR DATA
    ###########
    
    
    
    ## empty the model list of the Regressor
    # def empty_models():
    
    
    
    ## A function to create a model and simultaneously map the fitted properties
    def map_from_dens_regression(self, grid_path, im_list_mol, Tkin = None, Nmol = None, plot_verify_fitting = True, test_perc = 30., N_map = None, interpolate = False, plot_ver_in = False):
        ## create the regression model
        self.create_dens_regression_model_for_molecule(grid_path, im_list_mol, Tkin = Tkin, Nmol = Nmol, plot_verify_fitting = plot_verify_fitting, test_perc = test_perc)
        
        ## construct the map
        output_list = self.predict_map(im_list_mol, N_map = N_map, interpolate = interpolate, plot_ver_in = plot_ver_in)
        
        return output_list
    
    
    
    ## create a model prediction based on a provided grid path
    ## Currently available quantities: "log$_{10}$(n$_{H2}$)"
    ## grid_path: path to the directory
    ## Nmol and Tkin must be a list of floats
    def create_dens_regression_model_for_molecule(self, grid_path, im_list_mol, Tkin = None, Nmol = None, plot_verify_fitting = True, 
                                                  test_perc = 30., store_RADEX_data = True):
        ## sort the Nmol input from small to large values and store the sorted array
        self.Nmols = sorted(Nmol)
        
        ## get the training and testing data after verification that fitting can be done
        Xs_train, Ys_train, Xs_test, Ys_test = self._get_train_test_data(grid_path, im_list_mol, Tkin, self.Nmols, test_perc)
        
        ## verify if the training data is not too large to be used to create a regression model
        if(Ys_train.shape[1] > high_svr_regressor_grid_num):
            print("The number of data points on the grid is large, the regression might be time consuming. If you want to avoid this, make sure that your grid is smaller than: {num}".format(num = high_svr_regressor_grid_num))
        
        ## create the kernel model(s) ##
        for xs, ys in zip(Xs_train, Ys_train):
            
            ## SVR 
            if self.method == 'SVR':
                svr = SVR(kernel = kernel_t, 
                          C = C_t, 
                          gamma = gamma_t, 
                          degree = degree_t, 
                          epsilon = epsilon_svr, 
                          coef0 = coef0_t,
                          #random_state = SEED
                         )
                self.models.append(svr.fit(xs, ys))
                self.fitted_quantities.append("log$_{10}$[n$_{H2}$ (cm$^{-3}$)]")
            
            ## Kernel Ridge
            if self.method == 'KR':
                kr = KernelRidge(kernel = kernel_t,
                                 degree = degree_t,
                                 alpha = 0.5/C_t, ## see sklearn SVR and KernelRidge documentation
                                 #gamma = gamma_t,
                                 coef0 = coef0_t,
                                 #random_state = SEED
                                )
                self.models.append(kr.fit(xs, ys))
                self.fitted_quantities.append("log$_{10}$[n$_{H2}$ (cm$^{-3}$)]")
                
            ## CF
            if self.method == 'CF':
                cf = CF_model(degree_t)
                cf.fit(xs.T, ys) ## Transpose for curve_fit input
                self.models.append(cf)
                self.fitted_quantities.append("log$_{10}$[n$_{H2}$ (cm$^{-3}$)]")
                
            
            ## store the RADEX data that created the regression model (useful for later visualization)
            if store_RADEX_data: self.model_data_x.append(xs.T) ## transpose for shape (k, M) with k: features, M: num. of samples
                
        
        ## verify the regression result 
        if(plot_verify_fitting):
            self.__verify_regression(Xs_test, Ys_test)
        else:
            print("----> Skipping the regression verification")
    
    
    
    ## return predictions over the full map for the constructed regression model
    ## N_map: should be column densities predicted for the given molecule
    ## Interpolate: whether or not to interpolate to create the density map
    def predict_map(self, Xs, N_map = None, interpolate = False, plot_ver_in = False):
        ## verify that only one model exists if no column density map is provided
        if(N_map is None and len(self.models) > 1):
            raise TypeError('It is not possible to use multiple regression models if no column density map is provided.')
        
        ## Set all pixels to nan where there is no data for all provided transitions in the Image_List 
        Xs = Xs.get_uniform_nans()
        
        ## verify the size of the provided hdus
        Xs.verify_consistent_shapes()
        
        ## get the header and data sizes for the hdus in the hdu_list
        header, len_x, len_y = Xs[0].header, len(Xs[0].data[0]), len(Xs[0].data)
        
        ## predict the densities (either using interpolation or not)
        indices, outputs = self.__pred_dens_from_models(Xs, N_map, interpolate, plot_ver_in = plot_ver_in)#, indices, outputs)
        
        ## store the results into an HDU list to return
        output_list = self.__create_HDU_list_from_outputs(outputs, indices, len_x, len_y, header) 
            
        return output_list
        
    
    
    
    ##########
    # PRIVATE FUNCTIONS
    ##########   
    
    ## return a list with the column density intervals for each model
    def __get_N_intervals(self, interpolate):
        ## verification in the case of interpolate
        if(interpolate and len(self.Nmols) < 3):
            raise ValueError('Performing interpolation requires models for at least 3 different column densities')
        
        ## start the interval at 0
        intervals = [0.]
        
        ## define the end of the loop depending on whether interpolation will be used
        end_loop = len(self.Nmols)
        if(interpolate): end_loop = len(self.Nmols) - 1
        
        ## loop over the column densities (self.Nmols) of the models to create a list of densities that defines the interval
        for idx in range(1, end_loop):
            if(interpolate): intervals.append(self.Nmols[idx])
            else: intervals.append(np.nanmean([self.Nmols[idx-1], self.Nmols[idx]]))
            
        ## complete the intervals
        intervals.append(float('inf'))
        
        return intervals
    
    
    
    ## Predict the densities for an input np.array for a given model (defined by its idx in self.models)
    ## Option: use the interpolation, which employs the provided column density information
    def __make_pred_for_array(self, idx, input_data, N_data, N1, N2, interpolate):
        ## Transpose the input data for the curve_fit option
        if(self.method == 'CF'): input_data = input_data.T
        
        ## take the model for prediction
        model1 = self.models[idx-1]
        pred1 = model1.predict(input_data)
        
        ## the case where interpolation has to be done
        if(interpolate):
            ## verify N_data
            if(N_data is None): raise TypeError("No column density data is provided. Interpolation is thus not possible")
            
            ## make second model prediction
            model2 = self.models[idx]
            pred2 = model2.predict(input_data)
            
            ## perform the interpolation
            pred = (N_data - N1)*(pred2 - pred1)/(N2 - N1) + pred1
            
            return pred
        
        return pred1
    
    
    
    
    def __plot_input_verification(self, idx, in_data, label_x = None, label_y = None):
        input_vals = self.x_inputs[idx]
        
        ## plot the RADEX data
        print(idx)
        rad_data = self.model_data_x[idx]
        plt.plot(rad_data[0], rad_data[1], 'ro')
        
        #p1 = plt.plot([], [], '-', label = 'fit data')
        #sns.kdeplot(x = input_vals[:, 0], y = input_vals[:, 1], fill = False, color = p1[0].get_color())
        #p2 = plt.plot([], [], '-', label = 'data')
        #sns.kdeplot(x = in_data[:, 0], y = in_data[:, 1], fill = False, color = p2[0].get_color())
        
        if label_x is not None: plt.xlabel(label_x)
        if label_y is not None: plt.ylabel(label_y)
        
        plt.legend()
        plt.show()
    
    
    ## predict the density over the provided map using the available models
    ## Option: Use the interpolation approach or not
    def __pred_dens_from_models(self, Xs, N_map, interpolate, plot_ver_in = False): #, indices, outputs
        ## verify that the number of models is equal to the number of column densities
        if(len(self.models) != len(self.Nmols)):
            raise ValueError('The number of models and column densities are inconsistent.')
        
        ## get the column density intervals for the different models
        N_intervals = self.__get_N_intervals(interpolate)
        
        ## initialize return lists
        outputs, indices = [], []
        
        ## set the end value for the loop
        end_loop = len(self.Nmols) + 1
        if(interpolate):
            end_loop = len(self.Nmols)
        
        ## make the predictions
        for idx in range(1, end_loop):
            input_data, inds_col, N_data = [], None, None
            
            ## loop over the molecular lines
            for j, x in enumerate(Xs):
                ## get the input data within the given column density range from the provided HDUs
                input_data.append(x.data[(~np.isnan(x.data)) & (N_intervals[idx-1] < N_map) & (N_map <= N_intervals[idx])])
                
                ## store the corresponding indices once
                if(j == 0):
                    inds_col = np.where((~np.isnan(x.data)) & (N_intervals[idx-1] < N_map) & (N_map <= N_intervals[idx]))
                    ## in the case of interpolation, add the column density data
                    if(interpolate):
                        N_data = N_map[(~np.isnan(x.data)) & (N_intervals[idx-1] < N_map) & (N_map <= N_intervals[idx])]
                        
            ## transpose to the correct data format
            input_data = np.array(input_data).transpose()
            
            ## Verify the value distribution of the input and the model
            if plot_ver_in: self.__plot_input_verification(idx-1, input_data)
            
            ## make predictions for the density and append them to the return lists
            pred = self.__make_pred_for_array(idx, input_data, N_data, N_intervals[idx-1], N_intervals[idx], interpolate)
            outputs.append(pred)
            indices.append(inds_col)
            
        return indices, outputs
    
    
    
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
            ## transpose data if using 'CF' method
            if(self.method == 'CF'): Xs_test = [xs.T for xs in Xs_test]
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
                print('BE CAREFUL WITH THE HEADER OF THE OUTPUT FITS FILE, IT MIGHT NOT CONTAIN THE RIGHT UNITS')
        
        for output, inds in zip(outputs, indices):
                
            ## store numpy ndarray for the PrimaryHDU
            for i, (x, y) in enumerate(zip(inds[0], inds[1])):
                new_arr[x][y] = output[i]
        
        ## store the output PrimaryHDU in
        output_list.append(pyfits.PrimaryHDU(new_arr, header))
            
        ## to HDUList format
        output_list = pyfits.HDUList(output_list) 
            
        return output_list