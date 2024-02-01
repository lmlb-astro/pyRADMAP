import numpy as np
import pandas as pd
import astropy.io.fits as pyfits

from sklearn.svm import SVR

import os
import random

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
    ## Currently available quantities: "log$_{10}$(n$_{H2}$)","Nmol" and "Tkin"
    ## grid_path: path to the directory
    ## Nmol and Tkin must be a list of floats, MAKE EXTENSION WITH NP.NDARRAY
    def create_dens_SVRregression_model_for_molecule(self, grid_path, im_list_mol, Tkin = None, Nmol = None, plot_verify_fitting = True, test_perc = 30.):
        ## get the training and testing data after verification that fitting can be done
        Xs_train, Ys_train, Xs_test, Ys_test = self.__get_train_test_data(grid_path, im_list_mol, Tkin, Nmol, test_perc)
        
        ## verify if the training data is not too large to be used to create an SVR regression model
        if(len(Xs_train[0]) > high_svr_regressor_grid_num):
            print("The number of data points on the grid is large, the regression might thus be time consuming. If you want to avoid this, make sure that your grid is smaller than: {num}".format(num = high_svr_regressor_grid_num))
        
        ## create the kernel model
        svr = SVR(kernel = kernel_svr, C = C_svr, gamma = gamma_svr, degree = degree_svr, epsilon = epsilon_svr, coef0 = coef0_svr)
        for ys in Ys_train:
            self.models.append(svr.fit(Xs_train, ys))
            self.fitted_quantities.append("log$_{10}$(n$_{H2}$)") # ALLOW THIS TO INCLUDE NMOL AND TKIN
        
        ## verify the regression result 
        self.__verify_regression(plot_verify_fitting, Xs_test, Ys_test)
    
    
    
    ## return predictions over the full map for the constructed regression model
    def predict_map(self, Xs):
        ## Set all pixels to nan where there is no data for all provided transitions in the Image_List 
        Xs = Xs.get_uniform_nans()
        
        ## verify the size of the provided hdus
        Xs.verify_consistent_shapes()
        
        ## get the header and data sizes for the hdus in the hdu_list
        header, len_x, len_y = Xs[0].header, len(Xs[0].data[0]), len(Xs[0].data)
        
        ## get the indices where data should be fitted
        indices = np.where(~np.isnan(Xs[0].data))
        
        ## extract the data from the HDUs into a numpy array to fit
        input_data = []
        for x in Xs:
            input_data.append(x.data[~np.isnan(x.data)])
        input_data = np.array(input_data).transpose()
        
        ## make predictions for each model
        outputs = []
        for model in self.models:
            outputs.append(model.predict(input_data))
        
        ## store the results into an HDU list to return
        output_list = self.__create_HDU_list_from_outputs(outputs, indices, len_x, len_y, header)
            
        return output_list
    
    
    
    #def compare_observed_and_training_data(self, im_list_mol, grid_path, Tkin = None, Nmol = None):
    #    # PLOT COMPARISON
    
    
    
    ##########
    # PRIVATE FUNCTIONS
    ##########
    
    ## Get the X and Y value data at given indices
    def __get_data_subset_from_inds(self, dataX, dataY, indices):
        return np.take(dataX, indices, axis = 0), np.take(dataY, indices, axis = 1)
    
    
    
    ## returns the input data for the regressor
    ## returns y values in the form of a list such that it can be iterated when multiple physical quantities are being fitted
    def __get_input_data(self, grid_path, sorted_file_list, Tkin, Nmol, transitions, test_perc = 30.):
        ## get the data
        Xs = None
        Ys = None
        # CONSIDER IF GET DENSITY FEATURES CAN BE GENERALIZED
        if((Tkin is not None) and (Nmol is not None)):
            Xs, Ys = self.__get_density_features(grid_path, sorted_file_list, Tkin, Nmol, transitions)
        
        ## verify that plausible value is given for test_perc
        if(test_perc >= 100.):
            raise ValueError("The test percentage has to be in the range [0., 100.)")
        elif(test_perc > 50.):
            print("Be aware that with the current input you are using more than half of your data to test the model")
        
        ## split the data in training and testing data
        if(test_perc > 0.):
            ## generate random indices for test and train data set
            num_test_points = int(0.01*test_perc*Xs.shape[0] + 0.5)
            inds_arr = [index for index in range(0, Xs.shape[0])]
            test_inds = np.array(random.sample(inds_arr, num_test_points))
            train_inds = np.delete(np.array(inds_arr), test_inds)
            
            ## take the training and testing data from the right indices
            Xs_train, Ys_train = self.__get_data_subset_from_inds(Xs, Ys, train_inds)
            Xs_test, Ys_test = self.__get_data_subset_from_inds(Xs, Ys, test_inds)
            
            return Xs_train, Ys_train, Xs_test, Ys_test
        
        ## If the training data is None to raise warning
        if(Xs_train is None or Ys_train is None):
            raise ValueError("The training data is empty, model cannot be trained.")
        
        return Xs, Ys, None, None
    
    
    
    ## returns input data for the regressor if only fitting the density
    ## returns y values in the form of a list with one element
    def __get_density_features(self, grid_path, sorted_file_list, Tkin, Nmol, transitions):
        xs_list = []
        ys_arr = None
        
        ## loop over files to find the transitions and Nmol to extract the Tkin
        for file in sorted_file_list:
            result = file.split('_')
            NmF = self._get_Nmol_float(result[2].replace('.dat',''))
            if((result[1] in transitions) and (NmF == Nmol)):
                df = self.read_RADEX_file_into_dataframe(grid_path+file)
                df = df[df["Tkin"] == Tkin]
                if(ys_arr is None):
                    ys_arr = np.array(df["log$_{10}$(n$_{H2}$)"].values)
                xs_list.append(np.array(df["Tmb"].values))
                
        ## convert xs_list into a numpy array
        xs_arr = np.array(xs_list).transpose()
        
        return xs_arr, np.array([ys_arr])
        
    
    
    ## store the outputs into an HDU list to return
    def __create_HDU_list_from_outputs(self, outputs, indices, len_x, len_y, header):
        output_list = []
        
        for output, quantity in zip(outputs, self.fitted_quantities):
            ## update the header
            header['BUNIT'] = quantity
                
            ## initialize storage array
            new_arr = np.zeros((len_y, len_x), dtype = float)
            new_arr[new_arr == 0.] = np.nan
                
            ## store as PrimaryHDU
            for i, (x, y) in enumerate(zip(indices[0], indices[1])):
                new_arr[x][y] = output[i]
            output_list.append(pyfits.PrimaryHDU(new_arr, header))
            
        ## to HDUList format
        output_list = pyfits.HDUList(output_list) 
            
        return output_list
    
    
    
    ## quantify and verify the deviations for a model fit to a given physical quantity
    def __inspect_deviations_fit(self, y_in, y_fit, param):
        ## Calculate the relative deviation for each test point
        rel_dev_arr = []
        for a, b in zip(y_in, y_fit):
            rel_dev_arr.append(np.sqrt((a-b)**2)/a)
        
        ## Get the min and max of the test Y data
        y_min, y_max = np.nanmin(y_in), np.nanmax(y_in)
        
        ## plot the fit as function of the input
        plfuncs.plot_two_arrays_and_curve(y_in, y_fit, [y_min, y_max], [y_min, y_max], "input ({param})".format(param = param), "fit ({param})".format(param = param))
        
        ## plot the deviation of the fit as a function of the input
        plfuncs.plot_two_arrays(y_in, rel_dev_arr, "input ({param})".format(param = param), "relative deviation")
    
    
    
    ## function to get the training and testing data for a model
    def __get_train_test_data(self, grid_path, im_list_mol, Tkin, Nmol, test_perc):
        ## Perform a full verification that the Regressor can create a model
        self._run_verification(grid_path, im_list_mol, Tkin, Nmol) 
        
        ## get the sorted list of RADEX files
        sorted_file_list = self.get_sorted_list_of_radex_files_for_molecule_in_directory(grid_path, im_list_mol.mol_name)
        # USE SORTED FILE LIST ALREADY IN RUN VERIFICATION: AVOID DOUBLE WORK
        
        ## verify that provided Nmol and Tkin are available in the grid path
        bool_Nmol, bool_Tkin = self._verify_Tkin_Nmol(grid_path, sorted_file_list, Tkin, Nmol)
        if(bool_Nmol == False or bool_Tkin == False):
            raise ValueError("The value provided for Nmol or Tkin cannot be found in the grid path.")
        
        ## read the data to train the regression model
        print("Reading data input")
        Xs_train, Ys_train, Xs_test, Ys_test = self.__get_input_data(grid_path, sorted_file_list, Tkin, Nmol, im_list_mol.transitions, test_perc)
        print("----> Done")
        
        return Xs_train, Ys_train, Xs_test, Ys_test
    
    
    
    ## verify the regression result 
    def __verify_regression(self, plot_verify_fitting, Xs_test, Ys_test):
        if(Xs_test is None or Ys_test is None):
            print("WARNING: The testing data is empty, skipping the regression verification.")
            
        elif(plot_verify_fitting):
            ## get the predictions for the training data
            ver_ys = []
            for model in self.models:
                ver_ys.append(model.predict(Xs_test))
            
            ## compare the fit to the input
            for y_in, y_fit, param in zip(Ys_test, ver_ys, self.fitted_quantities):
                self.__inspect_deviations_fit(y_in, y_fit, param)
                
        else:
            print("----> Skipping the regression verification")