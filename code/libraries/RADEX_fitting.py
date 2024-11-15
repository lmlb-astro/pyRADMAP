import numpy as np
import pandas as pd

import os
import random

import matplotlib.pyplot as plt

#####
# DEFINES THE RADEX FITTING CLASS
#####

########
# PUBLIC FUNCTIONS
########


## definition of the RADEX_Fitting class
class RADEX_Fitting():
    
    ## constructor
    def __init__(self):
        self.available_phys_quantities = ["log$_{10}$(n$_{H2}$)", "Nmol", "Tkin"]
    
    
    
    ########
    # PUBLIC FUNCTIONS
    ########
    
    
    ## return the list of files for this specific molecule in the directory
    def get_sorted_list_of_radex_files_for_molecule_in_directory(self, grid_path, mol_name):
        ## get full list of .dat files
        files = self.__get_list_of_dat_files_in_directory(grid_path)
    
        ## get files that contain the molecule of interest in this class
        mol_files = [f for f in files if f.split('_')[0] == mol_name]
    
        return sorted(mol_files)
    
    
    
    ## read a RADEX file into a DataFrame
    def read_RADEX_file_into_dataframe(self, path_file):
        try:
            df = pd.read_csv(path_file, sep="\s+", names = ["Tkin","log$_{10}$(n$_{H2}$)","Tmb"])
        except:
            raise NameError("Could not read or find the file.")
    
        return df

    
    
    #######
    # PROTECTED FUNCTIONS
    #######
    
    
    ## convert Nmol from string into a float and integer
    def _get_Nmol_float(self, Nmol):
        if('p' in Nmol):
            Nmol = Nmol.replace('p','.')
        NmolF = float(Nmol)
        return NmolF
    
    
    
    ## returns the input data for the regressor
    ## returns y values in the form of a list such that it can be iterated when multiple physical quantities are being fitted
    def _get_input_data(self, grid_path, sorted_file_list, Tkin, Nmol, transitions, test_perc = 30.):
        ## get the data
        Xs, Ys = None, None
        if((Tkin is not None) and (Nmol is not None)): Xs, Ys = self.__get_density_features(grid_path, sorted_file_list, Tkin, Nmol, transitions)
        
        ## verify that plausible value is given for test_perc
        self.__check_test_perc(test_perc)
        
        ## split the data in training and testing data
        if(test_perc > 0.):
            return self.__split_train_test(Xs, Ys, test_perc)
        
        ## If the training data is None to raise warning
        if(Xs_train is None or Ys_train is None):
            raise ValueError("The training data is empty, model cannot be trained.")
        
        return Xs, Ys, None, None
    
    
    
    ## function to get the training and testing data for a model
    def _get_train_test_data(self, grid_path, im_list_mol, Tkin, Nmol, test_perc):
        ## get the sorted list of RADEX files for a given molecule
        sorted_file_list = self.get_sorted_list_of_radex_files_for_molecule_in_directory(grid_path, im_list_mol.mol_name)
        
        ## read the data to train the regression model
        print("Reading data input")
        Xs_train, Ys_train, Xs_test, Ys_test = self._get_input_data(grid_path, sorted_file_list, Tkin, Nmol, im_list_mol.transitions, test_perc)
        print("----> Done")
        
        return Xs_train, Ys_train, Xs_test, Ys_test
    
    
    ########
    # PRIVATE FUNCTIONS
    #########
    
    ## return the list of '.dat' files in a provided directory
    def __get_list_of_dat_files_in_directory(self, grid_path):
        files = [f for f in os.listdir(grid_path) if (os.path.isfile(os.path.join(grid_path, f)) and f.split('.')[-1] == 'dat')]
    
        return files

    
    
    ## Get the X and Y value data at given indices
    def __get_data_subset_from_inds(self, dataX, dataY, indices):
        return np.take(dataX, indices, axis = 1), np.take(dataY, indices, axis = 1)
    
    
    
    ## split the data into training and testing data sets (both for Xs and Ys)
    def __split_train_test(self, Xs, Ys, test_perc):
        ## generate random indices for test and train data set
        num_test_points = int(0.01*test_perc*Ys.shape[1] + 0.5)
        inds_arr = np.arange(Ys.shape[1]) 
        test_inds = np.array(random.sample(list(inds_arr), num_test_points))
        train_inds = np.delete(inds_arr, test_inds) 
            
        ## take the training and testing data from the right indices
        Xs_train, Ys_train = self.__get_data_subset_from_inds(Xs, Ys, train_inds)
        Xs_test, Ys_test = self.__get_data_subset_from_inds(Xs, Ys, test_inds)
        
        return Xs_train, Ys_train, Xs_test, Ys_test
    
    
    
    ## function that verifies that the fraction of data points used for testing is valid
    def __check_test_perc(self, test_perc):
        if(test_perc >= 100. or test_perc < 0.):
            raise ValueError("The test percentage has to be in the range [0., 100.)")
        elif(test_perc > 50.):
            print("Be aware that with the current input you are using more than half of your data to test the model")
            
            
    
    ## Returns a hashmap that connects the float column density value to the string value in the file names
    def __get_Nmol_dict(self, sorted_file_list):
        ## initiate the hashmap
        Nmol_dict = dict()
        
        ## loop over all the files
        for file in sorted_file_list:
            ## get the column density value
            result = file.split('_')
            Nmol = result[2].replace('.dat','')
            Nmol_fl = self._get_Nmol_float(Nmol)
            
            ## update the hash map
            if(Nmol_fl not in Nmol_dict):
                Nmol_dict[Nmol_fl] = Nmol
        
        return Nmol_dict
    
    
    
    ## Function that verifies that all Tkin values in a list are found in the "Tkin" column of a DataFrame
    ## Raises an error if a given temperature is not found in the DataFrame
    def __verify_Tkin_prensence(self, df, Tkins):
        ## loop over the Tkins
        for Tkin in Tkins:
            if(Tkin not in df['Tkin'].values):
                raise ValueError("The temperature {Tkin} K is not in the RADEX output file.".format(Tkin = Tkin))
    
    
    
    ## Function that reads a single RADEX output file and stores the values from the file in a DataFrame for the Tkin values of interest (given in the Tkins input)
    ## returns the DataFrame
    def __get_XY_from_file(self, grid_path, file_name, Tkins):
        ## read the data file (if the file does not exist, this is handled in the read_RADEX_... function)
        df = self.read_RADEX_file_into_dataframe(grid_path+file_name)
        
        ## verify that all Tkin values of interest are contained in the data files
        self.__verify_Tkin_prensence(df, Tkins)
                
        ## select the rows with Tkin values that are present in Tkins
        df = df.loc[df['Tkin'].isin(Tkins)]
        
        return df
    
    
    ## returns input data for the regressor if only fitting the density based on a single molecule
    ## sorted_file_list only contains files with the give molecule
    ## returns y values in the form of a list with one element
    def __get_density_features(self, grid_path, sorted_file_list, Tkins, Nmols, transitions):
        ## initiate return list
        xs_list, ys_list = [], []
        
        ## initiate the name of the molecule
        mol_name = sorted_file_list[0].split('_')[0]
        
        ## create a dictionary (or hashmap) that connects the float to string values for all column density values in the sorted_file_list
        Nmol_dict = self.__get_Nmol_dict(sorted_file_list)
        
        ## loop over each column density
        for Nmol in Nmols:
            ## get Nmol in string format
            Nmol_str = Nmol_dict[Nmol]
            
            ## temporary storage for each transition
            xs_temp, ys_temp = [], None
        
            ## loop over each transition necessary
            for tr in transitions:
                print(mol_name)
                print(tr)
                ## create the file name
                file_name = '{mol}_{tr}_{nm}.dat'.format(mol = mol_name, tr = tr, nm = Nmol_str)
                
                ## get the X and Y values of interest in a DataFrame
                df = self.__get_XY_from_file(grid_path, file_name, Tkins)
                
                ## append the data points of interest to the x_list and ys_list
                ## only add to the ys_list when the Y-values have not been stored yet. (multiple transitions -> one density)
                xs_temp.append(np.array(df["Tmb"].values))
                if(ys_temp is None): ys_temp = np.array(df["log$_{10}$(n$_{H2}$)"].values)
                
            ## ravel the temp arrays and add them to the return arrays
            xs_list.append(np.array(xs_temp).transpose()) 
            ys_list.append(np.array(ys_temp))
            
                
        ## convert xs_list into a numpy array
        xs_arr = np.array(xs_list)
        ys_arr = np.array(ys_list)
        
        return xs_arr, ys_arr 