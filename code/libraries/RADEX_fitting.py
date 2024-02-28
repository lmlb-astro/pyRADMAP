import numpy as np
import pandas as pd

import os

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
    
    
    
    ########
    # PRIVATE FUNCTIONS
    #########
    
    ## return the list of '.dat' files in a provided directory
    def __get_list_of_dat_files_in_directory(self, grid_path):
        files = [f for f in os.listdir(grid_path) if (os.path.isfile(os.path.join(grid_path, f)) and f.split('.')[-1] == 'dat')]
    
        return files
    
    
    
    ## verify that the quantities are correct and can be returned
    def __check_requested_quantities(self, len_list, Tkin, Nmol):
        
        ## counts the provided input and calculate the number of parameters that will be fitted from this 
        input_len = 0
        if(Tkin is not None):
            input_len += 1
        if(Nmol is not None):
            input_len += 1
        num_fitted_params = len(self.available_phys_quantities) - input_len
        print(num_fitted_params)
        
        ## verify that the requested number of physical quantities is not higher than the provided number of transitions
        if(num_fitted_params > len_list):    
            raise ValueError("Not enough spectral line transitions are provided to fit the requested number of physical quantities. Make sure that the requested number of physical quantities is smaller than or equal to the number of transitions provided.")

