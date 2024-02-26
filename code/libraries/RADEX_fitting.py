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
    
    ## verify that the grid path contains data for all provided files
    def verify_molecular_data_in_grid(self, transitions, sorted_file_list):
        all_in_grid = False
        
        ## get the list of files for the grid
        #file_list = self.get_sorted_list_of_radex_files_for_molecule_in_directory(grid_path, mol_name)
        
        ## verify the molecule and the available lines in the data
        tr_copy = transitions.copy()
        len_temp_list = 0
        for i, tr in enumerate(transitions):
            temp_list = [f for f in sorted_file_list if f.split('_')[1] == tr]
            if(len(temp_list) != 0):
                if(i == 0):
                    len_temp_list = len(temp_list)
                    tr_copy.remove(tr)
                elif(len_temp_list == len(temp_list)):
                    tr_copy.remove(tr)
        
        ## update boolean
        if(len(tr_copy) == 0):
            all_in_grid = True
            
        return all_in_grid
    
    
    
    ## return the list of files for this specific molecule in the directory
    def get_sorted_list_of_radex_files_for_molecule_in_directory(self, grid_path, mol_name):
        ## get full list of .dat files
        files = self.__get_list_of_dat_files_in_directory(grid_path)
    
        ## get files that contain the molecule of interest in this class
        mol_files = [f for f in files if f.split('_')[0] == mol_name]
        print(mol_files)
    
        return sorted(mol_files)
    
    
    
    ## read a RADEX file into a DataFrame
    def read_RADEX_file_into_dataframe(self, path_file):
        df = pd.read_csv(path_file, sep="\s+", names = ["Tkin","log$_{10}$(n$_{H2}$)","Tmb"])
    
        return df

    
    
    #######
    # PROTECTED FUNCTIONS
    #######
    
    ## perform a full verification that it will be able to derive the physical quantities for the chosen fitting option
    def _run_verification(self, im_list_mol, Tkin, Nmol, sorted_file_list):
        
        ## verify that the grid path contains data for all provided files
        verification_grid = self.verify_molecular_data_in_grid(im_list_mol.transitions, sorted_file_list)
        if(verification_grid == False):
            raise ValueError("The provided path does not possess all the necessary data. Be sure to verify the files in the provided path.")
        else:
            print("Grid verification: passed")
        
        ## verify that sufficient lines are provided for the number of quantities that will be fitted
        self.__check_requested_quantities(len(im_list_mol), Tkin, Nmol)
        
    
    
    ## verify that the given kinetic temperature and molecular column density are present in the files of the given directory
    def _verify_Tkin_Nmol(self, grid_path, sorted_file_list, Tkin, Nmol):
        
        ## verify the boolean for Nmol
        bool_Nmol = True
        if(Nmol is not None):
            bool_Nmol = self.__verify_Nmol(sorted_file_list, Nmol)
        
        ## verify the boolean for Tkin
        bool_Tkin = True
        if(Tkin is not None):
            bool_Tkin = self.__verify_Tkin(grid_path, sorted_file_list, Tkin)
            
        return bool_Nmol, bool_Tkin
    
    
    
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
    
    
    
    ## Verify that Nmol is located in the file list
    def __verify_Nmol(self, sorted_file_list, Nmol):
        ## define the return value
        bool_Nmol = False
        
        ## create lists of the transitions and column density
        list_tr = []
        list_Nmol = []
        for file in sorted_file_list:
            result = file.split('_')
            list_tr.append(result[1])
            list_Nmol.append(result[2])
        
        ## transform the transitions to a set
        set_tr = set(list_tr)
        
        ## Remove the transitions from set_tr if they have the required column density
        for tr, Nm in zip(list_tr, list_Nmol):
            ## create float and int version
            Nm = Nm.replace('.dat','')
            NmF = self._get_Nmol_float(Nm)
            
            if(NmF == Nmol):
                set_tr.remove(tr)
        
        ## verify that all transitions have been removed from set_tr
        if(len(set_tr) == 0):
            bool_Nmol = True
            
        return bool_Nmol
    
    
    
    ## verify that Tkin is located in the file list
    def __verify_Tkin(self, grid_path, sorted_file_list, Tkin):
        ## define the return value
        bool_Tkin = False
        
        ## count the number of files that contain the necessary temperature
        count = 0
        for file in sorted_file_list:
            df = self.read_RADEX_file_into_dataframe(grid_path+file)
            if(Tkin in df["Tkin"].values):
                count += 1
        
        ## verify the count is equal to the length of the file list
        if(len(sorted_file_list) == count):
            bool_Tkin = True
        
        return bool_Tkin
        

