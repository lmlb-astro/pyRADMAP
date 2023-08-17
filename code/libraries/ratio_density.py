import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import astropy.wcs as wcs

import pandas as pd
from scipy.optimize import brentq
import itertools

import os

from image import Astro_Image
import plotting_functions as plfunc
import fitting_functions as fitfunc
import functions as funcs

#######
# DEFINES THE RATIO_MAP CLASS
#######

####
# Make reading and writing files more robust
####


## Create a map of ratios from maps that cover the same area and have the same size
class ratio_map(Astro_Image):
    def __init__(self, data1, data2, header = None):
        try:
            ratio = data1/data2
            Astro_Image.__init__(self, ratio, header)
        except:
            print("Cannot create a ratio map, verify that both maps have the same size.")

    #### functions to work on the data ####
    
    def create_ratio_dataframe(self, df_1, df_2):
        df_3 = pd.DataFrame(df_1["Tmb"]/df_2["Tmb"])
        df_3.columns = ["T_ratio"]
        df_3.insert(1, "nH2", df_1["nH2"])
        df_3.insert(2, "Tkin", df_1["Tkin"])
        
        return df_3
    
    ## construct the of the brightness temperature ratio as a function of density
    def get_ratio_curve(self, x_vals, popt):
        temp_func = funcs.get_poly_function(len(popt)-1) ## len(popt)-1 gives order of the polynomial
        curve_vals = temp_func(x_vals, *popt)
        
        return curve_vals
    
    ## create the input function for the brentq method
    def calculate_density(self, data, x_beg, x_end, popt):
        vals = np.zeros((len(data), len(data[0])), dtype = float)
        vals[vals==0.] = np.nan
        indices = np.where(~np.isnan(data))
        for x, y in zip(indices[1], indices[0]):
            vals[y][x] = brentq(funcs.get_poly_function_min_yval(len(popt)-1), x_beg, x_end, args = (*popt, data[y][x]))
            
        return vals
    
    ##  verify that all values in a list are the same (returns boolean, if not equal: returns False)
    def verify_equality_in_list(self, ver_list): 
        verify_bool = True
        verification_val = ver_list[0]
        for val in ver_list:
            if((verification_val - val) != 0):
                verify_bool = False
        
        return verify_bool
    
    ## return the available column densities in a given directory (return a list and dictionary with the directories)
    def get_colDens_list_from_files(self, grid_path):
        ## get all the .dat filenames
        files = [f for f in os.listdir(grid_path) if (os.path.isfile(os.path.join(grid_path, f)) and f.split('.')[-1] == 'dat')]
        
        ## store the column densities in a list and verify the amount
        return_list = []
        file_names = {}
        count_list = []
        for f in files:
            colDens = f.split('_')[-1].split('.')[0] ## exctract the column density from the file name
            colDens = float(colDens.replace('p','.'))
            if(colDens not in return_list):
                return_list.append(colDens)
                file_names[colDens] = [os.path.join(grid_path, f)]
                count_list.append(1)
            else:
                ind = return_list.index(colDens) ## verify the number of data files with this 
                count_list[ind] += 1
                file_names[colDens].append(os.path.join(grid_path, f))
        
        ## verify that there are the same number of files for each column density (MAKE THIS INTO A FUNCTION)
        if(self.verify_equality_in_list(count_list) == False):
            print("There might be missing files for the analysis, please verify the directory with your grid of RADEX results.")
        
        return return_list, file_names
    
    ## return the available temperatures
    def get_Tkin_list_from_directories(self, file_names_dict):
        list_keys = list(file_names_dict.values())
        directories_list = list(itertools.chain.from_iterable(list_keys))
        
        Tkin_list = []
        length_files = []
        
        for i, directory in enumerate(directories_list):
            readFile = open(directory)
            lines = readFile.readlines()
            readFile.close()
            length_files.append(len(lines))
            
            if(i == 0):
                for line in lines:
                    result = line.split()
                    Tkin = float(result[0])
                    if(Tkin not in Tkin_list):
                        Tkin_list.append(Tkin)
        
        ## verify equal length of all the files
        if(self.verify_equality_in_list(length_files) == False):
            print("The RADEX result files do not have the same length, please verify the directory with your grid of RADEX results.")
        
        return Tkin_list
    
    ## find the nearest value in an array for a give values
    def find_nearest_value(self, array, value):
        idx = (np.abs(array - value)).argmin()
        
        return array[idx]
    
    ## find the nearest values over the full map for a given list of options
    def get_nearest_map(self, real_map, options_list):
        ## convert list to array
        options = np.array(options_list)
        
        ## create a new map to store the nearest available value
        new_map = real_map.copy()
        diff_map = new_map.copy()
        
        for x in range(0, len(real_map[0])):
            for y in range(0, len(real_map)):
                val = real_map[y][x]
                if(~np.isnan(val)):
                    new_map[y][x] = self.find_nearest_value(options, val)
        
        diff_map = real_map - new_map
        
        return new_map, diff_map
    
    ## read a RADEX file into a dataframe
    def read_file_into_dataframe(self, path_file, column_names):
        df = pd.read_csv(path_file, sep="\s+", names = column_names)
        
        return df
    
    ## verify that the DataFrames are self-consistent
    def verify_df_self_consistent(self, df_1, df_2):
        if(df_1.shape != df_2.shape):
            print("The RADEX files for the two molecular lines do not have the same size and can thus not calculate the density.")
        if(df_1["nH2"].equals(df_2["nH2"]) == False):
            print("The RADEX files for the two molecular lines do not have the same densities and can thus not calculate the density in the region.")
            
    ## fit polynomial to the ratio as a function of density in a DataFrame
    def fit_poly_density_to_DataFrame(self, df, poly_order):
        popt, pcov = fitfunc.fit_polynomial(df["nH2"], df["T_ratio"], order=poly_order)
        
        return popt, pcov
    
    ## get the fitted curve to the ratio as a function of density (returns two arrays)
    def get_curve_vals(self, df, popt):
        x_curve = np.linspace(df["nH2"][0], df["nH2"][len(df)-1],len(df))
        curve_vals = self.get_ratio_curve(x_curve, popt)
        
        return x_curve, curve_vals
    
    ## remove all the values from a copy of a ratio map outside a given value range (< min_rat, > max_rat)
    def __get_ratio_map_copy_range_cut(self, min_rat, max_rat, ratio_copy):
        ratio_copy[ratio_copy<min_rat] = np.nan
        ratio_copy[ratio_copy>max_rat] = np.nan
        
        return ratio_copy
    
    ## remove all the values from a copy of a ratio map based on the upper and lower value of a curve
    def __get_min_max_from_fitted_ratio_curve(self, curve_vals):
        min_rat = np.nanmin(curve_vals); max_rat = np.nanmax(curve_vals)
        #print("The minimal ratio that allows a solution for the give input is: " + str(np.round(min_rat,2)))
        #print("The maximal ratio that allows a solution for the give input is: " + str(np.round(max_rat,2)))
        
        return min_rat, max_rat
    
    ## calculate the density in each pixel of the given copy of the ratio map for which a density can be calculated
    def get_density_in_pixels_from_curve_fitting(self, ratio_copy, df, poly_order, mol, line_1, line_2, run_num = 0, num_verify_ratio_fit = 20, save_path=None):
            popt, pcov = self.fit_poly_density_to_DataFrame(df, poly_order)
            x_curve, curve_vals = self.get_curve_vals(df, popt)
            if(run_num%num_verify_ratio_fit == 0):
                plfunc.plot_two_arrays_and_curve(df["nH2"], df["T_ratio"], x_curve, curve_vals, 'n$_{H_{2}}$', mol+'('+line_1+')/'+mol+'('+line_2+')', save_path = save_path)
        
            ## Clean the ratio map so that the density can be calculated in every pixel
            min_rat, max_rat = self.__get_min_max_from_fitted_ratio_curve(curve_vals)
            ratio_copy = self.__get_ratio_map_copy_range_cut(min_rat, max_rat, ratio_copy)
        
            ## calculate the density for each pixel in the map using the brentq method
            n_beg = x_curve[np.where(curve_vals == min_rat)[0]]
            n_end = x_curve[np.where(curve_vals == max_rat)[0]]
            log_dens_map = self.calculate_density(ratio_copy, n_beg, n_end, popt)
            
            return log_dens_map
    
    ## sort the paths appropriately
    def __get_sorted_paths(self, line_1, line_2, paths):
        val_1 = int(line_1.split('-')[0])
        val_2 = int(line_2.split('-')[0])
        paths.sort()
        if(val_1 > val_2):
            paths.sort(reverse = True)
        
        return paths
    
    ## plot the deviation of the nearest map with respect to the original map
    def plot_deviation(self, diff_map, im_original, label_1, label_2):
        rel_diff = np.abs(100.*diff_map/im_original.astro_image)
        
        im_diff = Astro_Image(diff_map, self.header)
        im_diff.plot_image(label_1)
        im_rel_diff = Astro_Image(rel_diff, self.header)
        im_rel_diff.plot_image(label_2)
     
    ## store density map into an image
    def dens_map_to_image(self, log_dens_map, log_bool):
        ## adapt the unit based on the data that will be saved
        bunit = "log[n$_{H_{2}}$ (cm$^{-3}$)]"
        if(log_bool == False):
            log_dens_map = 10**log_dens_map
            bunit = "n$_{H_{2}}$ (cm$^{-3}$)"
        
        ## store the density map as an Astro_Image, first adapt header
        header_density = self.header.copy()
        header_density['BUNIT'] = bunit
        density_map_image = Astro_Image(log_dens_map, header = header_density)
        
        return density_map_image, bunit
    
    #### Public functions that can be called to return a density map ####
    
    ## calculate the density over the map for a single molecular column density, kinetic temperature and fwhm
    def get_density_ratio_single_temperature(self, mol, line_1, line_2, mol_col_dens, t_kin, fwhm, poly_order=2, log_bool=True, save_path_hist = None, save_path_ratio = None):
        
        ## get the data for both molecular lines in data frames
        df_1 = self.read_file_into_dataframe('../Radex/bin/results/T='+t_kin+'K-FWHM='+fwhm+'/'+mol+'_'+line_1+'_'+mol_col_dens+'.dat', ["Tkin","nH2","Tmb"])
        df_2 = self.read_file_into_dataframe('../Radex/bin/results/T='+t_kin+'K-FWHM='+fwhm+'/'+mol+'_'+line_2+'_'+mol_col_dens+'.dat', ["Tkin","nH2","Tmb"])
        
        ## verify that the data is the same size and have same densities
        self.verify_df_self_consistent(df_1, df_2)
        
        ## create new data frame with the ratio
        df_3 = self.create_ratio_dataframe(df_1, df_2)
        
        ## create a copy of the ratio map
        ratio_copy = self.astro_image.copy()
        
        ## fit polynomial to the ratio as a function of density, get the fitted curve and plot the result
        log_dens_map = self.get_density_in_pixels_from_curve_fitting(ratio_copy, df_3, poly_order, mol, line_1, line_2, save_path=save_path_ratio)
        
        ## create an Astro_Image version of the density map
        density_map_image, bunit = self.dens_map_to_image(log_dens_map, log_bool)
        
        ## plot histogram of the density distribution
        plfunc.plot_histogram(density_map_image.astro_image.ravel(), bunit, xscale_log = not log_bool, save_path = save_path_hist)
        
        return density_map_image
    
    
    ## calculate the density over the map for a given fwhm, while the temperature and column density map are given
    def get_density_ratio_from_colDens(self, mol, line_1, line_2, fwhm, im_colDens, im_Tdust, grid_path, Tdust_corr = 0, poly_order=2, verify_deviation = True, log_bool=True, save_path_hist = None, save_path_ratio = None, num_verify_ratio_fit = 20):
        ## create a map where density results will be stored
        log_dens_map = np.zeros((len(self.astro_image),len(self.astro_image[0])), dtype = float)
        log_dens_map[log_dens_map == 0] = np.nan
        
        ## get a list of the available column densities in a given directory
        colDens_list, file_names_dict = self.get_colDens_list_from_files(grid_path)
        
        ## get the temperature options of the grid
        Tkin_list = self.get_Tkin_list_from_directories(file_names_dict)
        
        ## create a temperature corrected imTdust
        im_Tdust_corr = Astro_Image(im_Tdust.astro_image + Tdust_corr, im_Tdust.header)
        
        ## get a 'RADEX' Tkin and column density map from the real map
        colDens_nearest, colDens_diff = self.get_nearest_map(im_colDens.astro_image, colDens_list)
        Tkin_nearest, Tkin_diff = self.get_nearest_map(im_Tdust_corr.astro_image, Tkin_list)
        
        ## verify the relative deviations for the temperature and column density
        if(verify_deviation):
            self.plot_deviation(colDens_diff, im_colDens, '$\Delta$ N (cm$^{-2}$)', r'$\frac{\Delta N}{N}$ (%)')
            self.plot_deviation(Tkin_diff, im_Tdust_corr, '$\Delta$ T (K)', r'$\frac{\Delta T}{T}$ (%)')
        
        ## loop over list of column densities and temperatures to fill up the density map
        for i, colDens in enumerate(colDens_list):
            ## create the mask
            mask = np.zeros((len(colDens_nearest),len(colDens_nearest[0])), dtype = int)
            mask[colDens_nearest == colDens] = 1
            
            ## get the paths of the files and sort the file_paths to avoid inversing the ratio (+ it shows the column density being handled)
            file_paths = file_names_dict[colDens]
            file_paths = self.__get_sorted_paths(line_1, line_2, file_paths)
            
            ## read both files into a DataFrame and verify they are consistent
            df_1 = self.read_file_into_dataframe(file_paths[0], ["Tkin","nH2","Tmb"])
            df_2 = self.read_file_into_dataframe(file_paths[1], ["Tkin","nH2","Tmb"])
            self.verify_df_self_consistent(df_1, df_2)
            
            ## create new data frame with the ratio
            df_3 = self.create_ratio_dataframe(df_1, df_2)
            
            for j, Tkin in enumerate(Tkin_list):
                ## add temperature to mask and apply it to a copy of the ratio map
                mask_1 = mask.copy()
                mask_1[Tkin_nearest != Tkin] = 0
                ratio_copy = self.astro_image.copy()
                ratio_copy[mask_1 == 0] = np.nan
                
                ## select only the relevant temperature out of the DataFrame
                df_4 = df_3[df_3["Tkin"] == Tkin]
                df_4 = df_4.reset_index()
                
                log_dens_map_local = self.get_density_in_pixels_from_curve_fitting(ratio_copy, df_4, poly_order, mol, line_1, line_2, run_num = j, num_verify_ratio_fit = num_verify_ratio_fit, save_path=save_path_ratio)
                
                ## store results in log_dens_map
                indices = np.where(~np.isnan(log_dens_map_local))
                for x, y in zip(indices[1], indices[0]):
                    log_dens_map[y][x] = log_dens_map_local[y][x]
                
        ## create an Astro_Image version of the density map
        density_map_image, bunit = self.dens_map_to_image(log_dens_map, log_bool)
        
        ## plot histogram of the density distribution
        plfunc.plot_histogram(density_map_image.astro_image.ravel(), bunit, xscale_log = not log_bool, save_path = save_path_hist)
        
        return density_map_image
    
    
    
    ## calculate the density over the map for a given fwhm, while the temperature is given and the column density map of the molecule is derived from a column density map and a relative molecular abundance
    def get_density_ratio_from_colDens_and_abundance(self, mol, line_1, line_2, fwhm, mol_abundance, im_colDens, im_Tdust, grid_path, Tdust_corr = 0, poly_order=2, verify_deviation = True, log_bool=True, save_path_hist = None, save_path_ratio = None, num_verify_ratio_fit = 20):
        ## create the image of the molecule column density map
        im_mol_colDens = im_colDens.get_mol_colDens(mol_abundance)
        
        density_map_image = self.get_density_ratio_from_colDens(mol, line_1, line_2, fwhm, im_mol_colDens, im_Tdust, grid_path, Tdust_corr = Tdust_corr, poly_order = poly_order, verify_deviation = verify_deviation, log_bool=log_bool, save_path_hist = save_path_hist, save_path_ratio = save_path_ratio)
        
        return density_map_image
        





