import numpy as np
import os

from astropy.io.fits import HDUList
from astropy.io.fits.hdu.image import PrimaryHDU

#####
# DEFINES THE MOLECULE_HDUList CLASS
#####


#### the Molecule_HDUList class ####      
class Molecule_HDUList(HDUList):
    
    ## the constructor
    def __init__(self, hdus, mol_name, transitions):
        if(len(hdus) != len(transitions)):
            raise ValueError("The number of provided images should match the number of provided transitions.")
        HDUList.__init__(self, hdus = hdus)
        self.mol_name = mol_name
        self.transitions = transitions
    
    
    ##########
    # PUBLIC FUNCTIONS
    ##########
    
    
    ## return an Molecule_HDUList where each pixel that has a nan value also has a nan value in all the other images of the list
    def get_uniform_nans(self):
        ## create a reference data set to identify all pixels that have incomplete data in a loop by setting them to nan
        ref_data = self[0].data
        for i in range(1,len(self)):
            ref_data[np.isnan(self[i].data)] = np.nan
        
        ## create a new list to store images, update the images and store them
        new_list = []
        for i, image in enumerate(self):
            if(i == 0):
                new_list.append(PrimaryHDU(ref_data, image.header))
            else:
                data = image.data
                data[np.isnan(ref_data)] = np.nan
                new_list.append(PrimaryHDU(data, image.header))
        
        return Molecule_HDUList(new_list, self.mol_name, self.transitions)
    
    
    ## verify that all data files in the hdu_list have the same size
    def verify_consistent_shapes(self):
        ref_shape = self[0].data.shape
        
        for hd in self:
            if(hd.data.shape != ref_shape): raise ValueError("The shape of the provided observational data files are not consistent.")
    
    ##########
    # PRIVATE FUNCTIONS
    ##########