import numpy as np
import astropy.io.fits as pyfits
import astropy.wcs as wcs
import matplotlib.pyplot as plt

from astropy.io.fits.hdu.image import PrimaryHDU

import os
import plotting_functions as plfunc

############
# DEFINES THE COLDENS_PrimaryHDU CLASS
# -> Provides column density specific operations on the data
############


##### DEFINITION OF THE COLDENS_PrimaryHDU CLASS ######
class ColDens_PrimaryHDU(PrimaryHDU):
    
    ## initialization
    def __init__(self, data, header):
        PrimaryHDU.__init__(self, data = data, header = header)
    
    
    
    #### PUBLIC FUNCTIONS ####
    
    ## returns an astro_image of a molecular line column density based on the relative given molecular abundance
    def get_mol_colDens(self, rel_mol_abundance):
        colDens_rel = self.data*rel_mol_abundance
        return_phdu = ColDens_PrimaryHDU(colDens_rel, self.header)
        
        return return_phdu
    
    
    
    ## returns the minimal and maximal molecular column density value over a column density map, based on the molecular abundance
    def get_mol_colDens_range(self, rel_mol_abundance = 1.):
        colDens_rel = self.data*rel_mol_abundance
        min_colDens = np.nanmin(colDens_rel)
        max_colDens = np.nanmax(colDens_rel)
        
        return min_colDens, max_colDens
    
    
    
    ## plots a histogram of the molecular column density values over a Herschel map, based on the molecular abundance
    def histogram_mol_colDens(self, bunit, rel_mol_abundance = 1., log_xscale_bool = False):
        colDens_rel = self.data*rel_mol_abundance
        plfunc.plot_histogram(colDens_rel.ravel(), bunit, xscale_log = log_xscale_bool)

    
    