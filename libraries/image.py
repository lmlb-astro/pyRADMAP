import numpy as np
import astropy.io.fits as pyfits
import astropy.wcs as wcs
import matplotlib.pyplot as plt
from reproject import reproject_interp

import os
import plotting_functions as plfunc

############
# DEFINES THE ASTRO_IMAGE CLASS
# DEFINES THE COLDENS_IMAGE SUBCLASS
############


##### DEFINITION OF THE ASTRO_IMAGE CLASS ######
class Astro_Image():
    
    def __init__(self, image, header = None):
        self.astro_image = image
        
        self.w = None
        self.header = header
        if(self.header is not None):
            self.w = wcs.WCS(self.header)    
        #print(self.w)
    
    #### plotting functions for an image ####
    
    ## set image data values to nan based on mask
    def mask_values(self, mask):
        self.astro_image[mask==1] = np.nan
    
    ## A function to create contour levels if none were specified (standard: creates 5 contours)
    def create_linear_contour_levels(self, min_val, max_val, num_conts = 5):
        step = (1./float(num_conts))*(max_val - min_val)
        levs = [min_val + i*step for i in range(0,num_conts)]
        wids = [0.6 for lev in levs]
        
        return levs, wids
        
    
    ## plotting of the image
    def plot_image(self, label, contour_hdu = None, min_val = None, max_val = None, levs_cont = None, wids_cont = None, plot_lims = None, save_path = None):
        fig, ax = plt.subplots()
        ax1 = fig.add_subplot(111,projection=self.w)
        im = ax1.imshow(self.astro_image, origin='lower', cmap='jet', vmin = min_val, vmax=max_val)
        
        if(contour_hdu is not None):
            data_cont = contour_hdu[0].data
            header_cont = contour_hdu[0].header
            w_cont = wcs.WCS(header_cont)
            if(levs_cont == None):
                levs_cont, wids_cont = self.create_linear_contour_levels(np.nanmin(data_cont), np.nanmax(data_cont))
            ax1.contour(data_cont, colors = 'k', levels=levs_cont,linewidths=wids_cont,transform=ax1.get_transform(w_cont))
            
        if(plot_lims is not None):
            plt.xlim([plot_lims[0],plot_lims[1]])
            plt.ylim([plot_lims[2],plot_lims[3]])

        plt.xlabel('RA [J2000]')
        plt.ylabel('DEC [J2000]',labelpad=-1.)

        cbar = fig.colorbar(im)
        cbar.set_label(label, labelpad=15.,rotation=270)

        ax.axis('off')
        
        if(save_path is not None):
            plt.savefig(save_path, dpi=300)
        
        plt.show()
    
    #### other functions for an astro image ####
    
    ## save the image as fits file
    def save_fits(self, path_save, name_save, BUNIT_label = None, overwrite_bool = True):
        ## create header if possible
        header_temp = None
        if(self.header is not None):
            header_temp = self.header.copy()
        
        ## update the BUNIT label of the header
        if(BUNIT_label is not None):
            try:
                header_temp['BUNIT'] = BUNIT_label
            except:
                print("There is no header, make sure to add a header. The file was saved with a basic header.")
        
        ## verify if the plotting directory exists and create if not
        if(os.path.isdir(path_save) == False):
            os.mkdir(path_save)
    
        ## save the fits file
        if(header_temp is not None):
            newHDU = pyfits.PrimaryHDU(self.astro_image,header_temp)
        else:
            newHDU = pyfits.PrimaryHDU(self.astro_image)
        newHDU.writeto(path_save+name_save,overwrite = overwrite_bool)
        
    ## reproject the image to a given target header using the reproject_interp function from astropy reproject (return as astro_image)
    def reproject_image(self, target_header):
        ## print a warning if the header is None
        if(self.header is None):
            print("The header has a non-value such that a reprojection is not possible. Please provide a header to the Astro Image")
        
        ## preform the reprojection
        tempHDU = pyfits.PrimaryHDU(self.astro_image, self.header)
        image_rep, footprint = reproject_interp(tempHDU, target_header)
        
        ## transform the repojected image into an Astro_Image
        target_header['BUNIT'] = self.header['BUNIT']
        return_image = Astro_Image(image_rep, target_header)
        
        return return_image


        
        
        
        
        
        
        
        
##### DEFINITION OF THE COLDENS_IMAGE CLASS ######
class ColDens_Image(Astro_Image):
    
    ## initialization
    def __init__(self, image, header = None):
        Astro_Image.__init__(self, image, header = header)
    
    
    #### functions to operate on the column density image ####
    
    ## returns an astro_image of a molecular line column density based on the relative given molecular abundance
    def get_mol_colDens(self, rel_mol_abundance):
        colDens_rel = self.astro_image*rel_mol_abundance
        return_im = ColDens_Image(colDens_rel, self.header)
        
        return return_im
    
    ## returns the minimal and maximal molecular column density value over a column density map, based on the molecular abundance
    def get_mol_colDens_range(self, rel_mol_abundance):
        colDens_rel = self.astro_image*rel_mol_abundance
        min_colDens = np.nanmin(colDens_rel)
        max_colDens = np.nanmax(colDens_rel)
        
        return min_colDens, max_colDens
    
    ## plots a histogram of the molecular column density values over a Herschel map, based on the molecular abundance
    def histogram_mol_colDens(self, rel_mol_abundance, bunit, log_xscale_bool = False):
        colDens_rel = self.astro_image*rel_mol_abundance
        plfunc.plot_histogram(colDens_rel.ravel(), bunit, xscale_log = log_xscale_bool)
        
    #### override functions ####
    
    ## override the reproject image class to return a ColDens_Image object
    def reproject_image(self, target_header):
        ## print a warning if the header is None
        if(self.header is None):
            print("The header has a non-value such that a reprojection is not possible. Please provide a header to the Astro Image")
        
        ## preform the reprojection
        tempHDU = pyfits.PrimaryHDU(self.astro_image, self.header)
        image_rep, footprint = reproject_interp(tempHDU, target_header)
        
        ## transform the repojected image into an Astro_Image
        target_header['BUNIT'] = self.header['BUNIT']
        return_image = ColDens_Image(image_rep, target_header)
        
        return return_image