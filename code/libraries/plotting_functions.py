import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import astropy.io.fits as pyfits
import astropy.wcs as wcs



#### PLOTTING OF ARRAYS ####

## plot two arrays
def plot_two_arrays(arr1, arr2, label_x, label_y, save_path = None, plot_code = 'ro', dpi = 300):
    fig, ax = plt.subplots()
    
    ax.plot(arr1, arr2, plot_code)
    
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    
    if(save_path is not None):
        plt.savefig(save_path, dpi = dpi)
    
    plt.show()


## plot two arrays as well as a function
def plot_two_arrays_and_curve(arr1, arr2, x_curve, y_curve, label_x, label_y, save_path = None, plot_code = 'ro', dpi = 300):
    fig, ax = plt.subplots()
    
    ax.plot(arr1, arr2, plot_code)
    ax.plot(x_curve, y_curve, 'k-')
    
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    
    if(save_path is not None):
        plt.savefig(save_path, dpi = dpi)
    
    plt.show()


    
#### PLOTTING RELATED TO HISTOGRAMS ####

## plot a simple histogram ##
def plot_histogram(arr, label_x, num_bins = None, xscale_log = False, save_path = None, dpi = 300):
    if(num_bins == None):
        num_bins = int(np.sqrt(len(arr))+1)
    hist, bins, _ = plt.hist(arr, bins=num_bins)
    
    if(xscale_log):
        log_bins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
        plt.clf()
        plt.hist(arr, bins=log_bins)
        plt.xscale('log')
    
    plt.xlabel(label_x)
    plt.ylabel("count")
    
    if(save_path is not None):
        plt.savefig(save_path, dpi = dpi)
    
    plt.show()


    
#### PLOTTING RELATED TO HDU_LIST ####

## A function to create contour levels if none were specified (standard: creates 5 contours)
def create_linear_contour_levels(self, min_val, max_val, num_conts = 5):
    step = (1./float(num_conts))*(max_val - min_val)
    levs = [min_val + i*step for i in range(0,num_conts)]
    wids = [0.6 for lev in levs]
        
    return levs, wids


## plotting of the first hdu in a HDUList
## plot_image
def plot_hdu(hdu, label, contour_hdu = None, min_val = None, max_val = None, levs_cont = None, wids_cont = None, plot_lims = None, save_path = None, dpi = 300):
    ## get the information from the hdu
    data = hdu[0].data
    w = wcs.WCS(hdu[0].header)
    
    fig, ax = plt.subplots()
    ax1 = fig.add_subplot(111,projection=w)
    im = ax1.imshow(data, origin='lower', cmap='jet', vmin = min_val, vmax=max_val)
    
    if(contour_hdu is not None):
        data_cont = contour_hdu[0].data
        w_cont = wcs.WCS(contour_hdu[0].header)
        if(levs_cont == None):
            levs_cont, wids_cont = create_linear_contour_levels(np.nanmin(data_cont), np.nanmax(data_cont))
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
        plt.savefig(save_path, dpi = dpi)
        
    plt.show()














