import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_two_arrays(arr1, arr2, label_x, label_y, save_path = None):
    fig, ax = plt.subplots()
    
    ax.plot(arr1, arr2, 'ro')
    
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    
    if(save_path is not None):
        plt.savefig(save_path, dpi=300)
    
    plt.show()

def plot_two_arrays_and_curve(arr1, arr2, x_curve, y_curve, label_x, label_y, save_path = None):
    fig, ax = plt.subplots()
    
    ax.plot(arr1, arr2, 'ro')
    ax.plot(x_curve, y_curve, 'k-')
    
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    
    if(save_path is not None):
        plt.savefig(save_path, dpi=300)
    
    plt.show()

def plot_histogram(arr, label_x, num_bins = None, xscale_log = False, save_path = None):
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
        plt.savefig(save_path, dpi=300)
    
    plt.show()


















