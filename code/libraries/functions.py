import numpy as np
import sys
import matplotlib.pyplot as plt

#######################################
## Definitions of mathematical functions
## -> Gaussian functions (Up to 3 Gaussians)
## -> Polynomial functions (Up to 4th Order)
#######################################

#### DEFINITIONS TO FIT GAUSSIANS #####

## the 1d gaussian for fitting (x0: central velocity, sigma: width)
def gaussian1d(x,A,x0,sigma):
	return A*np.exp(-(x-x0)**2/(2.*sigma**2))

## a double 1d gaussian for fitting
def doubleGaussian1d(x,A1,x1,sigma1,A2,x2,sigma2):
	return A1*np.exp(-(x-x1)**2/(2.*sigma1**2)) + A2*np.exp(-(x-x2)**2/(2.*sigma2**2))

## a double 1d gaussian for fitting
def tripleGaussian1d(x,A1,x1,sigma1,A2,x2,sigma2,A3,x3,sigma3):
	return A1*np.exp(-(x-x1)**2/(2.*sigma1**2)) + A2*np.exp(-(x-x2)**2/(2.*sigma2**2)) + A3*np.exp(-(x-x3)**2/(2.*sigma3**2))

## get the result of a function (specified by 'func')
## works for gaussian1d, doubleGaussian1d, and tripleGaussian1d so far
def getGaussianFunction(func,x,inputVals):
	if(func == gaussian1d):
		return [func(xi,inputVals[0],inputVals[1],inputVals[2]) for xi in x]
	elif(func == doubleGaussian1d):
		return [func(xi,inputVals[0],inputVals[1],inputVals[2],inputVals[3],inputVals[4],inputVals[5]) for xi in x]
	elif(func == tripleGaussian1d):
		return [func(xi,inputVals[0],inputVals[1],inputVals[2],inputVals[3],inputVals[4],inputVals[5],inputVals[6],inputVals[7],inputVals[8]) for xi in x]
	else:
		sys.exit('You asked for a guassian that does not exist, so the program can not continue')

################################################


#### DEFINITIONS TO FIT POLYNOMIAL FUNCTIONS #####

def sec_ord_poly(x, a, b, c):
    return a*x**2 + b*x + c

def third_ord_poly(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

def fourth_ord_poly(x, a, b, c, d, e):
    return a*x**4 + b*x**3 + c*x**2 + d*x + e

def sec_ord_poly_min_yval(x, a, b, c, yval):
    return a*x**2 + b*x + c - yval

def third_ord_poly_min_yval(x, a, b, c, d, yval):
    return a*x**3 + b*x**2 + c*x + d - yval

def fourth_ord_poly_min_yval(x, a, b, c, d, e, yval):
    return a*x**4 + b*x**3 + c*x**2 + d*x + e - yval


def get_poly_function(order):
    if(order == 2):
        return sec_ord_poly
    elif(order == 3):
        return third_ord_poly
    elif(order == 4):
        return fourth_ord_poly
    else:
        print("Could not find the requested polynomial function")

def get_poly_function_min_yval(order):
    if(order == 2):
        return sec_ord_poly_min_yval
    elif(order == 3):
        return third_ord_poly_min_yval
    elif(order == 4):
        return fourth_ord_poly_min_yval
    else:
        print("Could not find the requested polynomial function")

####################################################


#### PLOTTING ####

## plots the components of the double and triple gaussians
def plotGaussianComponents(x,inputVals):
	if(len(inputVals)%3 == 0):
		for a in range(0,len(inputVals),3):
			gaussArr = [gaussian1d(xi,inputVals[a],inputVals[a+1],inputVals[a+2]) for xi in x]
			plt.plot(x,gaussArr,'g-')
	else:
		sys.exit('No correct input was given to plot the gaussian components')
