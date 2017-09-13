

import scipy.optimize
import numpy as np

def fit(fun,X,Y):
    try:
        popt,pcurv = scipy.optimize.curve_fit(fun,X,Y)
    except ValueError:
        print("NaN values encountered, fit skipped")
           # input()
        pcurv=[]
        popt=[np.nan]
    except RuntimeError:
        print("Fitting did not converge, arbitrarly chosen to 1")
        pcurv = []
        popt = [np.nan]
        
    return popt

def exp(x,tau):
    return np.exp(x*tau)
    
def parabola(x,a):
    return a*x**2