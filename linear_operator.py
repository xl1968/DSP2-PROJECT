
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image
from skimage import restoration
from scipy import linalg
import time
import skimage
from math import sqrt,log
from scipy import ndimage

def forward_gradient(img):
    h, w = img.shape
    gradient = np.zeros((h, w, 2), img.dtype)  # Allocate gradient array
    # Horizontal direction
    gradient[:, :-1, 0] = img[:, 1:] - img[:, :-1]
    # Vertical direction
    gradient[:-1, :, 1] = img[1:, :] - img[:-1, :]
    return (gradient)
    ## Return a m*n*2 size array, it contains the gradient of two direciotns 
def backward_divergence(grad):
    
    h, w = grad.shape[:2]
    inv = np.zeros((h, w), grad.dtype)
    # note that we just reversed left and right sides
    # of each line to obtain the transposed operator
    inv[:, :-1] -= grad[:, :-1, 0]
    inv[:, 1: ] += grad[:, :-1, 0]
    inv[:-1]    -= grad[:-1, :, 1]
    inv[1: ]    += grad[:-1, :, 1]
    return inv
    ## This function is the transpose of forward_gradient
    
def compute_linear_operator_L(z):
    ## get the linear operation L
    ## L is defined as: L(p,q)(i,j)=p(i,j)+q(i,j)-p(i-1,j)-q(i,j-1)
    p,q = z
    m,n = p.shape

    r = np.zeros( (m,n), dtype=np.float64 )

    r[1:-1,:] = p[1:-1,:] - p[:-2,:] + q[1:-1,:] - q[:-2,:]
    r[:,1:-1] = p[:,1:-1] - p[:,:-2] + q[:,1:-1] - q[:,:-2]

    r[0,:] += p[0,:]
    r[:,0] += q[:,0]

    return r

def compute_linear_operator_LT(x):
    ## get the linear operatior LT
    ## LT is defined as:
    ## p(i,j)=x(i,j)-x(i+1,j)
    ## q(i,j)=x(i,j)-x(i+1,j)
    m,n = x.shape

    p,q = np.zeros( (m,n),dtype=np.float64 ),np.zeros( (m,n),dtype=np.float64 )
    p[:-1,:] = x[:-1,:] - x[1:,:]
    q[:,:-1] = x[:,:-1] - x[:,1:]

    w = np.zeros( (2,m,n),dtype=np.float64 )

    w[0] = p
    w[1] = q

    return w