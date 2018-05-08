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
def L2norm(x):
    return np.sqrt((x*x).sum(-1))

def tvcost_rof_primal_dual(old,img,clambda):
    m, n = img.shape
    cost = np.sum(np.sqrt((img[0:m-1,0:n-1] - img[1:m,0:n-1])**2 + (img[0:m-1,0:n-1] - img[0:m-1,1:n])**2))
    cost += np.sum(np.abs(img[0:m-1,n-1]-img[1:m,n-1]))
    cost += np.sum(np.abs(img[m-1,0:n-1]-img[m-1,1:n]))
    cost += 0.5 * clambda * ((img-old)**2).sum()
    return cost

def tvcost_rof_gradient(old,img,clambda):
    m, n = img.shape
    cost = np.sum(np.sqrt((img[0:m-1,0:n-1] - img[1:m,0:n-1])**2 + (img[0:m-1,0:n-1] - img[0:m-1,1:n])**2))
    cost += np.sum(np.abs(img[0:m-1,n-1]-img[1:m,n-1]))
    cost += np.sum(np.abs(img[m-1,0:n-1]-img[m-1,1:n]))
    return clambda*cost + 0.5*((img-old)**2).sum()

def tvcost_fista(old,img,clambda):
    m, n = img.shape
    cost = np.sum(np.sqrt((img[0:m-1,0:n-1] - img[1:m,0:n-1])**2 + (img[0:m-1,0:n-1] - img[0:m-1,1:n])**2))
    cost += np.sum(np.abs(img[0:m-1,n-1]-img[1:m,n-1]))
    cost += np.sum(np.abs(img[m-1,0:n-1]-img[m-1,1:n]))
    return 2*clambda*cost + ((img-old)**2).sum()
