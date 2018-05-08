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
from cost_function import L2norm

def operatorF(p,r=1.0):
    # This is a proximity operator, it is equivalent to implicit gradient descent step for a function.
    n_p = np.maximum(1.0, L2norm(p) / r)
    return p / n_p[..., np.newaxis]

def operatorC(img):
    ## get the orthogonal projection operator onto the set C
    m,n = img.shape
    img[img>1] = 1
    img[img<0] = 0

    return img


def operatorP(x):
    ## get the orthogonal projection operator onto the set P
    p,q = x[0],x[1]
    m,n = p.shape

    ONES = np.ones( (m,n), dtype=np.float64)

    #From i=1...m-1 and j=1...n-1
    norm1 = (p[:-1,:-1]**2 + q[:-1,:-1]**2)**0.5
    norm1p_max = np.maximum(ONES[:-1,:-2],norm1[:,:-1])

    #From i=1...m-1 and j=n (Border case)
    norm2p = np.absolute(p[:-1,n-2])
    norm2p_max = np.maximum(ONES[:-1,0],norm2p)

    p[:-1,:-2] = p[:-1,:-2]/norm1p_max
    p[:-1,n-2] = p[:-1,n-2]/norm2p_max

    norm1q_max = np.maximum(ONES[:-2,:-1],norm1[:-1,:])

    #From i=m and j=1...n-1 (Border case)
    norm2q = np.absolute(q[m-2,:-1])
    norm2q_max = np.maximum(ONES[0,:-1],norm2q)

    q[:-2,:-1] = q[:-2,:-1]/norm1q_max
    q[m-2,:-1] = q[m-2,:-1]/norm2q_max

    w = np.zeros( (2,m,n), dtype=np.float64 )
    w[0] = p
    w[1] = q

    return w

##These are three kinds of proximal opreator, they performs
##like the impliciting of gradient descent step for a function