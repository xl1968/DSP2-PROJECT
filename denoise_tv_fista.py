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
from proximal_operator import operatorC,operatorP
from linear_operator import compute_linear_operator_L,compute_linear_operator_LT
from cost_function import tvcost_fista


lena = Image.open('lena512.bmp')
mat_lena=np.asarray(lena)
img_ref = mat_lena[140:,120:][:256,:256] / 255.0
plt.imshow(img_ref,cmap='gray')
plt.show()

img_ref = mat_lena[140:,120:][:256,:256] / 255.0
#img_noise = skimage.util.random_noise(img_ref, mode='gaussian')
#img_noise2 = skimage.util.random_noise(img_ref, mode='s&p')
img_noise = img_ref + 0.1 * np.random.normal(size=img_ref.shape)
plt.imshow(img_noise,cmap='gray')
plt.show()

def denoise_image(img,lbda,max_iteration):

    m,n = img.shape
    w0 = np.zeros( (2,m,n), dtype=np.float64 )
    z0 = np.zeros( (2,m,n), dtype=np.float64 )
    t0 = 1.0
    A = 1.0/(8*lbda)
    count=0
    lastcost=float('inf')
    costhist_fista=[]
    while count<max_iteration:
        count+=1
        L = compute_linear_operator_L( z0 ) 
        LT = compute_linear_operator_LT( operatorC( img - lbda*L ) )
        w1 = operatorP( z0 + A*LT ) 
        t1 = (1 + np.sqrt(1+4*t0**2) )/2.0
        z1 = w1 + (t0-1)/(t1)*(w1-w0)
        t0 = t1
        w0 = w1
        z0 = z1
        dimg = operatorC( img - lbda*compute_linear_operator_L( w0 ))
        cost=tvcost_fista(img,dimg,lbda)
        costhist_fista.append(cost)
        print 'current cost is %.2f  ' %cost,
        if abs(lastcost-cost)<0.5:
            break
        else:
            lastcost=cost
        count+=1
    print
    return dimg,count,costhist_fista

start=time.clock()
dimg,count,costhist_fista=denoise_image(img_noise,0.12,101)
end=time.clock()
print end-start,"seconds process time"
print 'Iteration tims is %d' %count
plt.figure(0)
plt.imshow(dimg,cmap='gray')
plt.show()

mse=np.sum((dimg-img_ref)**2)/float(dimg.shape[0]*dimg.shape[1])
maxi=dimg.max()
psnr=20*log(maxi/sqrt(mse))
print 'The Peak Signal-to-Noise Ratiod is %f'%psnr


plt.figure(1)
ite=[i for i in range(1,len(costhist_fista)+1)]
plt.plot(ite,costhist_fista)
plt.xlabel('iteration times')
plt.ylabel('cost')
plt.ylim(0,10000)
plt.show()