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
from proximal_operator import operatorF
from linear_operator import forward_gradient,backward_divergence
from cost_function import tvcost_rof_primal_dual


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

def rof_by_pd(img,L,tau,theta,max_iteration,lambda_rof=10.0):
    x = img
    h,w = img.shape
    sigma = 1.0 / (L * tau)
    y = forward_gradient(x)
    count=0
    lastcost=float('inf')
    costhist_pd=[]
    while count<max_iteration:
        # Dual update
        y = y + sigma * forward_gradient(x)
        y = operatorF(y, 1.0)
        # Primal update
        x_old = x
        xnew = (x - tau * backward_divergence(y) + lambda_rof * tau * img) / (1.0 + lambda_rof * tau)
        # update
        x = xnew + theta * (xnew - x_old)
        cost=tvcost_rof_primal_dual(img,x,lambda_rof)
        print 'current cost is %.2f  ' %cost,
        costhist_pd.append(cost)
        if abs(lastcost-cost)<0.5:
            break
        else:
            lastcost=cost
        count+=1
    print
    return x,count,costhist_pd  
start=time.clock()
img_denoised,count,costhist_pd=rof_by_pd(img_noise,L=8.0,tau=0.02,theta=0.1,max_iteration=101,lambda_rof=16.0)
end=time.clock()
print end-start,"seconds process time"
print 'Iteration tims is %d' %count
plt.figure(1)
plt.imshow(img_denoised,cmap='gray')
plt.show()
mse=np.sum((img_denoised-img_ref)**2)/float(img_denoised.shape[0]*img_denoised.shape[1])
maxi=img_denoised.max()
psnr=20*log(maxi/sqrt(mse))
print 'The Peak Signal-to-Noise Ratiod is %f'%psnr 

ite=[i for i in range(1,len(costhist_pd)+1)]
plt.plot(ite,costhist_pd)
plt.xlabel('iteration times')
plt.ylabel('cost')
plt.ylim(0,10000)
plt.show()