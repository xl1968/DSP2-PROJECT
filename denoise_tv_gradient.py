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
from cost_function import tvcost_rof_gradient

lena = Image.open('lena512.bmp')
mat_lena=np.asarray(lena)
img_ref = mat_lena[140:,120:][:256,:256] / 255.0
plt.figure(0)
plt.imshow(img_ref,cmap='gray')
plt.show()



img_ref = mat_lena[140:,120:][:256,:256] / 255.0
#img_noise = skimage.util.random_noise(img_ref, mode='gaussian')
#img_noise2 = skimage.util.random_noise(img_ref, mode='s&p')
img_noise = img_ref + 0.1 * np.random.normal(size=img_ref.shape)
plt.imshow(img_noise,cmap='gray')
plt.show()



def rof_gradient(img,U_init,tau=0.2,weight=1,max_iteration=101):
    m,n=img.shape
    U=U_init
    px=py=img
    count=0
    lastcost=float('inf')
    costhist_rof=[]
    while count<max_iteration:
        Uold=U
        gradUx=np.roll(U,-1,axis=1)-U
        gradUy=np.roll(U,1,axis=0)-U
        normnew=np.maximum(1,np.sqrt(gradUx**2+gradUy**2))
        px=gradUx/normnew
        py=gradUy/normnew
        rxpx=np.roll(px,-1,axis=1)
        rypy=np.roll(py,-1,axis=0)
        divp=(rxpx-px)+(rypy-py)
        step=tau*(np.exp(-count**2/(max_iteration))+0.01)
        U=Uold-step*((Uold-img)+weight*divp)
        cost=tvcost_rof_gradient(img,U,weight)
        costhist_rof.append(cost)
        print 'current cost is %.2f  ' %cost,
        if abs(lastcost-cost)<0.5:
            break
        else:
            lastcost=cost
        count+=1
    print
    return U,count,costhist_rof     
start=time.clock()
img_denoise_by_rof,count,costhist_rof=rof_gradient(img_noise,img_noise,tau=0.2,weight=1.0,max_iteration=201)
end=time.clock()
print end-start,"seconds process time"
print 'Iteration tims is %d' %count
plt.imshow(img_denoise_by_rof,cmap='gray')
plt.show()

mse=np.sum((img_denoise_by_rof-img_ref)**2)/float(img_denoise_by_rof.shape[0]*img_denoise_by_rof.shape[1])
maxi=img_denoise_by_rof.max()
psnr=20*log(maxi/sqrt(mse))
print 'The Peak Signal-to-Noise Ratiod is %f'%psnr 
plt.figure(3)
ite=[i for i in range(1,len(costhist_rof)+1)]
plt.plot(ite,costhist_rof)
plt.xlabel('iteration times')
plt.ylabel('cost')
plt.ylim(0,10000)
plt.show()