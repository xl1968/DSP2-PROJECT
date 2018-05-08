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
plt.imshow(img_ref,cmap='gray')



img_ref = mat_lena[140:,120:][:256,:256] / 255.0
#img_noise = skimage.util.random_noise(img_ref, mode='gaussian')
#img_noise2 = skimage.util.random_noise(img_ref, mode='s&p')
img_noise = img_ref + 0.1 * np.random.normal(size=img_ref.shape)
plt.figure(0)
plt.imshow(img_noise,cmap='gray')
plt.show()


start=time.clock()
a=restoration.denoise_tv_chambolle(img_noise, weight=0.1)
end=time.clock()
plt.figure(1)
plt.imshow(a,cmap='gray')
plt.show()
print end-start,"seconds process time"
cost=tvcost_rof_gradient(a,img_noise,0.1)
print 'The cost is %.2f' %cost
mse=np.sum((a-img_ref)**2)/float(a.shape[0]*a.shape[1])
maxi=a.max()
psnr=20*log(maxi/sqrt(mse))
print 'The Peak Signal-to-Noise Ratiod is %f'%psnr