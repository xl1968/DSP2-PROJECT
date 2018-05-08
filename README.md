# DSP2-Final Project:Different Optimization Method to Implement Total Variation Denoising
This project is aim to use different optimization method to do image denoising based on total variation denoising.
* To run these codes, install numpy,matplotlib,scipy,PIL,skimage packages.
* In this project, the original image is Lena512 which is often used in image processing. Add normal distributed noise to the image and do 
  TVD denoising.
# Packages used in project 
* The linear_operator.py and proximal_operator.py defines some function of processing data. Cost_function.py defines the function to         calculate the cost of the model we choose in iamge processing.

* denoise_tv_chambolle.py uses the chambolle method to do denoising.

* denoise_tv_gradient.py is based on the gradient descent model, and I design this method only based on mathematical way and assume the     image has constinuous value, this is an approximate method to do denoising.

* denoise_tv_rof_primal_dual.py is based on primal-dual algorithm which is widely used in optimization. This method is based on rof model   and it shows great ability to denoise normal distributed noise in image processing.

* denoise_tv_fista.py is also based on primal_dual algorithrm and it use different optimization method to reach the optimal denoisd         result. Usually it has fewer iterations to get covergence.



# Dependencies:
1. scipy, numpy, matplotlib



# References
1.Fast Gradient-Based Algorithms for Constrained Total Variation Image Denoising and Deblurring Problems.  Amir Beck and Marc Teboulle
2.A Tutorial on Primal-Dual Algorithm.                                                                     Shenlong Wang
3.ROF and TV-L1 denoising with Primal-Dual algorithm                                                       Alexander Mordvintsev
4.An introduction to Total Variation for Image Analysis                                                    Antonin Chambolle, Vicent                                                                                                                  Caselles, Matteo Novaga, Daniel                                                                                                            Cremers, Thomas Pock
