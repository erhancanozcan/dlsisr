#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 13:14:26 2021

@author: can
"""
import matplotlib.pyplot as plt
def draw_triplet(predicted, real_high_quality, interpolated,from_training=True):
    
    plot,axar=plt.subplots(1,3)
    axar[0].imshow(predicted,cmap="gray")
    axar[0].set_title("Generated")
    axar[0].set_xticklabels([])
    axar[0].set_yticklabels([])
    
    axar[1].imshow(real_high_quality,cmap="gray")
    axar[1].set_title(" Real High Quality")
    axar[1].set_xticklabels([])
    axar[1].set_yticklabels([])
    
    axar[2].imshow(interpolated,cmap="gray")
    axar[2].set_title(" Low Quality")
    axar[2].set_xticklabels([])
    axar[2].set_yticklabels([])
    plot.suptitle("Triplet")
    #if from_training==1:
    #    plot.suptitle("Triplet from Training set") 
    #else:
    #    plot.suptitle("Triplet from Test set") 
        
    plot
    



def draw_all(real_high_quality, interpolated, sparse, svr,rf):
    
    plot,axar=plt.subplots(1,5)
    
    axar[0].imshow(real_high_quality,cmap="gray")
    axar[0].set_title(" Real")
    axar[0].set_xticklabels([])
    axar[0].set_yticklabels([])
    
    
    axar[1].imshow(interpolated,cmap="gray")
    axar[1].set_title("Interpolated")
    axar[1].set_xticklabels([])
    axar[1].set_yticklabels([])
    
    axar[2].imshow(sparse,cmap="gray")
    axar[2].set_title(" Sparse")
    axar[2].set_xticklabels([])
    axar[2].set_yticklabels([])
    
    axar[3].imshow(svr,cmap="gray")
    axar[3].set_title(" SVR")
    axar[3].set_xticklabels([])
    axar[3].set_yticklabels([])
    
    axar[4].imshow(rf,cmap="gray")
    axar[4].set_title(" RF")
    axar[4].set_xticklabels([])
    axar[4].set_yticklabels([])
    #if from_training==1:
     #   plot.suptitle("Triplet from Training set") 
    #else:
     #   plot.suptitle("Triplet from Test set") 
        
    plot
    
    
    
    