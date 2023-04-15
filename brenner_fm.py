# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 17:46:07 2019

@author: raphael abele
"""


import numpy as np
import timeit
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count


def getBrenner(img):
    I1 = img.copy()
    I1[1:-1,:] = img[2:,:]
    I1[:,1:-1] = img[:,2:]
    
    I2 = img.copy()
    
    out = np.sum((I1 - I2)**2)
    
    return out  



def computeBrennerBasedIQA(image_stack_v):
    
    start_timer = timeit.default_timer()
    
    stat_values = []
    
    for img in image_stack_v:
        if 'int' not in str(img.dtype):
            img = (img*255).astype('uint8')
    
    """ Threaded """
    pool = ThreadPool(processes = cpu_count()*2)
    threads = []
    for image in image_stack_v:
        threads.append( pool.apply_async(getBrenner, (image, )) )
    for i_t in range(len(threads)):
        stat = threads[i_t].get()
        stat_values.append(stat)
    """ ############ """
        
    print("Runtime Brenner for "+ str(len(image_stack_v)) +" images:", timeit.default_timer()-start_timer)

    return stat_values

