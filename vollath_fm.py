# -*- coding: utf-8 -*-
"""
@author: Raph988
"""


import math
import numpy as np
import timeit
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count


def Vollath(img):
    # shape = np.shape(img)
    # u = np.mean(img)
    # out = -shape[0]*shape[1]*(u**2)
    
    # # out += sum( list( int(img[x,y])*int(img[x+1,y]) for y in range(0, shape[1]) for x in range(0, shape[0]-1)) )
    # for y in range(0, shape[1]):
    #     for x in range(0, shape[0]-1):
    #         out+=int(img[x,y])*int(img[x+1,y])
    
    image = img.copy()
    image = image.astype('float32')
    
    I1 = image.copy()
    I1[1:-1,:] = image[2:,:]
    
    I2 = image.copy()
    I2[1:-2,:] = image[3:,:]
    
    image = image*(I1-I2)
    
    out = np.mean(image)
    
    return out



def computeVollathBasedIQA(image_stack_v):
    
    start_timer = timeit.default_timer()
    
    stat_values = []
    

    for img in image_stack_v:
        if 'int' not in str(img.dtype):
            img = (img*255).astype('uint8')
    
    """ Threaded """
    pool = ThreadPool(processes = cpu_count()*2)
    threads = []
    for image in image_stack_v:
        threads.append( pool.apply_async(Vollath, (image, )) )
    for i_t in range(len(threads)):
        stat = threads[i_t].get()
        stat_values.append(stat)
    """ ############ """
        
    print("Runtime Vollath for "+ str(len(image_stack_v)) +" images:", timeit.default_timer()-start_timer)

    return stat_values

