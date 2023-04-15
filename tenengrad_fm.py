"""
Created on Fri Sep 11 23:30:00 2020

@author: Raph
"""


import timeit
import numpy as np
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
from skimage import filters

def Tenengrad(img):
    tmp = filters.sobel(img)
    out = np.sum(tmp**2)
    out = np.sqrt(out)
    return out



def mainTenengrad(image_stack_v):
    
    start_timer = timeit.default_timer()
    
    stat_values = []
    
    
    for img in image_stack_v:
        if 'int' not in str(img.dtype):
            img = (img*255).astype('uint8')
            
        stat_values.append(Tenengrad(img))
    
    # """ Threaded """
    # pool = ThreadPool(processes = cpu_count()*2)
    # threads = []
    # for image in image_stack_v:
    #     threads.append( pool.apply_async(Tenengrad, (image, )) )
    # for i_t in range(len(threads)):
    #     stat = threads[i_t].get()
    #     stat_values.append(stat)
    # """ ############ """
    
    print("Runtime Tenengrad for "+ str(len(image_stack_v)) +" images:", timeit.default_timer()-start_timer)

    return stat_values

