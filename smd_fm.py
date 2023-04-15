# -*- coding: utf-8 -*-
"""
@author: Raph988
"""

import math
import numpy as np
import timeit
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count

# def SMD2(img):
    
#     shape = np.shape(img)
    
#     # out = sum( list( math.fabs(int(img[x,y])-int(img[x+1,y]))*math.fabs(int(img[x,y]-int(img[x,y+1]))) for y in range(0, shape[1]-1) for x in range(0, shape[0]-1) ) )
#     out = 0
#     for y in range(0, shape[1]-1):
#         for x in range(0, shape[0]-1):
#             out+=math.fabs(int(img[x,y])-int(img[x+1,y]))*math.fabs(int(img[x,y]-int(img[x,y+1])))
    
#     return out

def SMD2(img): #'SFRQ' % Spatial frequency (Eskicioglu95)
    Ix = img.copy()
    Iy = img.copy()
    Ix[:,:-1] = np.diff(img, axis=1)**2
    Iy[:-1,:] = np.diff(img, axis=0)**2
    out = np.mean(math.sqrt((Iy+Ix).astype('float32')))
    return out


def ComputeSMDBasedIQA(image_stack_v):
    
    start_timer = timeit.default_timer()
    
    stat_values = []
    
    for img in image_stack_v:
        if 'int' not in str(img.dtype):
            img = (img*255).astype('uint8')
    
        stat_values.append(SMD2(img))
        
        
    # """ Threaded """
    # pool = ThreadPool(processes = cpu_count()*2)
    # threads = []
    # for image in image_stack_v:
    #     threads.append( pool.apply_async(SMD2, (image, )) )
    # for i_t in range(len(threads)):
    #     stat = threads[i_t].get()
    #     stat_values.append(stat)
    # """ ############ """
        
    print("Runtime SMD2 for "+ str(len(image_stack_v)) +" images:", timeit.default_timer()-start_timer)

    return stat_values
