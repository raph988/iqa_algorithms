# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 17:55:04 2016

@author: raph988

In this script an IQA based on wavelet packets decomposition.
"""

import numpy as np
import cv2
import pywt
import timeit
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count


wave = 'coif1'
wp = pywt.WaveletPacket2D(data=None, wavelet=wave, mode='symmetric', maxlevel=2)
def packetIQA(image):
    start_timer = timeit.default_timer()
    
    Kh = np.zeros((3,3), dtype=np.int8)
    Kv = Kh.copy()
    K45 = Kh.copy()
    K135 = Kh.copy()
    
    Kh[0, 0] = Kh[2, 0] = 1 
    Kh[0, 2] = Kh[2, 2] = -1
    
    Kv[0, 0] = Kv[0, 2] = -1 
    Kv[2, 0] = Kv[2, 2] = 1
    
    K45[0, 2] = -1
    K45[2, 0] = 1 
    
    K135[0, 0] = 1 
    K135[2, 2] = -1
    
    Gh = cv2.filter2D(image, 0, kernel=Kh)
    Gv = cv2.filter2D(image, 0, kernel=Kv)
    G45 = cv2.filter2D(image, 0, kernel=K45)
    G135 = cv2.filter2D(image, 0, kernel=K135)
    
    Nh, Nv, N45, N135 = np.sum(Gh.flatten()),  np.sum(Gv.flatten()), np.sum(G45.flatten()), np.sum(G135.flatten())
    Nsum = Nh+Nv+N45+N135
    pH = Nh / Nsum
    pV = Nv / Nsum
    pD = (N45+N135) / Nsum
    
    # https://pywavelets.readthedocs.io/en/latest/regression/wp2d.html
    # wave = imp.getWavelet("COW")
    # wave = 'coif1'
    # wp = pywt.WaveletPacket2D(data=None, wavelet=wave, mode='symmetric', maxlevel=2)
    wp.data = image
    wp.decompose()
    
    """ aa ah av ad
        ha hh hv hd
        va vh vv vd
        da dh dv dd """
#        ah, av, ad = (0,1), (0,2), (0,3)
#        ha, hh, hv, hd = (1,0), (1,1), (1,2), (1,2)
#        va, vh, vv, vd = (2,0), (2,1), (2,2), (2,3)
#        da, dh, dv, dd = (3,0), (3,1), (3,2), (3,3)
    M1 = "hh", "ah", "ha", "hv", "vh", "hd", "dh"
    M2 = "vv", "av", "va", "hv", "vh", "dv", "vd"
    M3 = "dd", "ad", "da", "hd", "dh", "dv", "vd"
    
    Dh = np.sum( list( (wp[name].data).flatten() for name in M1)) 
    Dv = np.sum( list( (wp[name].data).flatten() for name in M2)) 
    Dd = np.sum( list( (wp[name].data).flatten() for name in M3)) 


    fm = pH*Dh + pV*Dv + pD*Dd
    return fm
    


def computeWavePacketBasedIQA(image_stack_v):
    
    global wp
    if wp is None:
        wp = pywt.WaveletPacket2D(data=None, wavelet=wave, mode='symmetric', maxlevel=2)
        
    stat_values = []
    
    start_timer = timeit.default_timer()
    
    """ Threaded """
    pool = ThreadPool(processes = cpu_count()*2)
    threads = []
    for image in image_stack_v:
        threads.append( pool.apply_async(packetIQA, (image, )) )
    for i_t in range(len(threads)):
        stat = threads[i_t].get()
        stat_values.append(stat)
    """ ############ """
    
#    for i, image in enumerate(image_stack_v):
#        fm = packetIQA(image)
#        stat_values.append(fm)
        
    print("Runtime Packet (",len(image_stack_v), "images): ", timeit.default_timer()-start_timer)
    del wp
    wp = None
    return stat_values



