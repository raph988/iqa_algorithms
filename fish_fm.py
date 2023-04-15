# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 11:40:07 2016

@author: raph988

In this script is developped the Fast Image Sharpness (FISH) algorithm from paper
Phong V. Vu and Damon M. Chandler :
A Fast Wavelet-Based Algorithm for Global and Local Image Sharpness Estimation
2012
"""

import timeit
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
import pywt
import numpy as np
import cv2
import math

from imagesToolbox import scaleData, getWavelet
from waveletBasedIQA import threshWaveletCoeffs, DWT_coeffs_helper

    

def getFISH(image = None, wavelet = None, coeffs = None, thresh = False, thresh_type = "soft", w_lvl = 3):
    """
    Compute the Fast Image Sharpness algorithm from paper
    Phong V. Vu and Damon M. Chandler :
    A Fast Wavelet-Based Algorithm for Global and Local Image Sharpness Estimation
    2012
    
    If image is given, coeffs are computes with the given wavelet.
    If coeffs are given, image and wavelet are not necessary.
    
    Usage
    -----
    getFISH([image = None,] [wavelet = None,] [coeffs = None,] [thresh_value = None,] [w_lvl = 3])

    Parameter
    ---------
    image : ndarray         Image of interest
    wavelet : object        A wavelet object
    coeffs : ndarray        Coefficients of the wavelet decomposition (to be computed if None)
    thresh_value : float    If not None, threshold is applied on the coeffs before computing FISH
    w_lvl : int             The level of wevelet decomposition. 3 levels are used for the next step FISHbb.

    Return
    ------
    FISH (Fast Image SHarpness) : float
    """
    
    if coeffs is None:
        
        if image is None and wavelet is None:
            print("-E- Cannot compute FISH without image and the wavelet to use")
            return
        
        max_lvl = pywt.dwt_max_level(len(image.flatten()), wavelet.dec_len)
        if w_lvl is None or w_lvl > max_lvl:
            w_lvl = max_lvl
            print("Wavelet decomposition level given is too hight. Reaffected to ",w_lvl)
        
#            image = cv2.normalize(image  , None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        if 'int' in str(image.dtype) and np.min(image) > 1:
#                image = scaleData(image, 0, 1, 'float32')
            image = image.astype(np.float32)/255
        coeffs = pywt.wavedec2(image, wavelet, level = w_lvl)

            
    if thresh is True:
        coeffs = threshWaveletCoeffs(coeffs, thresh_type=thresh_type)
        
        
    coeffs_helper = DWT_coeffs_helper(coeffs)

    # Energie of each subbands xy in each decomposition lvl n
    Exyn = []
    for lvl in range(0, w_lvl):
        c = coeffs_helper.getCoeffs(lvl+1)[:3]
        c = list((subb*subb).flatten() for subb in c)
        Exyn.append( list(np.mean(subb) for subb in c) )


    # Energie of each decomposition lvl
    if len(Exyn[0]) == 3:
        alpha = 0.8
        En = list( (1 - alpha) * ((e[0] + e[1])/2) + alpha*e[2] for e in Exyn )
    elif len(Exyn[0]) == 2:
        En = list( abs( (e[0] + e[1])/2) for e in Exyn )
    
    # Index Sharpness (FISH)
    FISH = 0.0
    for i in range(0, len(En), 1):
        FISH += pow( 2, w_lvl-i ) * En[i]

    
    return FISH

    
def getFISHbb(image, wavelet, wavelet_lvl = 3, blockSize = 16, thresh = False, thresh_type = "soft", show_map = False):
    """
    Compute the Fast Image Sharpness Block-Based algorithm from paper
    Phong V. Vu and Damon M. Chandler :
    A Fast Wavelet-Based Algorithm for Global and Local Image Sharpness Estimation
    2012
    
    
    Usage
    -----
    getFISHbb(image, wavelet[, wavelet_lvl = 3][, blockSize = 0.5][, compute_thresh = False][, show_map = False])

    Parameter
    ---------
    image : ndarray         Image of interest
    wavelet : object        A wavelet object
    wavelet_lvl : int       The level of wevelet decomposition. 3 levels are used for the next step FISHbb.
    blockSize : float       Size of blocks (a ration of main image) from which FISHbb will be computed.
    thresh_value : float    If not None, threshold is applied on the coeffs before computing FISH
    show_map : bool         If True, create and show the sharpness map
    
    Return
    ------
    FISHbb (Fast Image SHarpness Block-Based) : float
    """
    if isinstance(blockSize, float):
        raise TypeError("Blocksize should an integer.")

    coeffs = pywt.wavedec2(image, wavelet, level = wavelet_lvl)  
    
    # only subbands coeffs of levels 1 to 3 are take in account
    if len(coeffs) > 3:
        coeffs = coeffs[len(coeffs)-3:]
        
    if thresh is True:
        coeffs = threshWaveletCoeffs(coeffs, thresh_type=thresh_type)
    
    # definition of block sizes for each dwt level
    # and the translation stepping
    h, w = image.shape
    
    min_side = min(h,w)
    max_side = max(h,w)
    if max_side < 2*blockSize:
        c = (int(h/2), int(w/2))
        start_x, end_x = c[1]-int(blockSize/2), c[1]+int(blockSize/2)
        start_y, end_y = c[0]-int(blockSize/2), c[0]+int(blockSize/2)
        image = image[start_x : end_x, start_y : end_y]
      
    try:
        print(
        """FISHbb parameters : 
                - block-size : {0}
                - threshold  : {1}
                - ROI coords : {2}
                - ROI size   : {3}""".format(blockSize, thresh,((start_x, end_x),(start_y, end_y)) , image.shape))
    except:
        print(
        """FISHbb parameters : 
                - block-size : {0}
                - threshold  : {1}""".format(blockSize, thresh))
        
    bs = [blockSize]*2
    
    bsize1 = ( int(bs[0]/2), int(bs[1]/2) )
    step_1 = ( int(bsize1[0]/2), int(bsize1[1]/2) )
    
    bsize2 = ( int(bsize1[0]/2), int(bsize1[1]/2) )
    step_2 = ( int(bsize2[0]/2), int(bsize2[1]/2) )
    
    bsize3 = ( int(bsize2[0]/2), int(bsize2[1]/2) )
    step_3 = ( int(bsize3[0]/2), int(bsize3[1]/2) )
    
    # check is block size is not too small
    # the condition is step_1 < (4,4) (step_1 is used for next steps)
    """ Unused since block-size is auto-assigned according to global image size.
    if step_1[0] < 4:
        try:
            x = Symbol('x')
            sol = solve((min(h,w)*x/4) < 4, dict=True)
            sol = str(sol)
            sol= re.findall(r'&(.*)\)',sol)[0]
            sol= re.findall(r'<(.*)',sol)[0].replace(' ', '')
            
            string = "Block-size ratio given : "+str(blockSize)+"( ->"+str(bs)+" ). Min is "+sol+"."
        except:
            string = ""
        raise ValueError("Blocksize to small.\n"+string)
    """
    
        
    FISH_list = []
    subb_blocks = []
    s_map = []
    j = 0
    while j*step_1[1] + bsize1[1] <= np.array(coeffs[2][0]).shape[0]:
        line = []
        i = 0
        while i*step_1[0] + bsize1[0] <= np.array(coeffs[2][0]).shape[1]:
            lvl1 = []
            t = 1
            for band in coeffs[2]:
                tmp = band[j*step_1[t]:j*step_1[t] + bsize1[t], i*step_1[not t]:i*step_1[not t] + bsize1[not t]]
                lvl1.append( tmp.copy() )
            
            lvl2 = []
            for band in coeffs[1]:
                tmp = band[j*step_2[t]:j*step_2[t] + bsize2[t], i*step_2[not t]:i*step_2[not t] + bsize2[not t]]
                lvl2.append( tmp.copy() )
                    
            lvl3 = []
            for band in coeffs[0]:
                tmp = band[j*step_3[t]:j*step_3[t] + bsize3[t], i*step_3[not t]:i*step_3[not t] + bsize3[not t]]
                lvl3.append( tmp.copy() )
                
            s_levels = [[], lvl3, lvl2, lvl1]
            v = getFISH(coeffs=s_levels, thresh = thresh, w_lvl = wavelet_lvl, thresh_type=thresh_type)
            FISH_list.append(v)
            
            b = subb_introspec(s_levels[1:], bs)
            line.append(b)
            i += 1
            
        subb_blocks.append(line)
        j+=1
    
    if len(FISH_list) > 1:
        FISH_list = scaleData(FISH_list, 0, 1, 'float32')
        
    s_map = None
    if show_map is True:
        
        tmp = np.reshape(np.array(FISH_list), np.array(subb_blocks).shape[:2])
        print(tmp.shape)
        s_map = []
        for y in range(0, len(tmp), 1):
            c_line = tmp[y]
            tmp_line = c_line[0]
            for x in range(1, len(c_line), 1):
                pix = c_line[x]
                tmp_line = np.hstack( (tmp_line, pix) )
            if len(s_map) < 1 :
                s_map = tmp_line
            else: 
                s_map = np.vstack( (s_map, tmp_line) )
        s_map = cv2.resize(s_map, (int(image.shape[1]/2), int(image.shape[0]/2)), interpolation = cv2.INTER_NEAREST) #CUBIC )
    
    # according to Phong et al., the 10% sharpest pixels are significant enough
    FISH_bb = 0.0
    FISH_list = sorted(FISH_list, reverse = True)
    percent10 = round(len(FISH_list)*0.1)
    if percent10 <= 0: percent10 = 1
    
    for f in FISH_list[:percent10]:
        FISH_bb += pow(f, 2)
    FISH_bb = math.sqrt( FISH_bb/len(FISH_list) )
    
    # the algo (from document) should return 
#        value = FISH_bb
    # but to take in account each pixels, I return the mean
    value = float(sum(FISH_list)/len(FISH_list))
    
    return value, s_map



def subb_introspec(levels, block_size, current_index = 0):
    """
    Recursive method.
    Construct a block composed with little blocks from differents subbands from wavelet decomp levels.
    See Figure 1 in paper :
    Phong V. Vu and Damon M. Chandler :²
    A Fast Wavelet-Based Algorithm for Global and Local Image Sharpness Estimation
    2012
    
    Usage
    -----
    subb_introspec(levels, block_size, [current_index = 0])

    Parameter
    ---------
    levels : ndarray        Coefficients ordonned by wavelet decomp levels, divided by subbands
    block_size : int        Size of the block (square) to construct
    current_index : int     The current index used for recursivity
    
    Return
    ------
    Constructed block : ndarray
    levels[0] = level 0
    levels[0][0] = level 0, subband LL
    """
    
    iLL, iLH, iHL, iHH = 0, 1, 2, 3
    
    block = np.zeros( block_size )
    if current_index == len(levels):
        return block

    sub_size = ( int(block_size[0]/2), int(block_size[1]/2) )
    current_lvl = levels[len(levels)-1 - current_index]

    HH = current_lvl[iHH-1]
    block[sub_size[0] : sub_size[0]+HH.shape[0], sub_size[1] : sub_size[1]+HH.shape[1]] = HH

    LH = current_lvl[iLH-1]
    block[sub_size[0]-HH.shape[0] : sub_size[0], sub_size[1] : sub_size[1]+HH.shape[1]] = LH

    HL = current_lvl[iHL-1]
    block[sub_size[0] : sub_size[0]+HH.shape[0], sub_size[1]-HH.shape[1] : sub_size[1]] = HL

    block[:sub_size[0], :sub_size[1]] = subb_introspec(levels, sub_size, current_index+1)
    return block




def computeFISHBasedIQA(image_stack_v):
    
    w_cdf = getWavelet("cdf_97")
    stat_values = []
    
    start_timer = timeit.default_timer()
    
    """ Threaded """
    pool = ThreadPool(processes = cpu_count())
    threads = []
    for image in image_stack_v:
        threads.append( pool.apply_async(getFISH, (image, w_cdf,)) )
    for i_t in range(len(threads)):
        stat = threads[i_t].get()
        stat_values.append(stat)
    """ ############ """
        


#    fish_result = []
#    start_timer = timeit.default_timer()
#    for img in noised_images:
#        fish = imp.getFISH(img, w_cdf, thresh=False, w_lvl=3)
#        stat_values.append(fish)

    print("Runtime FISH for "+str(len(image_stack_v))+"im : ", timeit.default_timer()-start_timer)
        
    return stat_values



def main():
    image = np.random.random((500,500))
    image_iqa = getFISH(image, getWavelet("cdf_97"))
    print("A random image IQA based on FISH algo: ", image_iqa)
    # for an autofocus algo, this IQA is normalized

    
        
if __name__ == "__main__":
    main()
    

