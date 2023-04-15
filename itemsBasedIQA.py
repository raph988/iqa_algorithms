# -*- coding: utf-8 -*-
"""
@author: Raph


In this script is developped a tool for IQA based on mutliple items detection : lines, blobs and a Fourier spectrum based IQA. 
"""

from PySide2 import QtCore, QtUiTools, QtWidgets, QtGui

import os
import numpy as np
from matplotlib import pyplot as plt
import cv2

from imagesToolbox import save_pyplot, luminanceCorrection, scaleData, isVertical, isHorizontal, getWavelet, histStretch
import pywt
from renderer import imshow




def saveStats(x_values, y_values, fig_path, fig_name, highlight_value = None, title = "", x_label="", y_label="", verbose = False):
    """
    Save an image of a graphic generated from the given as x and y values.

    Usage
    -----
    saveStats(x_values, y_values, [highlight_value = None, title = "", x_label="", y_label=""]))
                
    Parameters
    ----------
    x_values : list
        List of absisse values.
    y_values : list
        List of ordinate values.
    fig_path : str
        Where to save the graphic.
    fig_name : str
        Name of the saved file.
    highlight_value : int, optional
        If not None, draw a vertical line a the highlight_value index. The default is None.
    title : str, optional
        Graphic title. The default is "".
    x_label : str, optional
        X axis label. The default is "".
    y_label : str, optional
        Y axis label. The default is "".
    verbose : bool, optional
        Verbose mode. The default is False.

    Returns
    -------
    None.
    """

    if len(x_values) != len(y_values):
        try:
            stepping = x_values[0]-x_values[1]
            x_values = np.arange(0, len(y_values), stepping)+x_values[0]
        except:
            e = Exception("Unable to draw focus statistics...")
            print(e)
            return


    fig_stats, ax = plt.subplots(1, 1)
    ax.plot(x_values, y_values, 'r', label=title)
    if highlight_value is not None:
        ax.axvline(x=highlight_value, ls='--', lw=1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.grid(True)
    ax.legend(bbox_to_anchor=(1., 1.), loc=3, ncol=1, borderaxespad=1.)

    fig_stats.savefig(fig_path+fig_name, dpi=300)
    if verbose:
        print("Stats saved as", fig_path+fig_name)
    plt.close(fig_stats)



def savePyplot(p, save_image, verbose=False):
    """
    
    Compute a file name and the file path to save the plot p (see 'save_pyplot' function).

    Parameters
    ----------
    p : PyPlot
        The graphic to save.
    save_image : bool
        If True, also save a picture version of the plot.
    verbose : bool, optional
        Verbose mode. The default is False.

    Returns
    -------
    None.
    """

    this_script_path = os.path.dirname(os.path.realpath(__file__))
    fig_num = 0
    fig_name = '/stats'+'_'+fig_num+'.png'
    while os.path.isfile(this_script_path+fig_name):
        fig_num += 1
        fig_name = '/stats'+'_'+fig_num+'.png'

    ret = save_pyplot(this_script_path+fig_name, p, save_image = save_image)

    if verbose:
        if ret==0:
            print("Stats saved in", this_script_path, "as", fig_name)
        else:
            print("Fail to save stats...")




def fourierIQA(image, drawFourier = False):
        """
        Measure sharpness in the picture, performing sum of high frequencies in the Discrete Fourier Transform.

        Usage
        -----
        fourierIQA(image [, drawFourier = False])

        Parameters
        ---------
        image : ndarray         
            Image to correct
        drawFourier : bool      
            If True, draw a 2D representation of magnitude Fourier transform

        Returns
        -------
        float
            Blur quantification
        """

        image_norm = image.astype(np.float32)/255.0
        
        dft_R = cv2.dft(image_norm) #, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

        if drawFourier is True:
            dft = cv2.dft(image, flags = cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            magnitude_spectrum = np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
            imshow("magnitude_spectrum", magnitude_spectrum*0.1)


        median = np.median(np.sort(dft_R))
        upper = dft_R[ dft_R >= median ]
        try:
            s = sum(upper)/len(upper)
        except:
            s = sum(upper)
        return s
    
    
    
def blobsDetector(image, show = False):
    """
    Detect blobs due to dust on the microchip

    Usage
    -----
    blobsDetector(image [, show = True])

    Paramters
    ---------
    image : ndarray             
        Image

    Returns
    -------
    Tuple
        Center(s) of blob(s)
    """


    mask = np.ones(image.shape, dtype = 'uint8')*255
    mask[image <= np.min(image)+30] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))

    detector = cv2.Simpldetector = cv2.SimpleBlobDetector_create()
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 0

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.5

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.01

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(mask)
    if show is True:
        input_im = image.copy()
        input_im = scaleData(input_im,0, 255, 'uint8')
#            try:
#                input_im = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#            except: pass
        im_with_keypoints = cv2.drawKeypoints(input_im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#            cv2.imshow("blobs", im_with_keypoints), cv2.imshow("mask", mask), cv2.waitKey(0), cv2.destroyAllWindows()
        imshow("blobs", im_with_keypoints)

    mask_inv = mask.copy()
    mask_inv[mask == 0] = 255
    mask_inv[mask != 0] = 0
    retVal, labels = cv2.connectedComponents(mask_inv, None)

    return keypoints



def FastSegmentsDetector(image, limit_to_hv= False, delta=4, do_merge=False, draw=False):
    """
    LEE, Jin Han, LEE, Sehyung, ZHANG, Guoxuan, et al. 
    Outdoor place recognition in urban environments using straight lines. 
    In : 2014 IEEE International Conference on Robotics and Automation (ICRA). 
    IEEE, 2014. p. 5550-5557.

    Parameters
    ----------
    image : ndarray
        The image to analyze.
    limit_to_hv : bool, optional
        If True, the not horizontal neither vertical line segments are popped out. The default is False.
    delta : int, optional
        The delta to sort a line segment as hor or vert. The default is 4.
    do_merge : bool, optional
        If True, segments detect on both sides of a line item are merge (prevent double segment detection). The default is False.
    draw : bool, optional
        If True, detection is displayed. The default is False.

    Returns
    -------
    if limit_to_h is True, returns
        list
            List of h segments.
        list
            List of v segments.
    Otherwise, returns
        list
            Segments list.
        

    """
    border = round(0.8 * max(image.shape))
    tmp = np.zeros( (image.shape[0]+border*2, image.shape[1]+border*2), dtype=image.dtype )
    tmp[border:-border, border:-border] = image
    image = tmp
    # Create default Fast Line Detector class
    fld = cv2.ximgproc.createFastLineDetector(_do_merge=do_merge)
    # Get line vectors from the image
    try:
        res = fld.detect(image)
    except Exception as e:
        print(e)
        res = None
    if res is None:
        return [], []

    hs, vs, segments, sbis = [], [], [], []
    for item in res:
        coords = item[0]
        if len(coords) >= 4:
            x0, y0, xmax, ymax = coords[:4]
            if xmax - x0 == 0 or ymax - y0 == 0: continue
            if limit_to_hv:
                if abs(xmax - x0) > delta and abs(ymax-y0) > delta:
                    continue
                if abs(xmax - x0) > delta:
                    vs.append( ((x0, y0), (x0+xmax, y0+ymax)) )
                elif abs(ymax-y0) > delta:
                    hs.append( ((x0, y0), (x0+xmax, y0+ymax)) )
            segments.append([(x0, y0), (xmax, ymax)])
            sbis.append([x0, y0, xmax, ymax])

    sbis = np.array(sbis)

    if draw:
        im_out = image.copy()
        im_out = scaleData(im_out, 0, 255)
        try: im_out = cv2.cvtColor(im_out, cv2.COLOR_GRAY2BGR)
        except: pass
        im_out = fld.drawSegments(im_out, sbis, draw_arrow = True)
        imshow("match",im_out)

    if limit_to_hv:
        return hs, vs
    else:
        return segments, []




def findSegments(imageIn, method = None, eps = None, minLenght = None, show = False):
    """
    Find segments in the image. Caution : you should denoise image before (computeDenoise method)

    Usage
    -----
    findSegments(imageIn [, imageOut] [, method = 'hough_segment'] [, eps = 0.5] )

    Parameters
    ----------
    imageIn : ndarray       
        Image where to find segments
    imageOut : ndarray      
        Image out where to draw segments if found
    method : str            
        Method to use : 'hough_segment' for Hough transform method(segment)
                        'hough_lines' for Hough transform method (lines)
                        'lsd' for LineSegmentDetector method
    eps : float             
        Epsylon limit from which line/segment is considered as horizontal or vertical

    Returns
    -------
    ndarray
        List of segments/lines founded.
    """

    imageOut = imageIn.copy()
    try: imageOut = cv2.cvtColor(imageOut, cv2.COLOR_GRAY2RGB)
    except: pass

    #load default params from config file
    if method is None: method = "fsd"
    if eps is None: eps = 1.8
    if minLenght is None: minLenght = 10

    seg_list = []
    if method == 'hough_segment':
        Hlines = np.empty((0,1,4),'int')

        try:
            HlinesY = cv2.HoughLinesP(image=imageIn, rho=10, theta=np.pi/2, threshold=50, lines=None, minLineLength=minLenght, maxLineGap=1)
        except ValueError:
            print('Image must be grayscale UINT8')
        else:
            if HlinesY is not None:
                for i in range(0, len(HlinesY)):
                    p1 = (HlinesY[i][0][0], HlinesY[i][0][1])
                    p2 = (HlinesY[i][0][2], HlinesY[i][0][3])
                    if isHorizontal(p1, p2, eps) == True:
                        seg_list.append([p1, p2])
                        cv2.line(imageOut, p1, p2,(0,255,0), 1)
                    elif isVertical(p1, p2, eps) == True:
                        seg_list.append([p1, p2])
                        cv2.line(imageOut, p1, p2,(0,0,255), 1)

    elif method == 'hough_lines':
        # cv2.HoughLines(image, rho, theta, threshold[, lines[, srn[, stn[, min_theta[, max_theta]]]]]) > lines
        Hlines = cv2.HoughLines(imageIn, 1, np.pi/2, 80)
        w, h = imageIn.shape
        if Hlines is not None :
            for i in range(0,len(Hlines)):
                for rho,theta in Hlines[i]:
#                    rho,theta = Hlines[i][0]
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + h*(-b))
                    y1 = int(y0 + w*(a))
                    x2 = int(x0 - h*(-b))
                    y2 = int(y0 - w*(a))

                    pt1, pt2 = [(x1,y1),(x2,y2)]

                    seg_list.append([pt1, pt2])
                    
    elif method == "fsd":
        h, v = FastSegmentsDetector(imageIn, draw=show, limit_to_hv=True)
        if len(v) > 0:
            seg_list = h+v
        else:
            seg_list = h

    if show is True and method != "fsd" :

        for p1, p2 in seg_list:
            if len(imageOut.shape) > 2:
                imageOut = cv2.line(imageOut, (p1[0],p1[1]), (p2[0],p2[1]), (0, 0, 255), 2)
            else:
                imageOut = cv2.line(imageOut, (p1[0],p1[1]),(p2[0],p2[1]), 255, 2)

        cv2.imshow("lines detected "+method, imageOut)
        cv2.waitKey(0), cv2.destroyAllWindows()


    return seg_list, imageOut


def filterByWavelet(image, wavelet = None, wavelet_level = 1, noise_level = None, threshold_type = "soft"):
    """
    Denoise the given image with the specified wavelet filtering method.

    Usage
    -----
    filterByWavelet(image, wavelet [, wavelet_level = 1] [, noise_level = 3.5] [, threshold_type = 'soft'])

    Parameters
    ----------
    image : ndarray                
        Image to analyse.
    wavelet : object                
        Wavelet given to denoise.
    noise_level : float             
        Noise level of the image.
    threshold_type : str            
        Type of threshold used in wavelet filtering : 'soft', 'hard'

    Returns
    -------
    ndarray
        Filtered image
    """
    
    if wavelet is None: wavelet = getWavelet('COW')
    elif isinstance(wavelet,str): wavelet = getWavelet(wavelet)


    old_type = None
    if 'int' in str(image.dtype):
        old_type = image.dtype
        image = image.astype('float')/255
#            image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_64F)

    coeffs = pywt.wavedec2(image, wavelet, level = wavelet_level)
    # NOTE #
    # pywt.wavedec2 -> ( cAn, (cHn, cVn, cDn), ... (cH1, cV1, cD1))
    # coeffs[0] = LL
    # coeffs[1][1] = LH
    # coeffs[1][2] = HL
    # coeffs[1][3] = HH

    cA = coeffs[0] * 0.0
    new_coeffs = []
    for i in range(1, len(coeffs)):
        cH, cV, cD = coeffs[i]
        # cD *= 0.0 
        
        cD = pywt.threshold(cD, 10, "soft", substitute=0)
        new_coeffs.append((cH, cV, cD))
    coeffs = (cA, *new_coeffs)
    
    new_image = pywt.waverec2(coeffs, wavelet)

    if old_type is not None:
        new_image = histStretch(new_image)

    return new_image


def getImageItems(image, show_image = False):
    """
    Get an image analysis : sharpness measure, blobs count and lines count.

    Usage
    -----
    getImageItems(image)

    Params
    ------
    imageList : ndarray     
        List of images
    draw_stats : boolean   
        If True, statistics of the image are printed

    Returns
    -------
    ndarray
        Statistics array. Contains [sharpness_measure, blobs_count, lines_count]
    """
    image = luminanceCorrection(image)
    

    sharpness_measure = fourierIQA(image)
    blobs_count = len(blobsDetector(image))
    
    # Wavelet filtering with the wavelet given name and called getWavelet
    # No longer used: polynomial approach instead
    denoisedImage = filterByWavelet(image)

    try:
        lines_count = len(findSegments(denoisedImage, show = show_image))
    except: lines_count = 0

    if show_image == True:
        imshow("image", image)
        imshow("denoisedImage", denoisedImage)

    return [sharpness_measure, blobs_count, lines_count]




def getImagesStats(imageList, save_stats = True, count_blobs=False):
    """
    DEPRECATED.
    Give normalized images statistics :
        - sharpness measure based on the Fourier spectrum
        - amount of lines detected
        - amount of dusts detected
    For additional details see ImageProcessing  class -> getImageItems()

    Usage
    -----
    getImagesStats(imageList [, save_stats = False])

    Parameters
    ----------
    imageList : ndarray, optionnal   
        List of images.
    save_stats : bool, optionnal
        If True, statistics of the image are printed.

    Return
    ------
    ndarray
        Statistics arrays containing images info, sorted by each calcultated criteria
    """

    imagesStats = []

    index = 0
    for img in imageList:

        sharpness_measure, blobs_count, lines_count = getImageItems(img)
        imagesStats.append([index, sharpness_measure, blobs_count, lines_count])
        index += 1

    imagesStats = np.array(imagesStats)
    if np.max(imagesStats[:,1]) != 0:
        imagesStats[:,1] /= np.max(imagesStats[:,1])
    if np.max(imagesStats[:,2]) != 0:
        imagesStats[:,2] /= np.max(imagesStats[:,2])
    if np.max(imagesStats[:,3]) != 0:
        imagesStats[:,3] /= np.max(imagesStats[:,3])


    if save_stats is True:
        
        this_script_path = os.path.dirname(os.path.realpath(__file__))
        fig_num = 0
        fig_name = '/stats'+'_'+fig_num+'.png'
        while os.path.isfile(this_script_path+fig_name):
            fig_num += 1
            fig_name = '/stats'+'_'+fig_num+'.png'
        
        try:
            saveStats(imagesStats, fig_path = this_script_path, fig_name = fig_name)
        except Exception as e:
            print(e)

    return imagesStats
    



def main():
    image = np.random.random((500,500))
    image_items = getImageItems(image)
    print("Random image detected items, used to rank images in an autofocus system (see getImagesStats()): \n", 
          "Fourier spectrum based IQA: \t\t\t\t", image_items[0], "\n",
          "Opencv based blob detection - special case in IR: \t", image_items[1], "\n",
          "Detected lines based on Hough or FSD algorithm: \t", image_items[2])
    # for an autofocus algo, theses values are normalized
    

    
    
        
if __name__ == "__main__":
    main()
    
    
    
    
    
    