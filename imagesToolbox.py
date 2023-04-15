# -*- coding: utf-8 -*-
"""
@author: raph988

This script contains several tools often used in image analysis and image processing.

"""


import cv2
import numpy as np
from os import listdir, rename, path
from io import StringIO
import pandas as pd
import re
import math
import pywt
import sys
from skimage import filters
from matplotlib import pyplot as plt
from skimage import exposure

try:
    import _pickle as pickle # = cPickle which is faster than pickle
except:
    import pickle



class Image(np.ndarray):
    def __new__(cls, _input_array, colored=None, info=None):

        is_colored= False
        if len(_input_array.shape) > 2 and _input_array.shape[2] == 3:
            is_colored=True

        if colored is None: colored = is_colored
        input_array=None
        if colored is True and not is_colored:
            try: input_array = cv2.cvtColor(_input_array, cv2.COLOR_GRAY2RGB)
            except: input_array=_input_array
        elif colored is False and is_colored:
            try: input_array = cv2.cvtColor(_input_array, cv2.COLOR_RGB2GRAY)
            except: input_array=_input_array
        else:input_array=_input_array

        obj = np.asarray(input_array).view(cls)
        obj.info = info
        obj.colored = colored
        obj.w, obj.h = input_array.shape[1], input_array.shape[0]

        return obj

    def __array_finalize__(self, obj):

        if obj is None: return
        self.info = getattr(obj, 'info', None)
        self.colored = getattr(obj, 'colored', False)
        self.w, self.h = getattr(obj, 'w', None), getattr(obj, 'h', None)


class vec2():
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __eq__(self, p):
        if not isinstance(p, type(self)): raise Exception("Undefined comparison between "+str(type(self))+" and "+str(type(p))+".")
        if self.x == p.x and self.y == p.y:
            return True
        return False

    def __str__(self):
        return "vec2("+str(self.x)+","+str(self.y)+")"


class vec3(vec2):
    def __init__(self, x=0, y=0, z=0):
        super().__init__(x, y)
        self.z = z

    def __eq__(self, p):
        if not isinstance(p, type(self)) : raise Exception("Undefined comparison between "+str(type(self))+" and "+str(type(p))+".")
        if self.x == p.x and self.y == p.y and self.z == p.z:
            return True
        return False

    def __str__(self):
        p2 = super().__str__()[:-1]
        return p2+","+str(self.y)+")"




def isHorizontal(p1,p2, eps):
    """
    Determine if segment is horizontal, according to tolerance Epsilon.

    Attributs
    ---------
    p1                      First point of segment : tuple
    p1                      Last point of segment : tuple
    eps                     Epsilon from which segment could be determined as horizontal

    Return
    ------
    Boolean                 True if horizontal, False otherwise

    """
    return True if (abs(p2[1] - p1[1]) < eps) else False



def isVertical(p1,p2, eps):
    """
    Determine if segment is vertical, according to tolerance Epsilon.

    Attributs
    ---------
    p1                      First point of segment : tuple
    p1                      Last point of segment : tuple
    eps                     Epsilon from which segment could be determined as vertical

    Return
    ------
    Boolean                 True if vertical, False otherwise
    """

    return True if (abs(p2[0] - p1[0]) < eps) else False


def computeSquarePoints(center, size):
    """
    Find the 2 extrem points of center whose center is given.

    Attributs
    ---------
    center                  Center point of square : tuple
    size                    Dimension of one side of square : int

    Return
    ------
    p1, p2                  Extrem points of the square (Tuple, Tuple)
    """
    p1 = (center[0]-size, center[1]-size)
    p2 = (center[0]+size, center[1]+size)
    return p1, p2


def isOutOfImage(p, shape):
    """
    Determine if a point is in or out of an image

    Attributs
    ---------
    p                       Coordinates of the considering point : tuple
    shape                   Dimensions of the image : tuple

    Return
    ------
    Boolean                 True if point is out, False otherwise
    """
    if p[0] < 0 or p[0] > shape[0] or p[1] < 0 or p[1] > shape[1]:
        return True
    return False


def windowExtractor(image, xmin, xmax, ymin, ymax):
    w, h = image.shape
    if xmin is not None and xmin < 0: xmin = 0
    if ymin is not None and ymin < 0: ymin = 0
    if xmax is not None and xmax >= w: xmax = w-1
    if ymax is not None and ymax >= h: ymax = h-1
    # print('shape', w, h)
    res = image[ymin:ymax, xmin:xmax]
    return res




def histStretch(data, max_=255):
    p2, p98 = np.percentile(data, (2,98))
    return exposure.rescale_intensity(data, in_range=(p2,p98))

    # (x - x.mean()) / x.std() center the data around zero
    # 255.0/( 1.0 + exp(-(data-mean)/sd) );
    data = np.array(data)
    mean = data.mean()
    std = data.std()
    if std == 0.0: std = 0.000000001

    if max_ <= 1.0:
        return np.array(max_/( 1.0 + np.exp(-(data-mean)/std)), dtype=np.float32)

    return np.array(max_/( 1.0 + np.exp(-(data-mean)/std)), dtype=np.uint8)
    # return data


def scaleData(array, min_val=0, max_val=255, type_number = 'uint8'):
    """
    Set data value between min and max choosen with a data type

    Attributs
    ---------
    array                   Data to setup : ndarray
    min_val                 Min value of output data
    max_val                 Max value of output data
    type_number             Type of data to set

    Return
    ------
    Data reset             ndarray
    """
    if len(array) < 1:
        return array
    elif isinstance(array[0], list) or isinstance(array[0], np.ndarray):
        old_type = type(array)
        if not isinstance(array[0], np.ndarray):
            array = np.array(array)
        shape = array.shape[:]
        out = scaleData(array.flatten(), min_val, max_val, type_number)
        out = out.reshape(shape)
        if type(out) != old_type:
            out = old_type(out)
        return out


    # starting to scale data between 0 & 1 ( = normalization)
    old_type = None
    if not isinstance(array, np.ndarray):
        old_type = type(array)
        array = np.array(array)

    _min_ = np.min(np.ravel(array))
    _max_ = np.max(np.ravel(array))
    if (_max_ - _min_) == 0.0 or len(array) <= 2:
        return array

    norm = (array.astype('float') - _min_) / (_max_ - _min_)

    # set values between min an max
    if (min_val, max_val) != (0, 1):
        norm = (norm*(max_val-min_val) + min_val).astype(type_number)

    if old_type is not None:
        norm = old_type(norm)

    return norm


def loadBitmaps(path, image_names):
    """
    Load matrix from directory where images are stored as raster images (or bitmap) in multiple files

    Attributs
    ---------
    path                    Directory where to load images
    image_names             Generic name of data to load
    grayscale               If True images are grayscale, otherwise colored

    Return
    ------
    Matrix list
    """
    matList = []
    for f in listdir(path):
        if f.startswith(image_names):
            full_path = path+"/"+f
            mat = np.array(pd.read_csv(full_path, sep = '\t', header = None), dtype='uint8')

            matList.append(mat)
    return matList


def loadBitmaps2(path, image_names):
    """
    Load matrix from directory where images are stored as raster images (or bitmap) in a unique file, separated by '/' caractere

    Attributs
    ---------
    path                    Directory where to load images
    image_names             Generic name of data to load
    grayscale               If True images are grayscale, otherwise colored

    Return
    ------
    Matrix list
    """
    matList = []
    for f in listdir(path):
        if f.startswith(image_names):
            full_path = path+"/"+f
            text = open(full_path).read()
            for tab in text.split('/'):
                mat = np.array(np.loadtxt(StringIO(tab), dtype='uint8'))
                matList.append(mat)
    return matList



def loadImages(path, grayscale):
    """
    Load images from directory

    Attributs
    ---------
    path                    Directory where to load images
    grayscale               If True images are grayscale, otherwise colored

    Return
    ------
    Image list
    """
    images = []
    for f in listdir(path):
            i = Image(cv2.imread(path + f, not grayscale), colored=not grayscale)
            images.append(i)
    return images



def renameFiles(path, startwith, seps = '[\(\)]', nb_zeros = 2):
    """
    Rename all the files in the directory path with the names *(i)* into *00i* where i is a number.
    Params
    ------
    path    Directory containing the files
    seps    Separators delimiting the numbers

    """
    for f in listdir(path):
        if f.startswith(startwith):
            lname = re.split(seps, f)
            lname[1] = lname[1].zfill(nb_zeros)
            newName = ''.join(lname)
            rename(path+f, path+newName)


def meanImages(images, method = 0):
    """
    Mean of temporal sequencial images for impulsive noise reduction

    Parameterss
    ----------
    images : list
        Images of type ndarray.
    method : int, optional
        Which method to use ?
        0 -> manual mean of list images; 1 -> Non-local means denoising using images from list.
        The default is 0.

    Raises
    ------
    Exception
        No images.

    Returns
    -------
    total_mse : float
        The average mean square error.
    img : ndarray
        The averaged image.

    """

    mse = []
    
    images_norm = [scaleData(im, 0, 1) for im in images]
    if len(images_norm) < 1: 
        raise Exception("No images in stack or problem occured.")
    
    for i in range(0, len(images_norm)-1):
        for j in range(i+1, len(images_norm)-1):
            mse.append(getPSNR(images_norm[i], images_norm[j]))
    total_mse = np.mean(mse)
    
    if method == 0:
        img = scaleData(images_norm[0], 0, 1)
        for n in range(1,len(images_norm)):
            img = cv2.add(img, images_norm[n])
        img /= len(images_norm)
        img = scaleData(img, 0, 255, 'uint8')
    elif method == 1:
        # cv2.fastNlMeansDenoisingMulti(srcImgs, imgToDenoiseIndex, temporalWindowSize[, dst[, h[, templateWindowSize[, searchWindowSize]]]]) -> dst
        img = cv2.fastNlMeansDenoisingMulti(images, len(images)/2, len(images)/2 - 1, None, 5, 7, 21)
    return total_mse, img
    

def getWavelet(name = None):
    """
    Give the required wavelet from their filters bank.
    
    The COW wavelet is used in the paper :
    ABELE, Raphael, FRONTE, Daniele, LIARDET, Pierre-Yvan, et al. 
    Autofocus in infrared microscopy. 
    In : 2018 IEEE 23rd International Conference on Emerging Technologies and Factory Automation (ETFA). 
    IEEE, 2018. p. 631-637.
    

    Parameters
    ----------
    name : TYPE, optional
        Name of the wavelet required : 'haar', 'cdf_97', 'dmey', or 'cow'. The default is None.

    Raises
    ------
    Exception
        Unvalid wavelet name.

    Returns
    -------
    pywt.Wavelet
        Wavelet.
    """

    # load default params from config file
    if name is None: name = 'haar'

    if name.lower() == 'haar':
        c = math.sqrt(2)/2
        decomposition_lowPass   =   [c, c]
        decomposition_highPass  =   [-c, c]
        reconstruction_lowPass  =   [c, c]
        reconstruction_highPass =   [c, -c]
        filters = [decomposition_lowPass, decomposition_highPass, reconstruction_lowPass, reconstruction_highPass]
        w = pywt.Wavelet('myHaar', filter_bank = filters)
        w.orthogonal = True
        w.biorthogonal = False
        return w

    elif name.lower() == 'cdf_97':
        decomposition_lowPass = [.026748757411, -.016864118443, -.078223266529, .266864118443]
        decomposition_lowPass = decomposition_lowPass + [.602949018236] + decomposition_lowPass[::-1]
        decomposition_lowPass = np.array(decomposition_lowPass)

        decomposition_highPass = [.0, .045635881557, -.028771763114, -.295635881557]
        decomposition_highPass = decomposition_highPass + [.557543526229] + decomposition_highPass[::-1]
        decomposition_highPass = np.array(decomposition_highPass)

        reconstruction_lowPass = np.multiply(decomposition_highPass, [1, -1, 1, -1, 1, -1, 1, -1, 1])
        reconstruction_highPass = np.multiply(decomposition_lowPass, [1, -1, 1, -1, 1, -1, 1, -1, 1])

        filters = [decomposition_lowPass, decomposition_highPass, reconstruction_lowPass, reconstruction_highPass]
        w = pywt.Wavelet('mycdf_97', filter_bank = filters)
        w.orthogonal = False
        w.biorthogonal = True
        return w

    elif name.lower() == 'new_97':
        decomposition_lowPass = [0.0, -0.0441, -0.0256, 0.2941]
        decomposition_lowPass = decomposition_lowPass + [0.5513] + decomposition_lowPass[::-1]
        decomposition_lowPass = np.array(decomposition_lowPass)
        decomposition_highPass = [0.0502, -0.0292, -0.1649, -0.5292]
        decomposition_highPass = decomposition_highPass + [1.2295] + decomposition_highPass[::-1]
        decomposition_highPass = np.array(decomposition_highPass)

        reconstruction_lowPass = np.multiply(decomposition_highPass, [1, -1, 1, -1, 1, -1, 1, -1, 1])
        reconstruction_highPass = np.multiply(decomposition_lowPass, [1, -1, 1, -1, 1, -1, 1, -1, 1])

        filters = [decomposition_lowPass, decomposition_highPass, reconstruction_lowPass, reconstruction_highPass]
        w = pywt.Wavelet('mynew_97', filter_bank = filters)
        w.orthogonal = False
        w.biorthogonal = True
        return w

    elif name.lower() == "mallat":
        decomposition_lowPass = [0, 0, 0.125, 0.375, 0.375, 0.125, 0]
        decomposition_highPass = [0, 0, 0, -2.0, 2.0, 0, 0]
        reconstruction_lowPass = [0.0078125, 0.054685, 0.171875, -0.171875, -0.054685, -0.0078125, 0]
        reconstruction_highPass = [0.0078125, 0.046875, 0.1171875, 0.65625, 0.1171875, 0.046875, 0.0078125]

        filters = [decomposition_lowPass, decomposition_highPass, reconstruction_lowPass, reconstruction_highPass]
        w = pywt.Wavelet('mymallat', filter_bank = filters)
        w.orthogonal = True
        w.biorthogonal = False
        return w

    elif name.lower() == "cow":
        k=math.sqrt(2)
        fb = pywt.orthogonal_filter_bank( [1/k, 1/k]*2 )
        w = pywt.Wavelet('COW', filter_bank = fb)
        w.orthogonal = True
        w.biorthogonal = False
        return w

    elif isinstance(name, tuple) or isinstance(name, list):
        try:
            return pywt.Wavelet('mytest', filter_bank = name)
        except :
            raise Exception('Wavelet filter banks not valid.')


    else :
        try:
            return pywt.DiscreteContinuousWavelet(name)
        except:
            print(name, type(name))
            print("Wavelet "+ str(name)+ " not implemented.")
            return None
        

def getMSE(image1, image2):
    """
    Compute the Mean Squared Error between two images

    Parameters
    ----------
    image1 : ndarray
        Image 1
    image2 : ndarray
        Image 2

    Return
    ------
    MSE: float
        
    """

    # err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    # err /= float(image1.shape[0] * image1.shape[1])
    
    s1 = np.absolute(np.subtract(image1.astype("float"), image2.astype("float")))
    s1 = s1.astype('float32')
    s1 = np.multiply(s1, s1)
    _sum = np.sum(s1)

    return float(_sum/len(image1))



def getPSNR(image1, image2):
    """
    Compute Pixel Signal to Noise Ration

    Attributs
    ---------
    image1 : ndarray
        Image 1
    image2 : ndarray
        Image 2

    Return
    ------
    PSNR: float
        
    """

    mse = getMSE(image1, image2)
    if mse == 0: mse = 0.001
    psnr = np.log10((255**2)/mse)

    return psnr


def getSTD(data):
    return math.sqrt(np.var(data, dtype=np.float64))

def getTenengrad(img):
    tmp = filters.sobel(img)
    out = np.sum(tmp**2)
    out = np.sqrt(out)
    return out

def getEnergy(img):
    out = np.sum(img[:])
    return out

def getSMD2(img):
    Ix = img.copy()
    Iy = img.copy()
    Ix[:,:-1] = np.diff(img, axis=1)**2
    Iy[:-1,:] = np.diff(img, axis=0)**2
    out = np.mean(math.sqrt((Iy+Ix).astype('float32')))
    return out


def getVollath(img):    
    image = img.copy()
    image = image.astype('float32')
    
    I1 = image.copy()
    I1[1:-1,:] = image[2:,:]
    
    I2 = image.copy()
    I2[1:-2,:] = image[3:,:]
    
    image = image*(I1-I2)
    
    out = np.mean(image)
    
    return out



def getBrenner(img):
    I1 = img.copy()
    I1[1:-1,:] = img[2:,:]
    I1[:,1:-1] = img[:,2:]
    
    I2 = img.copy()
    
    out = np.sum((I1 - I2)**2)
    
    return out  



def getMAD(array):
    """
    Mean Absolute Deviation

    Attributs
    ---------
    images                  Images to mean

    Return
    ------
    MAD
    """
    mean = np.mean(array)
    MAD = np.absolute(array - mean)
    return np.mean(MAD)


def blobsDetector(image):
    """
    Detect blobs due to dust on the microchip

    Attributs
    ---------
    image                   Image : ndarray

    Return
    ------
    Center(s) of blob(s) : Tuple
    """
    blob = np.zeros(image.shape)
    blob[image < np.median(np.sort(image))/2] = 255
    blob = cv2.dilate(blob, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations = 5)

    GaussianKernel_2D = cv2.getGaussianKernel(10, -1) * np.transpose(cv2.getGaussianKernel(10, -1))
    blob = cv2.filter2D(blob, 0, GaussianKernel_2D)

    detector = cv2.Simpldetector = cv2.SimpleBlobDetector_create()
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = 1
    params.blobColor = 255

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.8

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.8

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(blob)

    return keypoints



def correctBlobs(image, show = 0):
    """
    Correct blobs due to dust on the microchip with 2D gaussian mask

    Attributs
    ---------
    image                   Image to correct
    show                    Boolean for Visualization of blobs detected

    Return
    ------
    Corrected image : ndarray
    """
    blobCenters = blobsDetector(image)

    if blobCenters:
        if show: cv2.imshow('Blobs detected', cv2.drawKeypoints(image, blobCenters, None, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
        for c in blobCenters:
            size = int(c.size)
            coord_center = (int(round(c.pt[1])), int(round(c.pt[0])))

            while True: # prevent out-of-memory access with image boundaries
                p1, p2 = computeSquarePoints(coord_center, size+3)
                if isOutOfImage(p1,image.shape) or isOutOfImage(p2,image.shape):
                    size -=1
                else:
                    break

            kernelSize = (abs(p1[0]-p2[0]), abs(p1[1]-p2[1]))
            GaussianKernel_2D = cv2.getGaussianKernel(kernelSize[0], 0) * np.transpose(cv2.getGaussianKernel(kernelSize[1], 0))
            patch = (scaleData(GaussianKernel_2D, 0, 1) * np.median(np.sort((image)))).astype('uint8')
            image[p1[0]:p2[0], p1[1]:p2[1]] += patch
    return image



def convolve_pixel(image, pixel_coords, kernel= None):
    if kernel is None:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    kernel_size = len(kernel)

    x_min, x_max = pixel_coords[0]-math.floor(kernel_size/2), pixel_coords[0]+math.ceil(kernel_size/2)
    y_min, y_max = pixel_coords[1]-math.floor(kernel_size/2), pixel_coords[1]+math.ceil(kernel_size/2)
    patch = image[x_min:x_max, y_min:y_max]
    p = sum(cv2.filter2D(patch, -1, kernel).flatten())

    return p



def sobelXY(image, dx, dy, kernel_size = 3):
    """
    Apply sobel filter on one axis, in each direction (right to left and left to right)

    Attributs
    ---------
    image                   Image to filter : ndarray
    dx                      Order of derivative x (0 or 1): int
    dy                      Order of derivative y (0 or 1): int

    Return
    ------
    Image filtered : ndarray
    """
    # Sobel sur l'image normale
    sob1 = cv2.Sobel(image, -1, dx, dy, None, kernel_size)
    # return abs(sob1.astype(np.float64))

    # Sobel sur l'image retournee !
    sob2 = cv2.Sobel(np.rot90(image,2), -1, dx, dy, None, kernel_size)
    sob2 = np.rot90(sob2,2) # On remet l'image a l'endroit
    return cv2.add(sob1, sob2)


    
def scharr(image, kernel_size = 3, merge_xy=True):
    """
    Apply sobel filter on each axis (x&y)

    Attributs
    ---------
    image                   Image to filter : ndarray
    kernel                  Size of sobel kernel (must be odd) : int

    Return
    ------
    Filtered image on X and filtered image on Y : tuple(ndarray)
    """
    gx = abs(filters.scharr_v(image))
    gy = abs(filters.scharr_h(image))
    if merge_xy == False:
        return gx, gy
    else:
        gx = scaleData(gx, 0, 1, 'float64')
        gy = scaleData(gy, 0, 1, 'float64')
        res = (gx+gy)/2
        return (res*255).astype(np.uint8)



def sobelDiag(image):
    """
    Apply sobel filter along diagonals

    Attributs
    ---------
    image                   Image to filter : ndarray

    Return
    ------
    Filtered image on X and filtered image on Y : tuple(ndarray)
    """
    sobx = sobelXY(image, 1, 1)
    rot = np.rot90(image)
    soby = sobelXY(rot, 1, 1)
    soby = np.rot90(soby, 3)
    return sobx, soby


def getAdaptativeBinary(image):
    filtered = cv2.GaussianBlur(image, (5,5), 5, None, 5)
    ret, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return binary



def flatten_list(l):
    return flatten_list(l[0]) + (flatten_list(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]




def changeBrightnessContrast(input_img, brightness = 0, contrast = 0):

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf



def changeContrasts(image, amount = 50, increase = True):
    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    limit = min(np.mean(imghsv[:,:,2])*3, 200)

    factor = 1
    if increase is False: factor = -1


    imghsv[:,:,2] = [[max(pixel - amount*factor, 0) if pixel < limit  else min(pixel + amount*factor, 255) for pixel in row] for row in imghsv[:,:,2]]

    image = cv2.cvtColor(imghsv, cv2.COLOR_HSV2BGR)
    return image


def luminanceCorrection(im, limit = None):
    """
    Compute an adaptative threshold with contrast limitation.

    Usage
    -----
    luminanceCorrection(im [, limit = None])

    Parameters
    ----------
    image : ndarray         
        Image to analyse.

    Returns
    -------
    ndarray
        Image thresholded.
    """
    #load default param from config file
    if limit is None: limit = 1.5
    if 'int' not in str(im.dtype):
        im = (im*255).astype('uint8')
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(8,8))
    cl1 = clahe.apply(im)

    return cl1


def save_pyplot(fileName, obj, save_image):
    """
    Save of matplolib plot to a stand alone python script containing all the data and configuration instructions to regenerate the interactive matplotlib figure.

    Parameters
    ----------
    obj (object)        : Python object corresponding to the figure to save (matplotlib.pyplot).
    fileName (string)   : Path of the python script file to be created.

    Returns
    -------
    Return 0 is success, 1 otherwise.
    """


    cpt = 0
    fileName_tmp = fileName
    while path.isfile(fileName+".bat"):
        fileName = fileName_tmp + '('+str(cpt)+')'
        cpt += 1


    pkl_name = fileName+'.pkl'
    bat_name = fileName+'.bat'

    """ save the figure object as a pkl file """
    print("saving", pkl_name)
    try:
        with open(pkl_name,'wb') as fid:
            pickle.dump(obj, fid)
    except Exception as e:
        print("Failure while saving the figure data (pickle.dump).")
        print(e)
        return 1

    py_version = sys.version_info[0]
    N_indent=4
    short_pkl_name = re.sub(r'.*/', './', pkl_name)
    """ the python code to load the pkl file and run the figure """
    python_code = 'import matplotlib.pyplot as plt\n'
    python_code += 'import pickle\n'
    python_code += 'def main():\n'
    python_code += ' '*N_indent+'with open("'+short_pkl_name+'","rb") as fid:\n'
    python_code += ' '*(N_indent*2)+'ax = pickle.load(fid)\n'
    python_code += ' '*N_indent+'plt.show()\n'
    python_code += 'if(__name__=="__main__"):\n'
    python_code += ' '*N_indent+'main()\n'

    """ the bat file containing some instruction and the python code """
    try:
        with open(bat_name,'w') as f:
            # first, instructions to specify that next part of the file is the python code
            f.write('0<0# : ^\n')
            f.write("'''\n")
            f.write('@echo off\n')
            f.write('echo Running plyplot figure...\n')
            f.write('py -'+str(py_version)+' %~f0 %*\n')
            f.write('exit /b 0\n')
            f.write("'''\n")
            # then, the python code
            f.write(python_code)
    except Exception as e:
        print("Failure while saving the figure instructions (bat).")
        print(e)
        return 1

    if save_image==True:
        plt.savefig(fileName+".png", dpi=300)

    return 0


def main():
    pass

if __name__ == "__main__":
    main()

