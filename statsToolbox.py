# -*- coding: utf-8 -*-
"""
@author: raph988

In this script a tool box dedicated to statics often used in image analysis.
"""

import numpy as np
import cv2
from skimage import feature
from skimage.filters.rank import entropy as sk_entropy
from skimage.morphology import square
from skimage.measure import moments as sk_moments
from skimage.measure import moments_central as sk_moments_central
from skimage.measure import moments_hu as sk_moments_hu
from scipy.ndimage.measurements import histogram as scipy_hist
from scipy.fftpack import dct as scipy_dct
import math

from imagesToolbox import scaleData

class StatsBox():
    """
    Here are major statistic tools used in image processing.
    The functions are imported using their string name, or used as usual functions.
    """

    def __init__(self):
        self.func_dic = {"entropy"  :    self.getEntropy,
                         "var"      :    self.getVariance,
                         "covar"    :    self.getCovariance,
                         "std"      :    self.getSTD,
                         "ac"       :    self.getAC,
                         "ag"       :    self.getAG,
                         "mean"     :    self.getMean,
                         "kurtosis" :    self.getKurtosis,
                         "skewness" :    self.getSkewness,
                         "sum"      :    np.sum,
                         "median"   :    np.median}

        self.stats_names = list(self.func_dic.keys())


    def getFunction(self, stat_name):
        if stat_name.lower() not in self.func_dic.keys():
            raise Exception("No such a function implemented...")

        try:
            return self.func_dic[stat_name.lower()]
        except Exception as e:
            print(e)
            return None


    def getEntropy(self, data):
        """
        The Entropy function
        Could use scipy_entropy() of non-normalized data
        """
        return np.mean(sk_entropy(data, square(5)))

        def getProb(intensity, hist, tot):
            s1 = hist[intensity]
            prob = s1/tot
            return prob

        if max(data.flatten()) <= 1.0:
            data = scaleData(data, 0, 255, 'uint8')
        _max = 255

        hist = scipy_hist( data, 0, _max, 256 )

        tot = sum(hist)

        s=0
        for i in range(0, 256):
            pi = getProb(i, hist, tot)
            if pi == 0.0:
                pi = 0.00000000000000000000000000001
            s += pi*math.log2(pi)
        return -s


    def getBSE(self, img, block_size = 8):
        """
        Bayes-spectral-entropy of an input image, based on normalized discrete cosinus transform.
        Image is decomposed in subblocks, nDCT is computed of each subblock. BSE is the mean of entropy of each nDCT.

        Usage
        -----
        getBSE(self, img, block_size = 8)

        Parameters
        ----------
        img : ndarray           
            Sourcemage
        block_size : int        
            Size of the block (square) to construct

        Returns
        -------
        float    
            BSE.
        """
#        try:
#            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#        except:
#            raise Exception
#        else:
#            if max(img.flatten()) > 1.0:
#                img = scaleData(img, 0, 1, "float64")

        def get_nDCT_entropy(im):
            dct = scipy_dct(im, norm=None)
            return self.getEntropy(dct)

        if block_size is None or block_size > min(img.shape) :
            return get_nDCT_entropy(img)

        e_list = []
        s = block_size
        i, j = 0, 0
        while j+s <= img.shape[1]:
            while i+s <= img.shape[0]:
                b = img[i:i+s, j:j+s]
                e = get_nDCT_entropy(b)
                e_list.append(e)
                i+=s
            j+=s

        if len(e_list) == 0.0:
            return 0

        return float(sum(e_list)/len(e_list))


    def getCovariance(self, data):
        """
        The coviariance function
        """
        m = sk_moments(data, order=2)
        cr = m[0, 1] / m[0, 0]
        cc = m[1, 0] / m[0, 0]
        moments = sk_moments_central(data, cr=cr, cc=cc, order=2)
#        moments = sk_moments_normalized(moments, order=4)
        return moments[1,1]



    def getVariance(self, data):
        """
        The Variance function; said "second central moment"
        numpy -> (abs(x - x.mean())**2
        """
#        varx = np.mean(np.var(data, axis=0, dtype=np.float64))
#        vary = np.mean(np.var(data, axis=1, dtype=np.float64))
#        return (varx+vary)/2
        return np.var(data, dtype=np.float64)

        m = sk_moments(data, order=2)
        cr = m[0, 1] / m[0, 0]
        cc = m[1, 0] / m[0, 0]

        moments = sk_moments_central(data, cr=cr, cc=cc, order=4)
##        moments = sk_moments_normalized(moments, order=4)
#        vx = moments[2,0]
#        vy = moments[0,2]
#        return (vx+vy)/2

        moments = sk_moments_hu(moments)
        return moments[2]


    def getSTD(self, data):
        """
        The STD function
        Could use sqrt(var)
        """
        return math.sqrt(self.getVariance(data))

#        return math.sqrt(np.var(data, dtype=np.float64))
#        return np.std(data, axis=(0,1), dtype=np.float32)


    def getAC(self, data):
        """ The auto-correlation function from Scikit-image (use the Grey Level Coocurrence Matrix, GLCM)
        """
#        ac = correlate2d(data, data)
#        return np.mean(ac)
        data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        M = feature.greycomatrix(data, [1], [0], levels=256, symmetric=True, normed=True)
        return np.mean(feature.greycoprops(M, prop='correlation')) # homogeneity, correlation


    def getAG(self, data):
        """
        The Average Gradient (AG) function
        """
        g = cv2.Laplacian(data, cv2.CV_64F)
        return np.mean(g)

    def getMean(self, data):
        """
        The Mean function
        """
        return np.mean(data)

    def getSkewness(self, data):
        """
        The Skewness function; said third central moment
        """
#        sx = np.mean(scipy_skewness(data, axis=0))
#        sy = np.mean(scipy_skewness(data, axis=1))
#        return (sx+sy)/2
        m = sk_moments(data, order=2)
        cr = m[0, 1] / m[0, 0]
        cc = m[1, 0] / m[0, 0]

        moments = sk_moments_central(data, cr=cr, cc=cc, order=4)
#        moments = sk_moments_normalized(moments, order=4)
        sx = moments[3,0]/(np.power(moments[2,0], 3/2))
        sy = moments[0,3]/(np.power(moments[0,2], 3/2))
        return (sx+sy)/2

#        moments = sk_moments_hu(moments)
#        return moments[3]/(np.power(moments[2], 3/2))


    def getKurtosis(self, data):
        """
        The Kurtosis function; said fourth central moment
        """
        # Kurtosis = m4 / m2^2 where mx is the xth central moment
#        kx = np.mean(scipy_kurtosis(data, axis=0))
#        ky = np.mean(scipy_kurtosis(data, axis=1))
#        return (kx+ky)/2

        m = sk_moments(data, order=2)
        cr = m[0, 1] / m[0, 0]
        cc = m[1, 0] / m[0, 0]

        moments = sk_moments_central(data, cr=cr, cc=cc, order=4)
#        moments = sk_moments_normalized(moments, order=4)
        kx = moments[4,0]/(np.power(moments[2,0],2)) -3
        ky = moments[0,4]/(np.power(moments[0,2],2)) -3
        return (kx+ky)/2

#        moments = sk_moments_hu(moments)
#        return moments[4]/(np.power(moments[2],2)) -3

def main():
    pass


if __name__ == "__main__":
    main()