# -*- coding: utf-8 -*-
"""
@author: raph988

In this script is developped a tool for IQA based on Discrete Wavelet Transform. 
A Custom Orthogonal Wavelet is used as it is presented in [1] as adapted for IR microscopy. 
Other wavelet can be used in this script.


[1] ABELE, Raphael, FRONTE, Daniele, LIARDET, Pierre-Yvan, et al. 
Autofocus in infrared microscopy. 
In : 2018 IEEE 23rd International Conference on Emerging Technologies and Factory Automation (ETFA). 
IEEE, 2018. p. 631-637.
"""


import numpy as np
import math
import pywt
import cv2
from imagesToolbox import getWavelet
from statsToolbox import StatsBox
statsTools = StatsBox()


class DWT_coeffs_helper():
    """ The purpose of this class is to organize the wavelet coefficients to ease there use. """
    
    def __init__(self, _coeffs_):
        self.coeffs = _coeffs_
        self.max_lvl = len(_coeffs_)-1


    def getApproximation(self):
        return self.coeffs[0]


    def getCoeffs(self, level):
        if level > self.max_lvl :
            s = "Level asked to high... : "+ str(level)+" asked, max is "+ str(self.max_lvl)
            raise Exception(s)
        elif level < 1 :
            s = "Level asked must de upper than 0..."
            raise Exception(s)

        (cHn, cVn, cDn) = self.coeffs[self.IoL(level)]
        return (cHn, cVn, cDn)


    def getHorizontalCoeffs(self, level=None):
        res = []
        if level is None:
            for i in range(0, self.max_lvl):
                res.append(self.getCoeffs(i)[0])
        else:
            res.append(self.getCoeffs(level)[0])
        return res


    def getVerticalCoeffs(self, level=None):
        res = []
        if level is None:
            for i in range(0, self.max_lvl):
                res.append(self.getCoeffs(i)[1])
        else:
            res.append(self.getCoeffs(level)[1])
        return res


    def getDiagonalCoeffs(self, level=None):
        res = []
        if level is None:
            for i in range(0, self.max_lvl):
                res.append(self.getCoeffs(i)[2])
        else:
            res.append(self.getCoeffs(level)[2])
        return res


    def updateCoeffs(self, level, newCoeffs):
        self.coeffs[self.IoL(level)] = newCoeffs


    def IoL(self, level):
        """ Initially, coeffs are ordinantes like that: [cA3, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1)]
            This function translate the level expected in the corresponding index in the coefficients data.
        """
        index = self.max_lvl - level + 1
        return index




def showWaveletDecomp(image = None, wavelet = None, w_lvl = 3, thresh_value = None):
    """
    Draw the input image decomposed by given wavelet.

    Usage
    -----
    showWaveletDecomp(image, wavelet [, drawFourier = False])

    Parameters
    ----------
    image : ndarray         
        Image to correct. The default is None.
    wavelet : Wavelet, optional    
        Wavelet to use in the decomposition. The default is None.
    w_lvl : int, optional         
        Decomposition maximum level. The default is 3.
    thresh_value : bool, optional  
        If True, the decomposition coeffs will be thresholded before the draw.. The default is None.

    Returns
    -------
    None.
    """

    if image is None or wavelet is None or w_lvl is None:
        raise ValueError("image is None or wavelet is None or w_lvl, they shouldn't.")


    coeffs = pywt.wavedec2(image, wavelet, level = w_lvl)

    coeffs_helper = DWT_coeffs_helper(coeffs)
    for i in range(1, w_lvl+1, 1):
        c = coeffs_helper.getCoeffs(i)[:2]
        for j in range(0, len(c)):
            subb = c[j]
            im = cv2.resize(subb, (int(image.shape[1]), int(image.shape[0])), interpolation = cv2.INTER_CUBIC) #NEAREST )
            title = "lvl "+ str(i)+ ", subb "+str(j)
            im = cv2.convertScaleAbs(abs(im), 256)
            cv2.imshow(title, im)



def threshWaveletCoeffs(coeffs, thresh_type = "soft"):
    """
    Compute image entropy based on its wavelet decomposition.

    Usage
    -----
    threshWaveletCoeffs(coeffs [, thresh_type])

    Parameters
    ----------
    coeffs : ndarray        
        Array of wavelet coefficients
    thresh_type : str       
        Threshold type to apply of coeffs; default is "soft", can be "hard"

    Returns
    -------
    ndarray
        Thresholded coeffs
    """

    mad = np.std(coeffs[0])
    noise_variance = (mad/0.6754)**2
    threshold = math.sqrt(noise_variance)*math.sqrt(math.log2(1.2)*10)
    coeffs[0] = pywt.threshold(coeffs[0], threshold, thresh_type)

    for i in range(1, len(coeffs)):
        tmp = coeffs[i][2]
        mad = np.std(tmp)
        noise_variance = (mad/0.6754)**2
        coeffs[i] = pywt.threshold(coeffs[i], noise_variance, thresh_type, substitute=0)


    return coeffs






def displayWaveletFunc(self, wavelet, resolution=20):
    print('About this wavelet: \n', wavelet)
    print('--- Vanishing moments (w):', wavelet.vanishing_moments_psi)
    print('--- Vanishing moments (s):', wavelet.vanishing_moments_phi)
    print('--- Filters :')
    for f in wavelet.filter_bank:
        print('\t', f)
    
    level = resolution
    try:
        [phi, psi, x] = wavelet.wavefun(level=level)
    except:
        [phi_d, psi_d, phi_r, psi_r, x] = wavelet.wavefun(level=level)
        phi, psi = phi_d, psi_d
        
    import pylab as pl
    pl.figure(figsize=(4,3))
    delta = 0 # 1.5 (cow) 5 (new97, cdf97)
    pl.plot(x-delta, psi)
    # pl.title("Wavelet")
    
    pl.figure(figsize=(4,3))
    pl.plot(x-delta, phi)
    # pl.title("Scale")
    pl.show()




def waveletIQA(image, wavelet_name='COW', thresh_type=None, w_lvl = 3, statistic_name = "std"):
    """
    Perfom a wavelet decomposition of the image and analyze its coefficients.
    See also :
        ABELE, Raphael, FRONTE, Daniele, LIARDET, Pierre-Yvan, et al. 
        Autofocus in infrared microscopy. 
        In : 2018 IEEE 23rd International Conference on Emerging Technologies and Factory Automation (ETFA). 
        IEEE, 2018. p. 631-637.
        
    Usage
    -----
    waveletIQA(image, wavelet [, thresh=False] [, thresh] [, w_lvl] [, thresh_type="soft"])

    Parameters
    ----------
    image : ndarray         
        Image to estimate
    wavelet : Wavelet       
        Wavelet to use in the decomposition
    thresh_type : str       
        Threshold type to apply of coeffs; default is None, can be "soft" or "hard"
    w_lvl : int             
        Decomposition maximum level; default is 3.
    statistic_name : str    
        Name of the statistic to perform on the DWT coeffs; see StatsBox class.

    Returns
    -------
    float
        Estimated IQ.
    """


    wavelet = getWavelet(wavelet_name)
    if wavelet is None:
        raise Exception("-E- The specified wavelet cannt be loaded.")


    if wavelet is not None:
        max_lvl = pywt.dwt_max_level(len(image), wavelet.dec_len)
        if w_lvl is None or w_lvl > max_lvl:
            w_lvl = max_lvl
            print("Wavelet decomposition level given is too hight. Reaffected to ",w_lvl)

#            image = cv2.normalize(image  , None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
#            image = self.luminanceCorrection(image, 5)
        coeffs = pywt.wavedec2(image, wavelet, level = w_lvl)


        if thresh_type is not None and (thresh_type.lower() == "soft" or thresh_type.lower() == "hard"):
            coeffs = threshWaveletCoeffs(coeffs, thresh_type=thresh_type)
    else:
        coeffs = image

    coeffs_helper = DWT_coeffs_helper(coeffs)

    # Energie of each subbands xy in each decomposition lvl n #
    Exyn = []
    for lvl in range(1, w_lvl+1):
        c = coeffs_helper.getCoeffs(lvl)[:2]
        c = list(subb*subb for subb in c)
        res = []
        for i in range(0, len(c)):
            subb = c[i]
            stat_func = statsTools.getFunction(statistic_name)
            x = stat_func(subb)

            try:
                x
            except Exception as e:
                raise e
            else:
                res.append(x)


        Exyn.append( res )

    # Energie of each decomposition lvl
    if len(Exyn[0]) == 3:
        En = list( (e[0] + e[1])/2 -e[2] for e in Exyn )
    elif len(Exyn[0]) == 2:
        En = list( (e[0] + e[1])/2 for e in Exyn )

    # focus criterion
    sharpness = 0.0
    for i in range(0, len(En), 1):
        sharpness += pow( 2, i ) * En[i]

    return sharpness


    
    
def main():
    image = np.random.random((500,500))
    image_iqa = waveletIQA(image)
    print("A random image IQA based on COW wavelet: ", image_iqa)
    # for an autofocus algo, this IQA is normalized

    
    
        
if __name__ == "__main__":
    main()
    