# iqa_algorithms

Few Image Quality Assessment (IQA) algorithms.
You may also find the DWT-based IQA algo and its Custom Orthogonal Wavelet from the paper [1, 2]. This algo was made to autofocus integrated circuit through infrared microscopy. It directly compete with the algorithm presented in [3] based on polynomial decomposition.

- statsToolbox contains several mathematical tools used in image analysis
- imagesToolbox contains several tool used in image analysis & processing
- all other files contains IQA algo. Note that these algo were used to compute image-per-image focus measure and then normalized to reach the "best" image, and compared in [3].


[1] ABELÉ, R., J.-L. DAMOISEAUX, R. EL MOUBTAHIJ et al. (2021). 
« Spatial Location in Integrated Circuits through Infrared Microscopy ». 
In : Sensors 21.6. 
ISSN : 1424-8220. 
DOI : 10.3390/s21062175.

[2] ABELÉ, R., D. FRONTE et al. (2018). 
« Autofocus in infrared microscopy ». 
In : 23rd IEEE International Conference on Emerging Technologies and Factory Automation, ETFA 2018, Torino, Italy, September 4-7, 2018. IEEE, p. 631-637. 
DOI : 10.1109/ETFA.2018.8502648.

[3] ABELÉ, R., R. EL MOUBTAHIJ et al. (2019). 
«FMPOD : A Novel FocusMetric Based on Polynomial Decomposition for Infrared Microscopy ». 
In : IEEE Photonics Journal 11.5, p. 1-17. 
ISSN : 1943-0647. 
DOI : 10.1109/JPHOT.2019.2940689.
