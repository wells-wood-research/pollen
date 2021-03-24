import skimage.filters as sfi
import numpy as np
from PIL import Image, ImageEnhance
import skimage.morphology as smo
from pathlib import Path
import skimage.segmentation
from scipy import ndimage as nd
import skimage.measure as sme
from skimage.feature import canny, peak_local_max
from skimage.io import imread
from skimage.util import img_as_ubyte
from skimage.feature import texture as sft
import csv
import os
import matplotlib.pyplot as plt
import math
import cv2

inputfolder = Path("SAPS")
outputfolder = Path(os.getcwd())

# with open(outputfolder / "particle_logfull.csv", "w") as outf:
#     outf.write("Particle number, Area (µm), Perimeter (µm), Major axis length (µm), Minor axis length (µm), P/E ratio, Equivalent diameter (µm), Eccentricity, Extent, Local Centroid, Bounding box, GLCM contrast, GLCM dissimilarity, GLCM homogeneity, GLCM ASM, GLCM energy, GLCM correlation, \n")

def segment(path, outputfolder):
    img = Image.open(path).convert('L')
    neatarray = np.array(img)
    #For region exclusion and for future implementing of adaptability to different resolution images
    w,h = img.size
    area = w*h
    #Increase conrast of input to clarify edges
    enhance = ImageEnhance.Contrast(img)
    cont = enhance.enhance(10)
    cont = np.array(cont)
    selem = smo.disk(4)    
    dilated = smo.dilation(cont, selem)
    cont = enhance.enhance(100)
    cont = np.array(cont)
    #Dilation of image - This shrinks pollen silhouettes to help with clustering issues

   
    #Denoising filter. Devened used for edge detection while evened used to
    # accurately find highest and lowest grey values for watershed markers
    selem = smo.disk(2)
    devened = sfi.rank.mean_bilateral(dilated, selem)   
    selem = smo.disk(4)
    evened = sfi.rank.mean_bilateral(cont, selem)
    #Canny edge detection filter detects object edges and outputs boolean array
    elevation_map = canny(dilated, sigma = 0.8 ,low_threshold=100,high_threshold=1) 

    #Markers defined by highest and lowest grey levels. Level reduced/decreased by 1/4 SD 
    high = np.amax(evened)
    low = np.amin(evened)
    sd = np.std(evened) 

    elevation_map = elevation_map.astype(int)
    # print(elevation_map)

    markers = np.zeros_like(neatarray)
    markers[evened < low +(0.25*sd) ] = 3
    markers[evened > high-(0.25*sd)] = 2
    #Watershed function floods regions from markers
    segmentation = skimage.segmentation.watershed(evened, markers,mask=elevation_map, connectivity = 1)
    # thresh = sfi.threshold_otsu(segmentation)
    segmentation = nd.binary_fill_holes(segmentation)
    # elevation_map2 = canny(segmentation, sigma = 1 ,low_threshold=70 ,high_threshold=1) 
    # segmentation = skimage.segmentation.watershed(elevation_map2, markers,mask=elevation_map, connectivity = 0.5)

    
    label = sme.label(segmentation)
    # cv2.imshow('image',segmentation)
    plt.fig, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=1, ncols=4, figsize=(17,  17), sharex=True, sharey=True)
    ax1.imshow(dilated, cmap="gray")
    ax1.axis('off')
    ax1.set_title('contrast')
    ax2.imshow(elevation_map, cmap="gray")
    ax2.axis('off')
    ax2.set_title('canny')
    ax3.imshow(markers, cmap="gray")
    ax3.axis('off')
    ax3.set_title('markers')
    ax4.imshow(label, cmap="gray")
    ax4.axis('off')
    ax4.set_title('segmentation')
    plt.show(plt.fig)
    # for a in label:
    # for box in sme.regionprops(label):
    #     # print(box.area)
    #     if 1==1:#area > box.area > 250 and (box.perimeter**2)/(4*math.pi*box.area) < 2.5:
    #         coord = box.bbox            
    #         coins = imread(path, as_gray=True)
    #         coins = img_as_ubyte(coins)
    #         a = np.array([1])
    #         b = np.array([0, np.pi/8, np.pi/4, 3*(np.pi/8), np.pi/2, 5*(np.pi/8), 3*(np.pi/4), 7*(np.pi/8)])
    #         boxx = coins[coord[0]:coord[1],coord[2]:coord[3]]
            
            

    #         # print(box)
    #         if boxx.size>0:   
    #             glcm = sft.greycomatrix(boxx,a,b,levels=256)
    #             correlation = str(sft.greycoprops(glcm, prop='correlation')).split(' ')
    #             contrast = str(sft.greycoprops(glcm, prop='contrast')).split(' ')
    #             dissimilarity = str(sft.greycoprops(glcm, prop='dissimilarity')).split(' ')
    #             homogeneity = str(sft.greycoprops(glcm, prop='homogeneity')).split(' ')
    #             ASM = str(sft.greycoprops(glcm, prop='ASM')).split(' ')
    #             energy = str(sft.greycoprops(glcm, prop='energy')).split(' ')
    #         else:
    #             continue    
                
           
            
            # print(G)


 
    no = []
    
#     with open(outputfolder / "particle_logfull.csv", "a") as outcsv:
#         for i, region in enumerate(sme.regionprops(label)):
# #Take regions with large enough areas, smaller regions are background noise
#             if area > region.area > 250 and (region.perimeter**2)/(4*math.pi*region.area) < 2.5 and boxx.size>0:
#                 # print(glcm.ndim)
#                 PE = region.major_axis_length/region.minor_axis_length
#                 # print(region.bbox)
#                 no.append(i)
#                 outcsv.write(f"{path.stem}, {region.area}, {region.perimeter}, {region.major_axis_length}, {region.minor_axis_length}, {PE}, {region.equivalent_diameter}, {region.eccentricity}, {region.extent}, [{region.local_centroid}],list({region.bbox}), {contrast[1]}, {dissimilarity[1]}, {homogeneity[1]}, {ASM[1]}, {energy[1]}, {correlation[1]}, \n")
#     test = len(no)
#     with open(outputfolder / "particle_nofull.csv", "a") as outcsv:          
#         # outcsv.write("Image, test")
#         outcsv.write(f"{path.stem}, {test}, \n")
        
for path in inputfolder.glob("*.jpg"):

    segment(path, outputfolder)