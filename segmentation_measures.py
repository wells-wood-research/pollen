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
import pandas as pd

inputfolder = Path("SAPS")
outputfolder = Path(os.getcwd())
#Set dictionary names
measures = {}
count = {}
measures['Particle number'] = ['Area (µm)', 'Perimeter (µm)', 'Major axis length (µm)', 'Minor axis length (µm)', 'P/E ratio', 'Equivalent diameter (µm)', 'Eccentricity', 'Extent', 'Local Centroid', 'Bounding box', 'GLCM contrast', 'GLCM dissimilarity', 'GLCM homogeneity', 'GLCM ASM', 'GLCM energy', 'GLCM correlation']

def segment(path, outputfolder):
    img = Image.open(path).convert('L')
    neatarray = np.array(img)
#For region exclusion and for future implementing of adaptability to different resolution images
    w,h = img.size
    area = w*h
#Increase conrast of input to clarify edges; contrast works better at lower values for dilation so repeated at line 41
    enhance = ImageEnhance.Contrast(img)
    cont = enhance.enhance(10)
    cont = np.array(cont)
#Dilation of image - This shrinks pollen silhouettes to help with clustering issues
    selem = smo.disk(4)    
    dilated = smo.dilation(cont, selem)
    
    cont = enhance.enhance(100)
    cont = np.array(cont)

#Denoising filter. Devened used for edge detection while evened used to accurately find highest and lowest grey values for watershed markers
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
    
    markers = np.zeros_like(neatarray)
    markers[evened < low +(0.25*sd) ] = 3
    markers[evened > high-(0.25*sd)] = 2
#Convert canny output binary array to use as mask
    elevation_map = elevation_map.astype(int)
#Watershed function floods regions from markers
    segmentation = skimage.segmentation.watershed(evened, markers,mask=elevation_map, connectivity = 1)
    # thresh = sfi.threshold_otsu(segmentation)
#Fill smaller holes within regions - usually due to noise; Label each region
    segmentation = nd.binary_fill_holes(segmentation)
    # elevation_map2 = canny(segmentation, sigma = 1 ,low_threshold=70 ,high_threshold=1) 
    # segmentation = skimage.segmentation.watershed(elevation_map2, markers,mask=elevation_map, connectivity = 0.5)   
    label = sme.label(segmentation)
 
    for box in sme.regionprops(label):
#Attempt to filter out non-pollen regions using size and circularity factor
        if area > box.area > 250 and (box.perimeter**2)/(4*math.pi*box.area) < 2.5:
            coins = imread(path, as_gray=True)
            coins = img_as_ubyte(coins)
#Get region bounding box and crop image to it so GLCM functions only apply to relevant region
            coord = box.bbox                
            boxx = coins[coord[0]:coord[1],coord[2]:coord[3]]
#Defines inputs for GLCM
            distances = np.array([1])
            angles = np.array([0, np.pi/8, np.pi/4, 3*(np.pi/8), np.pi/2, 5*(np.pi/8), 3*(np.pi/4), 7*(np.pi/8)])
#if function is because some bounding box coordinates were coming back incorrect
            if boxx.size>0:   
                glcm = sft.greycomatrix(boxx,distances,angles,levels=256)
                correlation = str(sft.greycoprops(glcm, prop='correlation')).split(' ')
                contrast = str(sft.greycoprops(glcm, prop='contrast')).split(' ')
                dissimilarity = str(sft.greycoprops(glcm, prop='dissimilarity')).split(' ')
                homogeneity = str(sft.greycoprops(glcm, prop='homogeneity')).split(' ')
                ASM = str(sft.greycoprops(glcm, prop='ASM')).split(' ')
                energy = str(sft.greycoprops(glcm, prop='energy')).split(' ')
            else:
                continue    
#Empty list for detected no. of particles
    no = []

    for i, region in enumerate(sme.regionprops(label)):
#Take regions with aforementioned criteria
            if area > region.area > 250 and (region.perimeter**2)/(4*math.pi*region.area) < 2.5 and boxx.size>0:
#Calculate P/E ratio                
                PE = region.major_axis_length/region.minor_axis_length
#Add to count list
                no.append(i)
#Add key:value pair to dictionary of Filename:Measures
                measures[path.stem+f'{i}'] = [region.area, region.perimeter, region.major_axis_length, region.minor_axis_length, PE, region.equivalent_diameter, region.eccentricity, region.extent, region.local_centroid,region.bbox, contrast[1], dissimilarity[1], homogeneity[1], ASM[1], energy[1], correlation[1]]
#Add count to seperate dictionary
    test = len(no)
    count[path.stem] = [test]
        
for path in inputfolder.glob("*.jpg"):

    segment(path, outputfolder)