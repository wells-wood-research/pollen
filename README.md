# 'PyPollen' - A python module towards the automation of automated pollen species identificaiton from light microsopy images

This repository contains scripts and IPython Notebooks used in the development of PyPollen. Files are intended for use by future developers.

PyPollen.py is the module file, containg 5 functions, with 'get_props' being the primary function to segment grains from images and measure their geometric and textural
properties. 'SVM_error_prediction_model.sav' is required to be stored locally for this, which contains an SVM model for the prediction of whether the output of get_props is
erroneous or not. 'true_values.csv' is also required for the use of 'optimise_segmentation_variables'.

'FINAL_measure_adj_atd.csv' contains the output of 'get_props' applied to the full image dataset used in development of this module, adjusted for magnification and fully annotated. For development and machine learning this csv should be used.
'PaldatNormalisedFinal.csv' contains the morphological descriptions for 4181 species scraped from PalDat which have been manipulated to provide succinct categories which are applicable in classification.

IPython Notebooks contains self-explanatory annotations and markdown.
