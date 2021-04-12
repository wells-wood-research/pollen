"""Module for segmenting and measuring pollen particles from images."""

from pathlib import Path
import typing as t

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.filters as sfi
import skimage.measure as sme
import skimage.morphology as smo
import skimage.segmentation
from PIL import Image, ImageEnhance
from scipy import ndimage as nd
from skimage.feature import canny
from skimage.feature import texture as sft
from skimage.io import imread
from skimage.util import img_as_ubyte


def inc_contrast(
    image_array: np.ndarray,
    contrast_times: float = 15,
    dilation: bool = False,
    dilation_selem_size: int = 10,
) -> np.ndarray:
    """Increases contrast of image."""
    # Increase contrast of input to clarify edges
    enhance = ImageEnhance.Contrast(image_array)
    cont = enhance.enhance(contrast_times)
    cont = np.array(cont)

    if dilation == True:
        # Dilation of image - This shrinks pollen silhouettes to help with clustering
        selem = smo.disk(dilation_selem_size)
        dilated = smo.dilation(cont, selem)
        return dilated
    else:
        return cont


def get_markers(
    image_array: np.ndarray,
    evened_selem_size: int = 4,
    markers_contrast_times: float = 15,
    markers_sd: float = 0.25,
) -> np.ndarray:
    """Finds the highest and lowest grey scale values for image flooding."""
    selem = smo.disk(evened_selem_size)
    evened = sfi.rank.mean_bilateral(
        inc_contrast(image_array, contrast_times=markers_contrast_times), selem
    )
    # Markers defined by highest and lowest grey levels.
    # Level reduced/decreased by 1/4 SD
    high = np.max(evened)
    low = np.min(evened)
    std = np.std(evened)
    neatarray = np.array(image_array)
    markers: np.ndarray = np.zeros_like(neatarray)
    markers[evened < low + (markers_sd * std)] = 3
    markers[evened > high - (markers_sd * std)] = 2
    return markers


def segment(
    self,
    sigma=1,
    canny_lt=100,
    canny_ht=1,
    contrast_times=10,
    dilated_evened_selem_size=2,
    evened_selem_size=10,
    dilation_selem_size=4,
    markers_meaned_selem_size=4,
    markers_contrast_times=15,
    markers_sd=0.25,
    return_mask=False,
):

    selem = smo.disk(evened_selem_size)
    evened = sfi.rank.mean_bilateral(
        inc_contrast(self, contrast_times=contrast_times), selem
    )
    selem = smo.disk(dilated_evened_selem_size)
    devened = sfi.rank.mean_bilateral(
        inc_contrast(
            self,
            contrast_times=contrast_times,
            dilation=True,
            dilation_selem_size=dilation_selem_size,
        ),
        selem,
    )
    # Canny edge detection filter detects object edges and outputs boolean array
    elevation_map = canny(
        devened, sigma=sigma, low_threshold=canny_lt, high_threshold=canny_ht
    )

    # Convert canny output binary array to use as mask
    elevation_map = elevation_map.astype(int)

    # Watershed function floods regions from markers
    segmentation = skimage.segmentation.watershed(
        evened,
        get_markers(
            self,
            evened_selem_size=markers_meaned_selem_size,
            markers_contrast_times=markers_contrast_times,
            markers_sd=markers_sd,
        ),
        mask=elevation_map,
        connectivity=1,
    )
    # thresh = sfi.threshold_otsu(segmentation)
    # Fill smaller holes within regions - usually due to noise; Label each region
    segmentation = nd.binary_fill_holes(segmentation)
    # elevation_map2 = canny(segmentation, sigma = 1 ,low_threshold=70 ,high_threshold=1)
    # segmentation = skimage.segmentation.watershed(elevation_map2, markers,mask=elevation_map, connectivity = 0.5)
    label = sme.label(segmentation)
    if return_mask == True:
        return elevation_map
    else:
        return label


def measure_props(
    path: Path,
    sigma=1,
    canny_lt=100,
    canny_ht=1,
    contrast_times=10,
    dilated_evened_selem_size=2,
    evened_selem_size=10,
    dilation_selem_size=4,
    markers_meaned_selem_size=4,
    markers_contrast_times=15,
    markers_sd=0.25,
):
    img = Image.open(path).convert("L")
    w, h = img.size
    area = w * h
    no = []
    tempdf = pd.DataFrame()
    for i, region in enumerate(
        sme.regionprops(
            segment(
                img,
                sigma=sigma,
                canny_lt=canny_lt,
                canny_ht=canny_ht,
                contrast_times=contrast_times,
                dilated_evened_selem_size=dilated_evened_selem_size,
                evened_selem_size=evened_selem_size,
                dilation_selem_size=dilation_selem_size,
                markers_meaned_selem_size=markers_meaned_selem_size,
                markers_contrast_times=markers_contrast_times,
                markers_sd=markers_sd,
            )
        )
    ):
        # Attempt to filter out non-pollen regions using size and circularity factor
        # and (region.perimeter**2)/(4*math.pi*region.area) < 1.5:
        if area > region.area > 250:
            no.append(i)
            coins: np.ndarray = imread(path, as_gray=True)
            coins = img_as_ubyte(coins)
            # Get region bounding box and crop image to it so GLCM functions only apply to relevant region
            coord: t.Tuple[int, int, int, int] = region.bbox
            boxx = coins[coord[0] : coord[2], coord[1] : coord[3]]
            # Defines inputs for GLCM
            distances = np.array([1])
            angles = np.array(
                [
                    0,
                    np.pi / 8,
                    np.pi / 4,
                    3 * (np.pi / 8),
                    np.pi / 2,
                    5 * (np.pi / 8),
                    3 * (np.pi / 4),
                    7 * (np.pi / 8),
                ]
            )
            # if function is because some bounding box coordinates were coming back incorrect
            if boxx.size > 0:
                glcm = sft.greycomatrix(boxx, distances, angles, levels=256)
                correlation = sft.greycoprops(glcm, prop="correlation").mean()
                contrast = sft.greycoprops(glcm, prop="contrast").mean()
                dissimilarity = sft.greycoprops(glcm, prop="dissimilarity").mean()
                homogeneity = sft.greycoprops(glcm, prop="homogeneity").mean()
                ASM = sft.greycoprops(glcm, prop="ASM").mean()
                energy = sft.greycoprops(glcm, prop="energy").mean()
                tempdf = tempdf.append(
                    pd.Series(
                        [
                            region.area,
                            region.perimeter,
                            region.major_axis_length,
                            region.minor_axis_length,
                            region.equivalent_diameter,
                            region.eccentricity,
                            region.extent,
                            contrast,
                            dissimilarity,
                            homogeneity,
                            ASM,
                            energy,
                            correlation,
                            i,
                            region.bbox,
                            region.local_centroid,
                        ],
                        name=path.stem,
                        index=[
                            "Area",
                            "Perimeter",
                            "Major_axis_length",
                            "Minor_axis_length",
                            "Equivalent_diameter",
                            "Eccentricity",
                            "Extent",
                            "Contrast",
                            "Dissimilarity",
                            "Homogeneity",
                            "ASM",
                            "Energy",
                            "Correlation",
                            "P-no",
                            "Bounding_box",
                            "Local_centroid",
                        ],
                    )
                )
            else:
                continue
    # test = len(no)
    # countdf.append(pd.Series([test], name=path.stem))

    return tempdf


def run_analysis_on_images(inputfolder: Path):
    measuresdf = pd.DataFrame(
        columns=[
            "Area",
            "Perimeter",
            "Major_axis_length",
            "Minor_axis_length",
            "Equivalent_diameter",
            "Eccentricity",
            "Extent",
            "Contrast",
            "Dissimilarity",
            "Homogeneity",
            "ASM",
            "Energy",
            "Correlation",
            "P-no",
            "Bounding_box",
            "Local_centroid",
        ]
    )

    for i, path in enumerate(inputfolder.glob("*.jpg")):
        CANNY_SIGMA = 1
        CANNY_LOW_THRESHOLD = 100
        CANNY_HIGH_THRESHOLD = 2
        CONTRAST_INCREASE = 50
        DILATION_EVENED_CONTRASTING_DISK_SELEM_SIZE = 2
        DILATION_SELEM_SIZE = 4
        EVENED_DISK_SELEM_SIZE = 4
        MARKERS_MEANED_SELEM_SIZE = 4
        MARKERS_CONTRAST_TIMES = 100
        MARKERS_SD_TIMES = 0.25

        coins = imread(path, as_gray=True)

        output = measure_props(
            path,
            sigma=CANNY_SIGMA,
            canny_lt=CANNY_LOW_THRESHOLD,
            canny_ht=CANNY_HIGH_THRESHOLD,
            contrast_times=CONTRAST_INCREASE,
            dilated_evened_selem_size=DILATION_EVENED_CONTRASTING_DISK_SELEM_SIZE,
            evened_selem_size=EVENED_DISK_SELEM_SIZE,
            dilation_selem_size=DILATION_SELEM_SIZE,
            markers_meaned_selem_size=MARKERS_MEANED_SELEM_SIZE,
            markers_contrast_times=MARKERS_CONTRAST_TIMES,
            markers_sd=MARKERS_SD_TIMES,
        )
        measuresdf = measuresdf.append(output)

        inp = Image.open(path).convert("L")
        plt.fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
            nrows=1, ncols=5, figsize=(10, 15)
        )  # , sharex=True, sharey=True)
        ax1.imshow(inp, cmap="gray")
        ax1.axis("off")
        ax1.set_title("Raw Image")
        ax2.imshow(
            inc_contrast(
                inp,
                contrast_times=CONTRAST_INCREASE,
                dilation=False,
                dilation_selem_size=DILATION_SELEM_SIZE,
            ),
            cmap="gray",
        )
        ax2.axis("off")
        ax2.set_title("Contrasted Image")
        ax3.imshow(
            get_markers(
                inp,
                evened_selem_size=EVENED_DISK_SELEM_SIZE,
                markers_contrast_times=MARKERS_CONTRAST_TIMES,
                markers_sd=MARKERS_SD_TIMES,
            ),
            cmap="gray",
        )
        ax3.axis("off")
        ax3.set_title("Markers")
        ax4.imshow(
            segment(
                inp,
                sigma=CANNY_SIGMA,
                canny_lt=CANNY_LOW_THRESHOLD,
                canny_ht=CANNY_HIGH_THRESHOLD,
                contrast_times=CONTRAST_INCREASE,
                dilated_evened_selem_size=DILATION_EVENED_CONTRASTING_DISK_SELEM_SIZE,
                evened_selem_size=EVENED_DISK_SELEM_SIZE,
                dilation_selem_size=DILATION_SELEM_SIZE,
                markers_meaned_selem_size=MARKERS_MEANED_SELEM_SIZE,
                markers_contrast_times=MARKERS_CONTRAST_TIMES,
                markers_sd=MARKERS_SD_TIMES,
                return_mask=True,
            ),
            cmap="gray",
        )
        ax4.axis("off")
        ax4.set_title("Mask")
        ax5.imshow(
            segment(
                inp,
                sigma=CANNY_SIGMA,
                canny_lt=CANNY_LOW_THRESHOLD,
                canny_ht=CANNY_HIGH_THRESHOLD,
                contrast_times=CONTRAST_INCREASE,
                dilated_evened_selem_size=DILATION_EVENED_CONTRASTING_DISK_SELEM_SIZE,
                evened_selem_size=EVENED_DISK_SELEM_SIZE,
                dilation_selem_size=DILATION_SELEM_SIZE,
                markers_meaned_selem_size=MARKERS_MEANED_SELEM_SIZE,
                markers_contrast_times=MARKERS_CONTRAST_TIMES,
                markers_sd=MARKERS_SD_TIMES,
            ),
            cmap="gray",
        )
        ax5.axis("off")
        ax5.set_title("Labelled")
        plt.show(plt.fig)

        cols = 2
        fig = plt.figure(figsize=(20, 2), constrained_layout=True)
        gs = fig.add_gridspec(1, output["Bounding_box"].size)
        for i, _ in enumerate(output["Bounding_box"]):
            b = coins[
                output["Bounding_box"][i - 1][0] : output["Bounding_box"][i - 1][1],
                output["Bounding_box"][i - 1][2] : output["Bounding_box"][i - 1][3],
            ]
            cols += 1
            fig.add_subplot(gs[0, i:], title="BBox" + f"{i}").axis("off")

            plt.imshow(b)
        plt.show()
        pd.set_option("display.max_columns", 500)
        display(output)


if __name__ == "__main__":
    # Global variables
    inputfolder = Path("Saps excuded/Included SAPS")
    run_analysis_on_images(inputfolder)
