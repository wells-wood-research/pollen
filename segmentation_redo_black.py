"""Module for segmenting and measuring pollen particles from images."""

from pathlib import Path
import typing as t
import skimage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.filters as sfi
import skimage.measure as sme
import skimage.morphology as smo
import skimage.segmentation
from PIL import Image, ImageEnhance
from scipy import ndimage as nd
from scipy import stats
from skimage.feature import canny
from skimage.feature import texture as sft
from skimage.io import imread
from skimage.util import img_as_ubyte
import pickle

# Function to increase contast with PIL function, used throughout other functions
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


# Function to retrives maxima and minima of greyscale images for use as markers in watershed flood region filling
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
    # Markers defined by highest and lowest grey levels set as markers
    high = np.max(evened)
    low = np.min(evened)
    std = np.std(evened)
    neatarray = np.array(image_array)
    markers: np.ndarray = np.zeros_like(neatarray)
    # Level reduced/decreased by 1/4 SD
    markers[evened < low + (markers_sd * std)] = 3
    markers[evened > high - (markers_sd * std)] = 2
    return markers


# Function to identify grains in an images and extract these regions of the image
def segment(
    image_array: np.ndarray,
    sigma: float = 3,
    canny_lt: float = 25,
    canny_ht: float = 5,
    evened_contrast_times: float = 1,
    dilated_contrast_times: float = 8,
    dilated_evened_selem_size: int = 9,
    evened_selem_size: int = 4,
    dilation_selem_size: int = 9,
    markers_meaned_selem_size: int = 4,
    markers_contrast_times: float = 30,
    markers_sd: float = 0.25,
    return_mask: bool = False,
    use_longest_elements: bool = False,
    iterations=1,
):
    # Evened filter smoothes image to remove noise;
    # selem (Structuring element) flattens local areas within this element
    selem = smo.disk(evened_selem_size)
    evened = sfi.rank.mean_bilateral(
        inc_contrast(image_array, contrast_times=evened_contrast_times), selem
    )
    selem = smo.disk(dilated_evened_selem_size)
    # Dilation also applied for use in edge detection - Shrinks dark areas, pulling clustered grains apart from each other
    devened = sfi.rank.mean_bilateral(
        inc_contrast(
            image_array,
            contrast_times=dilated_contrast_times,
            dilation=True,
            dilation_selem_size=dilation_selem_size,
        ),
        selem,
    )
    # Canny edge detection filter detects object edges and outputs boolean array
    elevation_map = canny(
        devened, sigma=sigma, low_threshold=canny_lt, high_threshold=canny_ht
    )
    # 3x3 structuring element so binary closing detects true values in any neighbouring pixel as a connection
    s = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    if use_longest_elements == False:
        # Binary closing of elements in canny edge detection
        elevation_map = nd.binary_closing(elevation_map, s, iterations=2)
    if use_longest_elements == True:
        elevation_map = nd.binary_closing(elevation_map, s, iterations=iterations)
        label_im, nb_labels = nd.label(elevation_map, s)

        sizes = nd.sum(image_array, label_im, range(1, nb_labels + 1))
        ordered = np.sort(sizes, axis=0)
        choicelist = []
        # Selects the 8 longest elements, binary closes to form longer elements and uses that as mask.
        # If there are fewer than 8 elements it uses 4 or 2 elements
        if len(ordered) >= 8:
            max_label = np.where(sizes >= ordered[-8])[0] + 1
            for a in max_label:
                choicelist.append(np.asarray(label_im == a))
            a1 = np.logical_or(choicelist[0], choicelist[1])
            a2 = np.logical_or(choicelist[2], choicelist[3])
            a3 = np.logical_or(choicelist[4], choicelist[5])
            a4 = np.logical_or(choicelist[6], choicelist[7])
            a5 = np.logical_or(a1, a2)
            a6 = np.logical_or(a3, a4)
            output = np.logical_or(a5, a6)
        elif 4 <= len(ordered) <= 7:
            max_label = np.where(sizes >= ordered[-4])[0] + 1
            for a in max_label:
                choicelist.append(np.asarray(label_im == a))
            a1 = np.logical_or(choicelist[0], choicelist[1])
            a2 = np.logical_or(choicelist[2], choicelist[3])
            output = np.logical_or(a1, a2)
        elif len(ordered) < 4:
            try:
                max_label = np.where(sizes >= ordered[-2])[0] + 1
                for a in max_label:
                    choicelist.append(np.asarray(label_im == a))
                output = np.logical_or(choicelist[0], choicelist[1])
            except:
                output = elevation_map
        else:
            output = elevation_map
        elevation_map = nd.binary_closing(output, s, iterations=2)
    # Convert canny binary array to use as mask in segmentation
    elevation_map = elevation_map.astype(int)
    # Watershed function floods regions from markers
    segmentation = skimage.segmentation.watershed(
        evened,
        get_markers(
            image_array,
            evened_selem_size=markers_meaned_selem_size,
            markers_contrast_times=markers_contrast_times,
            markers_sd=markers_sd,
        ),
        mask=elevation_map,
        connectivity=1,
    )
    # Fill smaller holes within regions - usually due to noise; Label each region with different value
    segmentation = nd.binary_fill_holes(segmentation)
    label = sme.label(segmentation)
    if return_mask == True:
        return elevation_map
    else:
        return label


# Measures labelled regions of images which correspond to grains
# and uses selctive techniques to define whether each grain is an actual grain
def measure_props(
    path: Path,
    sigma: float = 2,
    canny_lt: float = 25,
    canny_ht: float = 5,
    dilated_contrast_times=8,
    evened_contrast_times=1,
    dilated_evened_selem_size: int = 9,
    evened_selem_size: int = 4,
    dilation_selem_size: int = 8,
    markers_meaned_selem_size: int = 4,
    markers_contrast_times: float = 30,
    markers_sd: float = 0.25,
    use_longest_elements: bool = False,
    iterations=1,
):
    # Image opened, converted to greyscale and dimensions measured for
    # calculating image:grain ratios, which allows for different size images
    img = Image.open(path).convert("L")
    w, h = img.size
    area = w * h
    tempdf = pd.DataFrame()
    a = segment(
        img,
        sigma=sigma,
        canny_lt=canny_lt,
        canny_ht=canny_ht,
        dilated_contrast_times=dilated_contrast_times,
        evened_contrast_times=evened_contrast_times,
        dilated_evened_selem_size=dilated_evened_selem_size,
        evened_selem_size=evened_selem_size,
        dilation_selem_size=dilation_selem_size,
        markers_meaned_selem_size=markers_meaned_selem_size,
        markers_contrast_times=markers_contrast_times,
        markers_sd=markers_sd,
        use_longest_elements=use_longest_elements,
        iterations=iterations,
    )
    for region in sme.regionprops(a):
        # Filter out regions which take up less than 1% of the image or if the whole image has been detected,
        # and uses a circularity factor to removes erroneous regions which have been detected
        if 1 > (region.area / area) > 0.01:
            # and (region.perimeter ** 2) / (4 * np.pi * region.area) < 3.5:
            # Get region bounding box, apply a mask to grayscaled image so only region
            # with grain in it has GLCM functions applied to it as all other values are 0
            uint_img: np.ndarray = imread(path, as_gray=True)
            uint_img = img_as_ubyte(uint_img)
            coord: t.Tuple[int, int, int, int] = region.bbox
            b = np.where(a, uint_img, 0)
            boxx = b[coord[0] : coord[2], coord[1] : coord[3]]
            # GLCM measures the differences of each pixel to its neighbouring pixels.
            # Sets inputs for GLCM relating to how each value is measured.
            # Distances = nth neighbour;angles = which directions (all)
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
            # Creates GLCM and measures properties thereof, appends to DataFrame
            glcm = sft.greycomatrix(boxx, distances, angles, levels=256, normed=True)
            correlation = sft.greycoprops(glcm, prop="correlation")[0].mean()
            contrast = sft.greycoprops(glcm, prop="contrast")[0].mean()
            dissimilarity = sft.greycoprops(glcm, prop="dissimilarity")[0].mean()
            homogeneity = sft.greycoprops(glcm, prop="homogeneity")[0].mean()
            ASM = sft.greycoprops(glcm, prop="ASM")[0].mean()
            energy = sft.greycoprops(glcm, prop="energy")[0].mean()
            entropy = sme.shannon_entropy(boxx)
            prediction_model = pickle.load(
                open("SVM_0.0044445000000000005_model.sav", "rb")
            )
            values = [
                h,
                region.area,
                region.perimeter,
                region.solidity,
                region.major_axis_length,
                region.minor_axis_length,
                region.equivalent_diameter,
                region.eccentricity,
                region.extent,
                region.filled_area,
                region.euler_number,
                entropy,
                contrast,
                dissimilarity,
                homogeneity,
                ASM,
                energy,
                correlation,
                region.bbox,
            ]
            series = pd.Series(
                values,
                name=path.stem,
                index=[
                    "img_h",
                    "Area",
                    "Perimeter",
                    "Solidity",
                    "Major_axis_length",
                    "Minor_axis_length",
                    "Equivalent_diameter",
                    "Eccentricity",
                    "Extent",
                    "Filled_area",
                    "Euler_number",
                    "Entropy",
                    "Contrast",
                    "Dissimilarity",
                    "Homogeneity",
                    "ASM",
                    "Energy",
                    "Correlation",
                    "Bounding_box",
                ],
            )
            # print(values)
            value = prediction_model.predict(
                np.array(
                    [
                        region.area,
                        region.eccentricity,
                        region.equivalent_diameter,
                        region.extent,
                        region.major_axis_length,
                        region.minor_axis_length,
                        region.perimeter,
                        ASM,
                        contrast,
                        correlation,
                        dissimilarity,
                        energy,
                        entropy,
                        homogeneity,
                        region.solidity,
                    ]
                ).reshape(1, -1)
            )
            # print(value)
            if value == 0:
                tempdf = tempdf.append(series)
            else:
                continue
        else:
            continue
    if tempdf.empty:
        return tempdf
    # elif tempdf["Area"].size > 2:
    #   tempdf = tempdf[(np.abs(stats.zscore(tempdf["Area"])) < 1.5)]
    #  tempdf["thresh_factor"] = tempdf["Area"].min() / tempdf["Area"].max()
    # tempdf = tempdf.loc[tempdf["thresh_factor"] > 0.3].drop("thresh_factor", axis=1)
    return tempdf


# Primary function for segmentation and measuring properties of grains.
# Runs measure_props, but if no grain returned, runs again but selects longest elements for
# mask rather than just canny filter. Works well for images with large grains.
def get_props(
    path: Path,
    sigma=1,
    canny_lt=75,
    canny_ht=5,
    dilated_contrast_times=8,
    evened_contrast_times=1,
    dilated_evened_selem_size=9,
    evened_selem_size=9,
    dilation_selem_size=9,
    markers_meaned_selem_size=3,
    markers_contrast_times=30,
    markers_sd=0.25,
    iterations=1,
):
    a = measure_props(path)
    if a.empty:
        return measure_props(
            path,
            sigma=sigma,
            canny_lt=canny_lt,
            canny_ht=canny_ht,
            dilated_contrast_times=dilated_contrast_times,
            evened_contrast_times=evened_contrast_times,
            dilated_evened_selem_size=dilated_evened_selem_size,
            evened_selem_size=evened_selem_size,
            dilation_selem_size=dilation_selem_size,
            markers_meaned_selem_size=markers_meaned_selem_size,
            markers_contrast_times=markers_contrast_times,
            markers_sd=markers_sd,
            use_longest_elements=True,
            iterations=iterations,
        )
    else:
        return a


# Function to show transformations at each stage of measure_props for each image in dataset
def plot_analysis_on_images(inputfolder: Path):
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
            "Bounding_box",
        ]
    )
    for i, path in enumerate(inputfolder.glob("*.jpg")):
        uint_img = imread(path, as_gray=True)
        output = get_props(path)
        measuresdf = measuresdf.append(output)
        # Matplotlib configurations to show 5 images of different stages of the process
        inp = Image.open(path).convert("L")
        plt.fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
            nrows=1, ncols=5, figsize=(10, 15)
        )
        ax1.imshow(inp, cmap="gray")
        ax1.axis("off")
        ax1.set_title("Raw Image")
        ax2.imshow(
            inc_contrast(
                inp,
            ),
            cmap="gray",
        )
        ax2.axis("off")
        ax2.set_title("Contrasted Image")
        ax3.imshow(
            get_markers(
                inp,
            ),
            cmap="gray",
        )
        ax3.axis("off")
        ax3.set_title("Markers")
        ax4.imshow(
            segment(
                inp,
                return_mask=True,
                use_longest_elements=False,
            ),
            cmap="gray",
        )
        ax4.axis("off")
        ax4.set_title("Mask")
        ax5.imshow(
            segment(
                inp,
                use_longest_elements=False,
            ),
            cmap="gray",
        )
        ax5.axis("off")
        ax5.set_title("Labelled")
        plt.show(plt.fig)
        # Bounding boxes are displayed seperately to observe the specific grains identified by the program
        try:
            cols = 2
            fig = plt.figure(figsize=(20, 2), constrained_layout=True)
            gs = fig.add_gridspec(1, output["Bounding_box"].size)
            for i, _ in enumerate(output["Bounding_box"]):
                b = uint_img[
                    output["Bounding_box"][i - 1][0] : output["Bounding_box"][i - 1][2],
                    output["Bounding_box"][i - 1][1] : output["Bounding_box"][i - 1][3],
                ]
                cols += 1
                fig.add_subplot(gs[0, i:], title="BBox" + f"{i}").axis("off")
                plt.imshow(b)
            plt.show()
        except:
            continue
        pd.set_option("display.max_columns", 500)
        # For use in IPython notebooks
        display(output)


# Optimises combination of input paramaters in terms to get the highest efficacy of positive grain detection
def optimise_segmentation_variables(inputfolder, test_mask_thresholding=False):
    # DataFrame containing magnifications, size in micrometers, and actual number of grains in image
    actualcount = pd.read_csv(
        "true_values.csv",
        names=["Image", "count", "size", "mag"],
        index_col="Image",
    )
    actualcountdf = pd.DataFrame(actualcount)
    # Each input variable is run through a range of values
    for path in inputfolder.glob("*.jpg"):
        size_efficacydf = pd.DataFrame(columns=["accuracy", "SD_diff"])
        for sigma in np.linspace(1, 4, num=12):
            for canny_lt in np.linspace(20, 70, num=10):
                for canny_ht in np.linspace(1, 21, num=5):
                    evened_contrast_times = 1
                    for dilated_contrast_times in np.linspace(7, 9, num=3):
                        dilated_evened_selem_size = 9
                        for dilation_selem_size in np.linspace(4, 10, num=4):
                            for iterations in np.linspace(1, 3, num=3):
                                output = get_props(
                                    path,
                                    sigma=sigma,
                                    canny_lt=canny_lt,
                                    canny_ht=canny_ht,
                                    dilated_contrast_times=dilated_contrast_times,
                                    evened_contrast_times=evened_contrast_times,
                                    dilated_evened_selem_size=dilated_evened_selem_size,
                                    dilation_selem_size=dilation_selem_size,
                                    iterations=iterations,
                                )
                                if output.empty:
                                    sd_diff_40x = np.nan
                                    sd_diff_100x = np.nan
                                    size_series = pd.Series(
                                        [
                                            sd_diff_40x,
                                            sd_diff_100x,
                                        ],
                                        index=["accuracy", "SD_diff"],
                                        name=(
                                            f"{sigma},{canny_lt},{canny_ht},{evened_contrast_times},{dilated_contrast_times},{dilated_evened_selem_size},{dilation_selem_size}"
                                        ),
                                    )
                                    size_efficacydf = size_efficacydf.append(
                                        size_series
                                    )
                                # The size, magnification and actual cumber of grains in the image are input manually.
                                # Double validation - must match the size and pass filters to be considered a grain,
                                # then count calculated as percentage of actual grains in the image
                                else:
                                    if actualcountdf["mag"][f"{path.stem}"] == "40x":
                                        xlist = []
                                        for a in output["Major_axis_length"]:
                                            sd_diff_40x = (
                                                (
                                                    (
                                                        a
                                                        / int(
                                                            actualcountdf["size"][
                                                                f"{path.stem}"
                                                            ]
                                                        )
                                                    )
                                                    - 0.686419
                                                )
                                            ) / 0.273305
                                            if abs(sd_diff_40x) <= 2:
                                                xlist.append(sd_diff_40x)
                                    if actualcountdf["mag"][f"{path.stem}"] == "100x":

                                        xlist = []
                                        for a in output["Major_axis_length"]:
                                            sd_diff_100x = (
                                                (
                                                    (
                                                        (a)
                                                        / int(
                                                            actualcountdf["size"][
                                                                f"{path.stem}"
                                                            ]
                                                        )
                                                    )
                                                    - 2.703794
                                                )
                                            ) / 0.371814
                                            if abs(sd_diff_100x) <= 2:
                                                xlist.append(sd_diff_100x)
                                    a = (len(xlist) * 100) / (
                                        int(actualcountdf["count"][f"{path.stem}"])
                                    )
                                    size_series = pd.Series(
                                        [a, xlist],
                                        index=["accuracy", "SD_diff"],
                                        name=(
                                            f"{sigma},{canny_lt},{canny_ht},{evened_contrast_times},{dilated_contrast_times},{dilated_evened_selem_size},{dilation_selem_size}"
                                        ),
                                    )
                                    # Appended to full DataFrame containing % efficacy for each grain,
                                    # with each index containing the combination of input variables
                                    size_efficacydf = size_efficacydf.append(
                                        size_series
                                    )
        size_efficacydf.to_csv(f"size_{path.stem}.csv")


if __name__ == "__main__":
    # Global variables
    inputfolder = Path("Saps excuded/Included SAPS")
    plot_analysis_on_images(inputfolder)