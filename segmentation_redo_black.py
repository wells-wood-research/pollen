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
from scipy import stats
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

    high = np.max(evened)
    low = np.min(evened)
    std = np.std(evened)
    neatarray = np.array(image_array)
    markers: np.ndarray = np.zeros_like(neatarray)
    # Level reduced/decreased by 1/4 SD
    markers[evened < low + (markers_sd * std)] = 3
    markers[evened > high - (markers_sd * std)] = 2
    return markers


def segment(
    image_array: np.ndarray,
    sigma: float = 1,
    canny_lt: float = 100,
    canny_ht: float = 1,
    evened_contrast_times: float = 10,
    dilated_contrast_times: float = 3,
    dilated_evened_selem_size: int = 2,
    evened_selem_size: int = 10,
    dilation_selem_size: int = 4,
    markers_meaned_selem_size: int = 4,
    markers_contrast_times: float = 15,
    markers_sd: float = 0.25,
    return_mask: bool = False,
    use_longest_elements: bool = False,
    iterations=1,
):

    selem = smo.disk(evened_selem_size)
    evened = sfi.rank.mean_bilateral(
        inc_contrast(image_array, contrast_times=evened_contrast_times), selem
    )
    selem = smo.disk(dilated_evened_selem_size)
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
    if use_longest_elements == True:
        s = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        elevation_map = nd.binary_closing(elevation_map, s, iterations=iterations)
        label_im, nb_labels = nd.label(elevation_map, s)

        sizes = nd.sum(image_array, label_im, range(1, nb_labels + 1))
        ordered = np.sort(sizes, axis=0)

        choicelist = []
        print(len(ordered))
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

    # Convert canny output binary array to use as mask
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
    sigma: float = 3.8,
    canny_lt: float = 25,
    canny_ht: float = 5,
    dilated_contrast_times=8,
    evened_contrast_times=1,
    dilated_evened_selem_size: int = 9,
    evened_selem_size: int = 9,
    dilation_selem_size: int = 4,
    markers_meaned_selem_size: int = 3,
    markers_contrast_times: float = 30,
    markers_sd: float = 0.25,
    use_longest_elements: bool = False,
    iterations=1,
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
        )
    ):
        # Attempt to filter out non-pollen regions using size and circularity factor
        if (
            area > region.area > 280
            and (region.perimeter ** 2) / (4 * np.pi * region.area) < 3
        ):
            no.append(i)
            coins: np.ndarray = imread(path, as_gray=True)
            coins = img_as_ubyte(coins)
            # Get region bounding box and crop image to it so GLCM functions only apply to relevant region
            coord: t.Tuple[int, int, int, int] = region.bbox
            y = int(abs(round((coord[3] - coord[1]) / 4)))
            x = int(abs(round((coord[2] - coord[0]) / 4)))
            boxx = coins[
                (coord[0] + x) : (coord[2] - y), (coord[1] + x) : (coord[3] - y)
            ]

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
            # If function is because some bounding box coordinates were coming back incorrect

            glcm = sft.greycomatrix(boxx, distances, angles, levels=256)
            correlation = sft.greycoprops(glcm, prop="correlation").mean()
            contrast = sft.greycoprops(glcm, prop="contrast").mean()
            dissimilarity = sft.greycoprops(glcm, prop="dissimilarity").mean()
            homogeneity = sft.greycoprops(glcm, prop="homogeneity").mean()
            ASM = sft.greycoprops(glcm, prop="ASM").mean()
            energy = sft.greycoprops(glcm, prop="energy").mean()
            entropy = sme.shannon_entropy(boxx)
            tempdf = tempdf.append(
                pd.Series(
                    [
                        region.area,
                        region.perimeter,
                        region.orientation,
                        region.solidity,
                        region.major_axis_length,
                        region.minor_axis_length,
                        region.equivalent_diameter,
                        region.eccentricity,
                        region.extent,
                        region.inertia_tensor,
                        entropy,
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
                        "Orientation",
                        "Solidity",
                        "Major_axis_length",
                        "Minor_axis_length",
                        "Equivalent_diameter",
                        "Eccentricity",
                        "Extent",
                        "Inertia_tensor",
                        "Entropy",
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
    if tempdf.empty:
        return tempdf
    # elif tempdf["Area"].size > 2:
    #   tempdf = tempdf[(np.abs(stats.zscore(tempdf["Area"])) < 1.5)]
    #  tempdf["thresh_factor"] = tempdf["Area"].min() / tempdf["Area"].max()
    # tempdf = tempdf.loc[tempdf["thresh_factor"] > 0.3].drop("thresh_factor", axis=1)
    return tempdf


def get_props(
    path: Path,
    sigma=1,
    canny_lt=75,
    canny_ht=5,
    dilated_contrast_times=8,
    evened_contrast_times=1,
    dilated_evened_selem_size=9,
    evened_selem_size=9,
    dilation_selem_size=4,
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


def plot_analysis_on_images(
    inputfolder: Path,
    sigma=3.8,
    canny_lt=25,
    canny_ht=5,
    evened_contrast_times=1,
    dilated_contrast_times=8,
    dilated_evened_selem_size=9,
    dilation_selem_size=4,
    evened_selem_size=4,
    markers_meaned_selem_size=4,
    markers_contrast_times=100,
    markers_sd=0.25,
):

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
        coins = imread(path, as_gray=True)
        output = get_props(
            path,
            # sigma=sigma,
            # canny_lt=canny_lt,
            # canny_ht=canny_ht,
            # dilated_contrast_times=dilated_contrast_times,
            # evened_contrast_times=evened_contrast_times,
            # dilated_evened_selem_size=dilated_evened_selem_size,
            # evened_selem_size=evened_selem_size,
            # dilation_selem_size=dilation_selem_size,
            # markers_meaned_selem_size=markers_meaned_selem_size,
            # markers_contrast_times=markers_contrast_times,
            # markers_sd=markers_sd,
        )
        measuresdf = measuresdf.append(output)

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
                contrast_times=evened_contrast_times,
                dilation=False,
                dilation_selem_size=dilation_selem_size,
            ),
            cmap="gray",
        )
        ax2.axis("off")
        ax2.set_title("Contrasted Image")
        ax3.imshow(
            get_markers(
                inp,
                evened_selem_size=evened_selem_size,
                markers_contrast_times=markers_contrast_times,
                markers_sd=markers_sd,
            ),
            cmap="gray",
        )
        ax3.axis("off")
        ax3.set_title("Markers")
        ax4.imshow(
            segment(
                inp,
                sigma=sigma,
                canny_lt=canny_lt,
                canny_ht=canny_ht,
                evened_contrast_times=evened_contrast_times,
                dilated_contrast_times=dilated_contrast_times,
                dilated_evened_selem_size=dilated_evened_selem_size,
                evened_selem_size=evened_selem_size,
                dilation_selem_size=dilation_selem_size,
                markers_meaned_selem_size=markers_meaned_selem_size,
                markers_contrast_times=markers_contrast_times,
                markers_sd=markers_sd,
                return_mask=True,
                use_longest_elements=True,
            ),
            cmap="gray",
        )
        ax4.axis("off")
        ax4.set_title("Mask")
        ax5.imshow(
            segment(
                inp,
                sigma=sigma,
                canny_lt=canny_lt,
                canny_ht=canny_ht,
                evened_selem_size=evened_selem_size,
                evened_contrast_times=evened_contrast_times,
                dilation_selem_size=dilation_selem_size,
                dilated_evened_selem_size=dilated_evened_selem_size,
                dilated_contrast_times=dilated_contrast_times,
                markers_meaned_selem_size=markers_meaned_selem_size,
                markers_contrast_times=markers_contrast_times,
                markers_sd=markers_sd,
                use_longest_elements=True,
            ),
            cmap="gray",
        )
        ax5.axis("off")
        ax5.set_title("Labelled")
        plt.show(plt.fig)
        try:
            cols = 2
            fig = plt.figure(figsize=(20, 2), constrained_layout=True)
            gs = fig.add_gridspec(1, output["Bounding_box"].size)
            for i, _ in enumerate(output["Bounding_box"]):
                b = coins[
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
        display(output)


def optimise_segmentation_variables(inputfolder, test_mask_thresholding=False):
    actualcount = pd.read_csv(
        r"C:/Users/pinto/OneDrive - University of Edinburgh/coding/true_values.csv",
        names=["Image", "count", "size", "mag"],
        index_col="Image",
    )
    actualcountdf = pd.DataFrame(actualcount)

    for path in inputfolder.glob("*.jpg"):
        # count_efficacydf = pd.DataFrame(columns=["diff_percentage"])

        size_efficacydf = pd.DataFrame(columns=["accuracy", "SD_diff"])
        for sigma in np.linspace(1, 3, num=1):
            for canny_lt in np.linspace(100, 125, num=1):
                for canny_ht in np.linspace(5, 55, num=1):
                    for evened_contrast_times in np.linspace(10, 15, num=1):
                        for dilated_contrast_times in np.linspace(3, 6, num=1):
                            for dilated_evened_selem_size in np.linspace(1, 5, num=1):
                                for dilation_selem_size in np.linspace(1, 5, num=1):
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
                                        # if output.empty:
                                        #     diff_percentage = np.nan
                                        # else:
                                        #     detection_diff = abs(
                                        #         output["Area"].size
                                        #         - int(
                                        #             actualcountdf["count"][f"{path.stem}"]
                                        #         )
                                        #     )
                                        #     diff_percentage = (
                                        #         (
                                        #             int(
                                        #                 actualcountdf["count"][
                                        #                     f"{path.stem}"
                                        #                 ]
                                        #             )
                                        #             - detection_diff
                                        #         )
                                        #         / int(
                                        #             actualcountdf["count"][f"{path.stem}"]
                                        #         )
                                        #     ) * 100
                                        # count_series = pd.Series(
                                        #     [diff_percentage],
                                        #     index=["diff_percentage"],
                                        #     name=(
                                        #         f"{sigma},{canny_lt},{canny_ht},{evened_contrast_times},{dilated_contrast_times},{dilated_evened_selem_size},{dilation_selem_size}"
                                        #     ),
                                        # )
                                        # count_efficacydf = count_efficacydf.append(
                                        #     count_series
                                        # )
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
                                        else:
                                            if (
                                                actualcountdf["mag"][f"{path.stem}"]
                                                == "40x"
                                            ):
                                                xlist = []
                                                for a in output["Major_axis_length"]:
                                                    sd_diff_40x = (
                                                        (
                                                            (
                                                                a
                                                                / int(
                                                                    actualcountdf[
                                                                        "size"
                                                                    ][f"{path.stem}"]
                                                                )
                                                            )
                                                            - 0.686419
                                                        )
                                                    ) / 0.273305
                                                    if abs(sd_diff_40x) <= 2:
                                                        xlist.append(sd_diff_40x)
                                            if (
                                                actualcountdf["mag"][f"{path.stem}"]
                                                == "100x"
                                            ):

                                                xlist = []
                                                for a in output["Major_axis_length"]:
                                                    sd_diff_100x = (
                                                        (
                                                            (
                                                                (a)
                                                                / int(
                                                                    actualcountdf[
                                                                        "size"
                                                                    ][f"{path.stem}"]
                                                                )
                                                            )
                                                            - 2.703794
                                                        )
                                                    ) / 0.371814
                                                    if abs(sd_diff_100x) <= 2:
                                                        xlist.append(sd_diff_100x)
                                            a = (len(xlist) * 100) / (
                                                int(
                                                    actualcountdf["count"][
                                                        f"{path.stem}"
                                                    ]
                                                )
                                            )
                                            size_series = pd.Series(
                                                [a, xlist],
                                                index=["accuracy", "SD_diff"],
                                                name=(
                                                    f"{sigma},{canny_lt},{canny_ht},{evened_contrast_times},{dilated_contrast_times},{dilated_evened_selem_size},{dilation_selem_size}"
                                                ),
                                            )
                                            size_efficacydf = size_efficacydf.append(
                                                size_series
                                            )
        # count_efficacydf.to_csv(f"count_{path.stem}.csv")
        size_efficacydf.to_csv(f"size_{path.stem}.csv")


if __name__ == "__main__":
    # Global variables
    inputfolder = Path("Saps excuded/Included SAPS")
    plot_analysis_on_images(inputfolder)