"""
Module for comparisons - analogous to the original CIUSuite compare, but for data with Gaussian fitting
previously accomplished.
author: DP
date: 11/14/2017
"""
import math
from Gaussian_Fitting import gaussfunc
import scipy.integrate
import numpy as np
import tkinter
from tkinter import filedialog
import os
import pickle
import matplotlib.pyplot as plt

width_compare = True
height_compare = True


def get_nearest_centroid(centroid, centroid_list):
    """
    Find the nearest point in the centroid list to the passed centroid. Removes that point from the centroid_list
    and returns the point found, its index before shortening, and the shortened list
    :param centroid: reference point to compare to (find nearest of)
    :param centroid_list: list of points from which to find the nearest to the reference
    :return: nearest centroid, index of that centroid in the original list, centroid_list with the point removed
    """
    abs_distance = float('inf')
    return_index = 0
    index = 0
    while index < len(centroid_list):
        current_dist = abs(centroid - centroid_list[index])
        if current_dist < abs_distance:
            abs_distance = current_dist
            return_index = index
        index += 1

    # return the found point and remove it from the centroid_list
    try:
        found_point = centroid_list[return_index]
        del centroid_list[return_index]
        return found_point, return_index, centroid_list
    except IndexError:
        # nothing remains in the list - return None
        return None, None, None


def compute_rmsd(filt_gaussians1, filt_gaussians2, dt_axis, cv_index, ciu1_index, ciu2_index, compare_widths, compare_heights):
    """
    Module to compute RMSD difference score between two gaussians by various methods.
    :param filt_gaussians1: CIU_obj1.gauss_filt_params - the list of filtered gaussian parameters from the first CIU
    :param filt_gaussians2: CIU_obj2.gauss_filt_params - the list of filtered gaussian parameters from the second CIU
    :param dt_axis: drift time axis for gaussian shared area computation
    :param cv_index: the collision voltage index at which to access the gaussians from the filtered lists
    :param ciu1_index: index of the peak in the collision voltage specific sublist for CIU #1
    :param ciu2_index: index of the relevant peak to compare in the collision voltage specific sublist for CIU #2
    :param compare_widths: boolean - whether to use the peak widths for comparison or not
    :param compare_heights: boolean - whether to use peak amplitudes (heights) for comparison or not
    :return: root mean square deviation between the two points given the specified comparison method
    """
    centroids1 = filt_gaussians1[cv_index][2::4]
    centroids2 = filt_gaussians2[cv_index][2::4]

    if not compare_widths and not compare_heights:
        # simplest scoring: return only the RMS distance between centroids 1 and 2
        rms = math.sqrt((centroids1[ciu1_index] - centroids2[ciu2_index])**2)
        return rms

    elif compare_widths and not compare_heights:
        # do a difference score analysis assuming that the peak heights are identical
        gauss1_params = filt_gaussians1[cv_index][ciu1_index:ciu1_index + 4]
        gauss2_params = filt_gaussians2[cv_index][ciu2_index:ciu2_index + 4]
        # set both amplitudes to be 1
        gauss1_params[1] = 1
        gauss2_params[1] = 1
        # zero the baseline
        gauss1_params[0] = 0
        gauss2_params[0] = 0

        # construct the specified gaussians to compute a shared area score
        return shared_area_score(dt_axis, gauss1_params, gauss2_params)

    else:
        # do a difference score analysis using the actual peak amplitudes
        gauss1_params = filt_gaussians1[cv_index][ciu1_index:ciu1_index + 4]
        gauss2_params = filt_gaussians2[cv_index][ciu2_index:ciu2_index + 4]
        # zero the baseline
        gauss1_params[0] = 0
        gauss2_params[0] = 0

        # construct the specified gaussians to compute a shared area score
        return shared_area_score(dt_axis, gauss1_params, gauss2_params)


def shared_area_score(dt_axis, gauss1_params, gauss2_params):
    """
    Compute a "shared area score" (shared area normalized against the area of the smaller peak being compared)
    and return it.
    :param dt_axis: the DT axis (x-axis) on which to plot the gaussian functions
    :param gauss1_params: the parameters describing gaussian 1 [baseline, amplitude, centroid, width]
    :param gauss2_params: the parameters describing gaussian 2 [baseline, amplitude, centroid, width]
    :return: shared area score (float)
    """
    # shared area is the area under the lower curve
    gauss1 = gaussfunc(dt_axis, *gauss1_params)
    gauss2 = gaussfunc(dt_axis, *gauss2_params)
    shared_area_arr = []

    # for each point along the x (DT) axis, determine the amount of shared area
    for index in np.arange(0, len(dt_axis)):
        if gauss1[index] > gauss2[index]:
            shared_area_arr.append(gauss2[index])
        elif gauss1[index] < gauss2[index]:
            shared_area_arr.append(gauss1[index])
        elif gauss1[index] == gauss2[index]:
            shared_area_arr.append(0)

    # integrate each peak and the shared area array to get total areas
    area1 = scipy.integrate.trapz(gauss1, dt_axis)
    area2 = scipy.integrate.trapz(gauss2, dt_axis)
    shared_area = scipy.integrate.trapz(shared_area_arr, dt_axis)

    # return the area score
    area_score = (np.min([area1, area2]) - shared_area) / np.min([area1, area2]) * 100
    return area_score


def pairwise_rmsd_centroids(ciu_obj1, ciu_obj2, compare_widths=False, compare_heights=False):
    """
    Perform a pairwise RMSD quantification of difference between two CIU analyses (represented by CIU_Analysis_Obj
    objects with Gaussian fitting previously performed). At each voltage, examines the centroids present in the
    FILTERED params of the CIU object (allowing for previous removal of bogus points).
        - If each CIU obj has a single centroid, the root mean squared distance between them is calculated
        - If one or more objects as multiple centroids, centroids are assigned to their nearest corresponding
        centroid in the other object (i.e. the one that generates the lowest RMS value) until all centroids from
        the object with fewer centroids have been assigned. If the number of centroids is not equal, the unpaired
        centroids will be IGNORED and not impact the difference analysis (as many CIU datasets contain noise peaks
        that may be fit but not correspond to a relevant feature).
    :param ciu_obj1: CIU_Analysis_Obj object with Gaussian fitting previously performed
    :param ciu_obj2: CIU_Analysis_Obj object with Gaussian fitting previously performed
    :param compare_widths: whether to use the width of each peak in comparisons
    :param compare_heights: whether to use the height (amplitude) of each peak in comparisons
    :return: total RMSD average, list of RMSDs by voltage
    """
    filtered_gaussians1 = ciu_obj1.gauss_filt_params
    filtered_gaussians2 = ciu_obj2.gauss_filt_params
    centroid_lists1 = [x[2::4] for x in filtered_gaussians1]
    centroid_lists2 = [x[2::4] for x in filtered_gaussians2]

    cv_axis = ciu_obj1.axes[1]
    rmsd_lists = []

    # for each column (CV) in the data, compute the RMSD between the fingerprints and save to rmsd_list
    index = 0
    for cv in cv_axis:
        # comparison order does not matter, so order the lists so that the one with more entries is #2
        if len(centroid_lists1[index]) <= len(centroid_lists2[index]):
            centroids1 = [x for x in centroid_lists1[index]]
            centroids2 = [x for x in centroid_lists2[index]]
            flip_order = False
        else:
            centroids1 = [x for x in centroid_lists2[index]]
            centroids2 = [x for x in centroid_lists1[index]]
            flip_order = True

        local_rmsds = []
        cent_index = 0
        for centroid in centroids1:
            # find nearest centroid remaining in centroids2 and remove it for future searches
            matched_centroid, matched_index, centroids2 = get_nearest_centroid(centroid, centroids2)
            if matched_centroid is not None:
                # compute RMSD and append to list - keeping track of whether we've flipped the comparison order
                if not flip_order:
                    matched_index = centroid_lists2[index].index(matched_centroid)
                    rmsd = compute_rmsd(filtered_gaussians1, filtered_gaussians2, ciu_obj1.axes[0], index, cent_index,
                                        matched_index, compare_widths, compare_heights)
                else:
                    matched_index = centroid_lists1[index].index(matched_centroid)
                    rmsd = compute_rmsd(filtered_gaussians2, filtered_gaussians1, ciu_obj1.axes[0], index, cent_index,
                                        matched_index, compare_widths, compare_heights)
                local_rmsds.append(rmsd)
            cent_index += 1

        # once all rmsds have been computed, append to overall list
        rmsd_lists.append(local_rmsds)
        index += 1
    return rmsd_lists


def plot_comparisons_by_cv(cv_axis, score_list, average_score, outputpath, filenames):
    """
    Plot comparisons vs CV for the provided score list. If multiple comparisons are present at a given voltage,
    plots all comparisons.
    :param cv_axis: List of collision voltages to become the x-axis
    :param score_list: List of lists of comparisons at each voltage to be plotted on y-axis
    :param outputpath: directory in which to save plot
    :param filenames: list of [filename1, filename2]
    :param average_score: combined average score to note on plot
    :return: void
    """
    plt.clf()
    for x, y in zip(cv_axis, score_list):
        plt.scatter([x] * len(y), y)
    plt.xlabel('Trap CV')
    plt.ylabel('Comparison score: RMS or shared area')
    compare_name = filenames[0] + '-' + filenames[1]
    plt.title(compare_name + '\nAvg score: ' + str(average_score))
    plt.grid('on')
    plt.savefig(os.path.join(outputpath, compare_name + '.png'), dpi=500)
    plt.close()


def compare_main(file1, file2, compare_width, compare_height):
    """
    Primary runner for pairwise comparisons.
    :param file1: first file to compare
    :param file2: second file to compare
    :param compare_width: whether to use the width of each peak in comparisons
    :param compare_height: whether to use the height (amplitude) of each peak in comparisons
    :return: avg_score
    """
    with open(file1, 'rb') as first_file:
        ciu1 = pickle.load(first_file)
    with open(file2, 'rb') as second_file:
        ciu2 = pickle.load(second_file)
    rmsds = pairwise_rmsd_centroids(ciu1, ciu2, compare_width, compare_height)

    # compute overall average score
    flat_scores = [np.average(x) for x in rmsds]
    average_score = np.average(flat_scores)

    # save output
    file_names = [os.path.basename(file1).rstrip('_raw.csv'), os.path.basename(file2).rstrip('_raw.csv')]
    plot_comparisons_by_cv(ciu1.axes[1], rmsds, average_score, file_dir, file_names)
    return average_score


if __name__ == '__main__':
    # testing methods - accessed by loading analysis object stored with pickle
    root = tkinter.Tk()
    root.withdraw()

    files = filedialog.askopenfilenames(filetypes=[('pickled gaussian files', '.pkl')])
    files = list(files)
    file_dir = os.path.dirname(files[0])
    if len(files) == 1:
        # re-open filechooser to get second file
        files.append(filedialog.askopenfilename(filetypes=[('pickled gaussian files', '.pkl')]))

    if len(files) == 2:
        # Read data and compute scores
        compare_main(files[0], files[1], width_compare, height_compare)

    elif len(files) > 2:
        rmsd_print_list = ['File 1, File 2, RMSD (%)']
        # batch compare - compare all against all
        for file in files:
            # don't compare the file against itself
            skip_index = files.index(file)
            file_index = 0
            while file_index < len(files):
                if not file_index == skip_index:
                    # Read data and compute scores
                    avg_score = compare_main(file, files[file_index], width_compare, height_compare)
                    rmsd_print_list.append('{},{},{:.2f}'.format(os.path.basename(file).rstrip('_raw.csv'),
                                                                 os.path.basename(files[file_index]).rstrip('_raw.csv'),
                                                                 avg_score))
                file_index += 1

        # print output to csv
        with open(os.path.join(file_dir, 'batch_RMSDs.csv'), 'w') as rmsd_file:
            for rmsd_string in rmsd_print_list:
                rmsd_file.write(rmsd_string + '\n')
