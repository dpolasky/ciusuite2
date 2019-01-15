"""
This file is part of CIUSuite 2
Copyright (C) 2018 Daniel Polasky and Sugyan Dixit

Module for classification schemes for CIU data groups
Authors: Dan Polasky, Sugyan Dixit
Date: 1/11/2018
"""
from Gaussian_Fitting import Gaussian

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import pickle
import os
import itertools
import time
from tkinter import messagebox
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import precision_score
from sklearn.feature_selection import f_classif, GenericUnivariateSelect
from sklearn.svm import SVC

from typing import List
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from CIU_analysis_obj import CIUAnalysisObj
    from CIU_Params import Parameters
    from Feature_Detection import Feature


def main_build_classification(labels, analysis_obj_list_by_label, params_obj, output_dir, known_feats=None):
    """
    Main method for classification. Performs feature selection followed by LDA and classification
    and generates output and plots. Returns a ClassificationScheme object to be saved for future
    classification of unknowns.
    :param labels: list of class labels (strings)
    :param analysis_obj_list_by_label: list of lists of analysis objects, sorted by class
    :type analysis_obj_list_by_label: list[list[CIUAnalysisObj]]
    :param output_dir: directory in which to save plots/output
    :param params_obj: Parameters object with classification parameter information
    :type params_obj: Parameters
    :param known_feats: list of
    :return: ClassificationScheme object with the generated scheme
    :rtype: ClassificationScheme
    """
    # generate a list of lists of labels in the same shape as the analysis object list
    shaped_label_list = []
    for index, label in enumerate(labels):
        shaped_label_list.append([label for _ in range(len(analysis_obj_list_by_label[index]))])

    if known_feats is None:
        # run feature selection and crossvalidation to select best features automatically
        all_features = univariate_feature_selection(shaped_label_list, analysis_obj_list_by_label, params_obj, output_dir)

        # assess all features to determine which to use in the final scheme
        # best_features = all_feature_crossval_lda(all_features, analysis_obj_list_by_label, shaped_label_list, output_dir)
        best_features, crossval_score, all_crossval_data = crossval_main(analysis_obj_list_by_label, labels, output_dir, params_obj, all_features)
    else:
        # Manual mode: use the provided features and run limited crossvalidation
        best_features, crossval_score, all_crossval_data = crossval_main(analysis_obj_list_by_label, labels, output_dir, params_obj, known_feats)
        best_features = known_feats

    # perform LDA and classification on the selected/best features
    constructed_scheme = lda_ufs_best_features(best_features, analysis_obj_list_by_label, shaped_label_list, params_obj, output_dir)
    constructed_scheme.crossval_test_score = crossval_score
    constructed_scheme.all_crossval_data = all_crossval_data

    # plot output here for now, will probably move eventually
    plot_classification_decision_regions(constructed_scheme, params_obj, output_dir)
    return constructed_scheme


def get_classif_data(analysis_obj, params_obj, ufs_mode=False, num_gauss_override=None, selected_cvs=None):
    """
    Initialize a classification data matrix in each analysis object in the lists according to the
    classification mode specified in the parameters object. In All_Data mode, this is simply the
    ciu_data matrix. In Gaussian mode, it will be Gaussian information from the object's fitted Gaussian
    lists.
    :param analysis_obj: analysis objects
    :type analysis_obj: CIUAnalysisObj
    :param params_obj: Parameters object with classification parameter information
    :type params_obj: Parameters
    :param ufs_mode: boolean, True if using for UFS (feature selection), which requires only centroids from gaussian fitting
    :param num_gauss_override: MUST be provided for Unknown data fitting (in Gaussian mode ONLY) - the number of gaussians in the scheme being used
    :param selected_cvs: for unknown analyses, only return the data in the specified CV columns
    :return: classification data matrix
    """
    classif_data = None

    if params_obj.classif_3_unk_mode == 'All_Data':
        classif_data = analysis_obj.ciu_data

    elif params_obj.classif_3_unk_mode == 'Gaussian':
        classif_data = []

        # for unknown data, num gaussians is provided (use it); for building scheme, num gaussians comes from params object (as a convenient save location)
        if num_gauss_override is not None:
            max_num_gaussians = num_gauss_override
        else:
            max_num_gaussians = params_obj.silent_clf_4_num_gauss

        # use Gaussian features if available, otherwise just all filtered Gaussians
        if analysis_obj.features_gaussian is not None:
            if analysis_obj.classif_gaussfeats is None:
                gaussian_list_by_cv = prep_gaussfeats_for_classif(analysis_obj.features_gaussian, analysis_obj)
            else:
                gaussian_list_by_cv = analysis_obj.classif_gaussfeats
        else:
            gaussian_list_by_cv = analysis_obj.raw_protein_gaussians

        if not ufs_mode:
            # assemble matrix of gaussian data
            for gaussian_list in gaussian_list_by_cv:
                # skip any non-selected CVs if requested (i.e. in unknown analysis mode)
                if selected_cvs is not None:
                    if not gaussian_list[0].cv in selected_cvs:
                        continue
                attributes = ['cent', 'width', 'amp']
                num_attributes = len(attributes)
                attribute_list = np.zeros(max_num_gaussians * num_attributes)

                attribute_index = 0
                if len(gaussian_list) == 0:
                    continue
                for gaussian in gaussian_list:
                    attribute_list[attribute_index] = gaussian.centroid
                    attribute_index += 1
                    attribute_list[attribute_index] = gaussian.width
                    attribute_index += 1
                    attribute_list[attribute_index] = gaussian.amplitude
                    attribute_index += 1

                classif_data.append(attribute_list)
            classif_data = np.asarray(classif_data).T
        else:
            # for UFS, only use centroids
            for gaussian_list in gaussian_list_by_cv:
                cent_list = np.zeros(max_num_gaussians)
                for gauss_index, gaussian in enumerate(gaussian_list):
                    cent_list[gauss_index] = gaussian.centroid

                classif_data.append(cent_list)
            classif_data = np.asarray(classif_data).T
    else:
        print('WARNING: INVALID CLASSIFICATION MODE: {}'.format(params_obj.classif_3_unk_mode))

    return classif_data


def prep_gaussfeats_for_classif(features_list, analysis_obj):
    """
    Assemble a Gaussian-list-by-CV list from input features data. Fills any gaps between and within
    features with Gaussian data from the filtered_gaussians list and assembles a complete list of
    Gaussians by CV.
    :param features_list: list of Features
    :type features_list: list[Feature]
    :param analysis_obj: CIUAnalysisObj with gaussian fitting and gaussian feature detect performed
    :type analysis_obj: CIUAnalysisObj
    :return: List of (Gaussian lists) sorted by CV
    :rtype: list[list[Gaussian]]
    """
    # make an empty list for Gaussians at each CV
    final_gaussian_lists = [[] for _ in analysis_obj.axes[1]]

    features_list = close_feature_gaps(features_list, analysis_obj.axes[1])

    # iterate over features, filling any gaps within the feature and entering Gaussians into the final list
    for feature in features_list:
        # determine if the feature contains gaps
        gaussian_cvs = [gaussian.cv for gaussian in feature.gaussians]

        for cv in feature.cvs:
            # append Gaussian(s) at this CV to the final list
            try:
                cv_index = np.where(analysis_obj.axes[1] == cv)[0][0]
            except IndexError:
                # A gaussian had a CV that was not in the analysis object's CV axis! This should be caught elsewhere
                print('Gaussian had CV {}, but that CV is not in the CV axis of this file (after axes were equalized across all files). It will be ignored.'.format(cv))
                continue

            this_cv_gaussian = [x for x in feature.gaussians if x.cv == cv]
            final_gaussian_lists[cv_index].extend(this_cv_gaussian)

            if cv not in gaussian_cvs:
                # a gap is present within this feature- create a Gaussian at median centroid/values to fill it
                new_gaussian = Gaussian(amplitude=np.median([x.amplitude for x in feature.gaussians]),
                                        centroid=feature.gauss_median_centroid,
                                        width=np.median([x.width for x in feature.gaussians]),
                                        collision_voltage=cv,
                                        pcov=None,
                                        protein_bool=True)
                final_gaussian_lists[cv_index].append(new_gaussian)

    # Finally, check if all CVs have been covered by features. If not, add highest amplitude Gaussian from non-feature list
    for cv_index, cv in enumerate(analysis_obj.axes[1]):
        if len(final_gaussian_lists[cv_index]) == 0:
            # no Gaussians have been added here yet, so we need to add one. Get the raw set of Gaussians (before feature detection) fit at this CV
            cv_gaussians_from_obj = analysis_obj.raw_protein_gaussians[cv_index]

            # Find the feature that extends closest to this CV, then find the raw Gaussian at this CV closest to that feature's median centroid
            min_cv_dist = np.inf
            nearest_feat = None
            for feature in features_list:
                for feat_cv in feature.cvs:
                    if abs(feat_cv - cv) < min_cv_dist:
                        nearest_feat = feature
                        min_cv_dist = abs(feat_cv - cv)
            nearest_centroid_index = np.argmin([abs(x.centroid - nearest_feat.gauss_median_centroid) for x in cv_gaussians_from_obj])
            try:
                # add the closest Gaussian to the list at this CV
                final_gaussian_lists[cv_index].append(cv_gaussians_from_obj[nearest_centroid_index])
            except IndexError:
                # no Gaussians found at this CV in the original fitting - leave empty
                continue

    analysis_obj.classif_gaussfeats = final_gaussian_lists
    return final_gaussian_lists


def close_feature_gaps(features_list, cv_axis):
    """
    Check all features for gaps in their CV lists, and fill in the gaps if any exist by inserting
    appropriate CV values
    :param features_list: list of Features
    :type features_list list[Feature]
    :param cv_axis: analysis_obj.axes[1]
    :return: updated features list with gaps closed (in feature.cvs ONLY)
    :rtype: list[Feature]
    """
    cv_step = cv_axis[1] - cv_axis[0]

    for feature in features_list:
        for index, current_cv in enumerate(feature.cvs):
            try:
                next_cv = feature.cvs[index + 1]
            except IndexError:
                # reached the end, ignore
                continue
            while not (next_cv - current_cv) == cv_step:
                # a gap is present - insert the next value to fill it
                correct_next = current_cv + cv_step
                feature.cvs.insert(index + 1, correct_next)
                next_cv = feature.cvs[index + 1]
    return features_list


def univariate_feature_selection(shaped_label_list, analysis_obj_list_by_label, params_obj, output_path):
    """
    Perform feature selection on the provided data.
    :param shaped_label_list: list of lists of class labels with the same shape as the analysis object list
    :param analysis_obj_list_by_label: list of lists of analysis objects, sorted by class
    :type analysis_obj_list_by_label: list[list[CIUAnalysisObj]]
    :param params_obj: parameters information
    :type params_obj: Parameters
    :param output_path: directory in which to save plot
    :return: list of all features, sorted in decreasing order of score
    :rtype: list[CFeature]
    """
    cv_axis = analysis_obj_list_by_label[0][0].axes[1]

    # generate all combinations of replicate datasets within the labels
    products = generate_products_for_ufs(analysis_obj_list_by_label, shaped_label_list, params_obj)

    # Create a CFeature object to hold the information for this CV (feature)
    scores = [product.fit_sc for product in products]
    mean_score = np.mean(scores, axis=0)
    std_score = np.std(scores, axis=0)

    features = []
    for cv_index, cv in enumerate(cv_axis):
        feature = CFeature(cv, cv_index, mean_score[cv_index], std_score[cv_index])
        features.append(feature)

    # sort feature scores either by mean - stdev ("error mode") or just mean alone.
    if params_obj.classif_6_ufs_use_error_mode:
        sorted_features = sorted(features, key=lambda x: (x.mean_score - x.std_dev_score), reverse=True)
    else:
        sorted_features = sorted(features, key=lambda x: x.mean_score, reverse=True)

    unique_labels = get_unique_labels([x for label_list in shaped_label_list for x in label_list])
    scheme_name = '_'.join(unique_labels)
    plot_feature_scores(sorted_features, params_obj, scheme_name, output_path)
    return sorted_features


def generate_products_for_ufs(analysis_obj_list_by_label, shaped_label_list, params_obj):
    """
    Generate all combinations of replicate data across classes for feature selection. Will
    create a DataProduct object with the key information for each combination.
    :param analysis_obj_list_by_label: list of lists of CIUAnalysisObj's, sorted by class label
    :type analysis_obj_list_by_label: list[list[CIUAnalysisObj]]
    :param shaped_label_list: list of lists of class labels with matching shape of analysis_obj_by_label
    :param params_obj: parameter info
    :type params_obj: Parameters
    :return: list of DataProduct objects for each combination
    :rtype: list[UFSResult]
    """
    products = []
    for object_tuple, label_tuple in zip(itertools.product(*analysis_obj_list_by_label), itertools.product(*shaped_label_list)):
        # create a UFSResult object for this combination
        # data_list = [x.ciu_data for x in object_tuple]
        data_list = [get_classif_data(x, params_obj, ufs_mode=True) for x in object_tuple]

        label_list = [x for x in label_tuple]
        product = UFSResult(data_list, label_list)

        # Run feature selection for this combination
        select = GenericUnivariateSelect(score_func=f_classif, mode='percentile', param=100)
        select.fit(product.combined_data, product.numeric_label_arr)

        product.fit_pvalues = select.pvalues_
        product.fit_scores = select.scores_
        product.fit_sc = -np.log10(select.pvalues_)

        products.append(product)
    return products


def crossval_main(analysis_obj_list_by_label, labels, outputdir, params_obj, features_list):
    """

    :param analysis_obj_list_by_label:
    :param labels:
    :param outputdir:
    :param params_obj:
    :type params_obj: Parameters
    :param features_list: List of CFeatures, sorted in decreasing order of score
    :type features_list: list[CFeature]
    :return: list of selected features, test score for that # features, and all cross validation data
    """
    # training_size = params_obj.classif_1_training_size
    # determine training size as size of the smallest class - 1 (1 test file at a time)
    min_class_size = np.min([len(x) for x in analysis_obj_list_by_label])
    training_size = min_class_size - 1

    current_features_list = []
    train_score_means = []
    train_score_stds = []
    test_score_means = []
    test_score_stds = []
    results = []

    time_start = time.time()
    # optional max number of features to consider
    if params_obj.classif_7_max_feats_for_crossval > 0:
        max_features = params_obj.classif_7_max_feats_for_crossval
        if max_features > len(features_list):
            max_features = len(features_list) + 1
    else:
        max_features = len(features_list) + 1

    for ind, feature in enumerate(features_list[:max_features]):
        print('\nPerforming cross validation for {} features'.format(ind+1))
        current_features_list.append(feature)

        # Generate all combinations
        crossval_obj = CrossValProduct(analysis_obj_list_by_label, labels, training_size, current_features_list)
        crossval_combos = crossval_obj.assemble_class_combinations(params_obj)
        result = assemble_products(crossval_combos)
        results.append(result)

    for result in results:
        # get scores and plot and stuff
        train_score_means.append(result[0])
        train_score_stds.append(result[1])
        test_score_means.append(result[2])
        test_score_stds.append(result[3])
    print('classification done in {:.2f}'.format(time.time() - time_start))

    train_score_means = np.array(train_score_means)
    train_score_stds = np.array(train_score_stds)
    test_score_means = np.array(test_score_means)
    test_score_stds = np.array(test_score_stds)

    # save and plot crossvalidation score information
    crossval_data = (train_score_means, train_score_stds, test_score_means, test_score_stds)
    unique_labels = get_unique_labels(labels)
    scheme_name = '_'.join(unique_labels)
    save_crossval_score(crossval_data, scheme_name, outputdir)
    plot_crossval_scores(crossval_data, scheme_name, params_obj, outputdir)

    # determine best features list from crossval scores
    best_num_feats, best_score = peak_crossval_score_detect(test_score_means, params_obj.classif_2_score_dif_tol)
    output_features = features_list[0: best_num_feats]
    return output_features, best_score, crossval_data


def assemble_products(all_class_combination_lists):
    """
    Assemble training and test (validation) datasets from each combination of classes using
    a leave-one-out validation method, then run the classification (LDA/SVM) to generate
    final scores.
    :param all_class_combination_lists: List of all class combination
    Assemble the products of all class combinations (DataCombination objects)
    :return: training and test score mean and standard deviation lists
    """
    probs = []
    train_scores, test_scores = [], []
    for combo_tuple in itertools.product(*all_class_combination_lists):
        # stack training and test data and labels
        stacked_train_data, stacked_train_labels = [], []
        stacked_test_data, stacked_test_labels = [], []
        for combo in combo_tuple:
            stacked_train_data.append(combo.training_data_final)
            stacked_train_labels.append(combo.training_labels_string)
            stacked_test_data.append(combo.test_data_final)
            stacked_test_labels.append(combo.test_labels_string)

        final_train_data, final_train_labels = [], []
        for rep_index, replicate_data_list in enumerate(stacked_train_data):
            for feature_data in replicate_data_list:
                final_train_data.append(feature_data)
            final_train_labels = np.concatenate((final_train_labels, stacked_train_labels[rep_index]))
        final_train_data = np.asarray(final_train_data)
        final_train_labels = np.asarray(final_train_labels)

        final_test_data, final_test_labels = [], []
        for rep_index, replicate_data_list in enumerate(stacked_test_data):
            for feature_data in replicate_data_list:
                final_test_data.append(feature_data)
            final_test_labels = np.concatenate((final_test_labels, stacked_test_labels[rep_index]))
        final_test_data = np.asarray(final_test_data)
        final_test_labels = np.asarray(final_test_labels)

        # replacing 'stacked' with 'final' in all lines below
        enc = LabelEncoder()
        enc.fit(final_train_labels)

        numeric_label_train = enc.transform(final_train_labels) + 1
        numeric_label_test = enc.transform(final_test_labels) + 1

        train_score, test_score = lda_for_crossval(final_train_data, numeric_label_train,
                                                   final_test_data, numeric_label_test)
        train_scores.append(train_score)
        test_scores.append(test_score)

    train_scores_mean = np.mean(train_scores)
    train_scores_std = np.std(train_scores)
    test_scores_mean = np.mean(test_scores)
    test_scores_std = np.std(test_scores)
    probs_mean = np.mean(probs)
    probs_std = np.std(probs)
    return train_scores_mean, train_scores_std, test_scores_mean, test_scores_std, probs_mean, probs_std


def peak_crossval_score_detect(test_score_means, diff_from_max):
    """
    Determine the best set of features based on crossvalidation testing scores generated in
    crossval_main. Chooses the first 'peak' (point after which score decreases) that is within
    tolerance of the overall maximum score in the data. This is to choose the minimum number of
    features while still achieving a high score.
    :param test_score_means: list of scores for cross validation test data, in increasing order of number of features
    :param diff_from_max: maximum distance below the max value of test_score_means that a peak is allowed for selection. Default 0.05
    :return: best number of features (index of test_score_means) and score (value)
    """
    max_score = np.max(test_score_means)

    for index, value in enumerate(test_score_means):
        try:
            if test_score_means[index + 1] < value:
                # stop here (reached a peak) if within tolerance of max
                if max_score - value <= diff_from_max:
                    return index + 1, value     # index + 1 because we're determining the NUM of features, indexed from 1 (not 0)
        except IndexError:
            # reached the end of the list - return final index + 1 (because indexing from 1 for num feats) and value
            return index + 1, value


def arrange_data_for_lda(flat_data_matrix_list, flat_label_list, features_list, flat_axes_list, flat_filenames=None):
    """
    Prepare data for LDA by arranging selected CV columns (from the original matrix) into
    the desired shape. Multiple options supported at this time, will likely choose best eventually.
    :param flat_data_matrix_list: list of 2D arrays of data (drift bin OR gaussian info, collision voltage)
    :param flat_label_list: list of class labels corresponding to input data, in same shape as flat_data_matrix
    :param features_list: list of Features (collision voltage indicies) to use
    :type features_list: list[CFeature]
    :param flat_axes_list: list of CV axes for each dataset in flat_data_matrix_list to enable correct indexing
    :param flat_filenames: (optional) list of filenames in same shape as flat label list. If provided, filenames are returned
    :return: assembled raw data list, assembled labels list - ready for LDA as x, y
    """
    # actual CVs used in original scheme - must be the same in all data
    cvfeats_list = [x.cv for x in features_list]

    lda_ciu_data, lda_label_data, lda_filenames, lda_feat_cvs = [], [], [], []

    # loop over each replicate of data provided
    for data_index in range(len(flat_data_matrix_list)):
        # loop over each feature (collision voltage) desired, saving the requested data in appropriate form
        for index, data_cv in enumerate(cvfeats_list):
            # get the correct index of the data_cv in the current raw data matrix
            current_cv_axis = flat_axes_list[data_index]
            this_cv_correct_index = (np.abs(np.asarray(current_cv_axis) - data_cv)).argmin()

            # use the correct index to find the raw data at this CV
            test_mat = flat_data_matrix_list[data_index]
            test_mat_t = test_mat.T
            final = test_mat_t[this_cv_correct_index]
            lda_ciu_data.append(final)
            # lda_ciu_data.append(flat_data_matrix_list[data_index].T[this_cv_correct_index])
            lda_label_data.append(flat_label_list[data_index])
            if flat_filenames is not None:
                lda_filenames.append(flat_filenames[data_index])
            lda_feat_cvs.append(data_cv)

    return lda_ciu_data, lda_label_data, lda_filenames, lda_feat_cvs


def lda_for_crossval(stacked_train_data, stacked_train_labels, stacked_test_data, stacked_test_labels):
    """
    Run a linear discriminant analysis using the provided training and test data. Scores are
    determined by whether test data is successfully classified by the generated LDA/SVM.
    :param stacked_train_data:
    :param stacked_train_labels:
    :param stacked_test_data:
    :param stacked_test_labels:
    :return:
    """
    lda = LinearDiscriminantAnalysis(solver='svd', n_components=5)
    lda.fit(stacked_train_data, stacked_train_labels)
    train_lda = lda.transform(stacked_train_data)
    test_lda = lda.transform(stacked_test_data)
    # max_iter=1000 needed to prevent occasional (and unpredictable) freezes with ridiculous iteration numbers
    svm = SVC(kernel='linear', C=1, probability=True, max_iter=1000)
    # svm = LinearSVC(C=1, max_iter=1000)
    try:
        svm.fit(train_lda, stacked_train_labels)
    except ValueError:
        print('Error in SVM fitting. This should not be reached - check your input data for duplicates')
    train_score = svm.score(train_lda, stacked_train_labels)
    test_score = svm.score(test_lda, stacked_test_labels)

    return train_score, test_score


def lda_ufs_best_features(features_list, analysis_obj_list_by_label, shaped_label_list, param_obj, output_dir):
    """
    Construct final classification scheme using a support vector machine in the optimal linear
    discriminant space for the provided data. Only uses data from the specified features (activation
    voltages).
    :param features_list: list of selected features from feature selection
    :type features_list: list[CFeature]
    :param analysis_obj_list_by_label: list of lists of CIUAnalysisObj's, sorted by class label
    :type analysis_obj_list_by_label: list[list[CIUAnalysisObj]]
    :param shaped_label_list: list of lists of class labels with matching shape of analysis_obj_by_label
    :param param_obj: parameters information
    :type param_obj: Parameters
    :param output_dir: directory in which to save output plot
    :return: generated classification scheme object with LDA and SVC performed
    :rtype: ClassificationScheme
    """
    # flatten input lists (sorted by class label) into a single list
    flat_ciuraw_list = [get_classif_data(x, param_obj) for label_obj_list in analysis_obj_list_by_label for x in label_obj_list]
    flat_label_list = [x for label_list in shaped_label_list for x in label_list]
    flat_filename_list = [x.short_filename for class_list in analysis_obj_list_by_label for x in class_list]
    flat_cv_axes = [x.axes[1] for label_obj_list in analysis_obj_list_by_label for x in label_obj_list]

    # create a concatenated array with the selected CV columns from each raw dataset
    input_x_ciu_data, input_label_data, input_filenames, input_feats = arrange_data_for_lda(flat_ciuraw_list, flat_label_list, features_list, flat_cv_axes, flat_filename_list)

    # finalize input data for LDA
    input_x_ciu_data = np.asarray(input_x_ciu_data)
    input_y_labels, target_label = createtargetarray_featureselect(input_label_data)

    # run LDA
    lda = LinearDiscriminantAnalysis(solver='svd', n_components=5)
    lda.fit(input_x_ciu_data, input_y_labels)
    x_lda = lda.transform(input_x_ciu_data)
    expl_var_r = lda.explained_variance_ratio_

    # build classification scheme
    clf = SVC(kernel='linear', C=1, probability=True, max_iter=1000)
    clf.fit(x_lda, input_y_labels)
    y_pred = clf.predict(x_lda)
    prec_score = precision_score(input_y_labels, y_pred, pos_label=1, average='weighted')
    probs = clf.predict_proba(x_lda)

    # initialize classification scheme object and return it
    scheme = ClassificationScheme()
    scheme.selected_features = features_list
    scheme.classifier = clf
    scheme.classifier_type = 'SVC'
    scheme.classif_prec_score = prec_score
    scheme.lda = lda
    scheme.explained_variance_ratio = expl_var_r
    scheme.numeric_labels = input_y_labels
    scheme.class_labels = target_label
    scheme.unique_labels = get_unique_labels(target_label)
    scheme.transformed_test_data = x_lda
    scheme.test_filenames = input_filenames
    scheme.params = clf.get_params()
    scheme.input_feats = input_feats
    scheme.name = generate_scheme_name(flat_label_list, [])

    # save_lda_output(x_lda, input_filenames, input_feats, scheme.name, output_dir, explained_variance_ratio=expl_var_r)
    x_lda_by_file, y_pred_by_file, probs_by_file, filenames_by_file = [], [], [], []
    num_feats = len(features_list)
    index = 0
    for filename in flat_filename_list:
        x_lda_by_file.append(x_lda[index: index + num_feats])
        y_pred_by_file.append(y_pred[index: index + num_feats])
        probs_by_file.append(probs[index: index + num_feats])
        filenames_by_file.append(filename)
        index += num_feats

    save_lda_and_predictions(scheme, x_lda_by_file, y_pred_by_file, probs_by_file, filenames_by_file, output_dir, unknowns_bool=False)
    plot_probabilities(param_obj, scheme, probs_by_file, output_dir, unknown_bool=False)
    return scheme


def createtargetarray_featureselect(inputlabel):
    """

    :param inputlabel:
    :return:
    """
    string_labels = []
    numeric_labels = []
    unique_labels = []
    class_index = 0
    # encode each class as an integer in the order they are presented from the initial lists
    for input_string in inputlabel:
        if input_string not in string_labels:
            unique_labels.append(input_string)
            class_index += 1
        string_labels.append(input_string)
        numeric_labels.append(unique_labels.index(input_string) + 1)
    return np.asarray(numeric_labels), np.asarray(string_labels)


def get_unique_labels(label_list):
    """
    Return a list of unique labels (i.e. without duplicates) from the provided label list
    :param label_list: list of labels (strings)
    :return: list of unique labels (strings)
    """
    unique_labels = []
    for label in label_list:
        if label not in unique_labels:
            unique_labels.append(label)
    return unique_labels


def plot_feature_scores(feature_list, params_obj, scheme_name, output_path):
    """
    Plot feature score by collision voltage
    :param feature_list: list of CFeatures
    :type feature_list: list[CFeature]
    :param params_obj: Parameters object with plot information
    :type params_obj: Parameters
    :param scheme_name: (string) name of scheme for labeling purposes
    :param output_path: directory in which to save output
    :return: void
    """
    plt.figure(figsize=(params_obj.plot_03_figwidth, params_obj.plot_04_figheight), dpi=params_obj.plot_05_dpi)

    mean_scores = [x.mean_score for x in feature_list]
    std_scores = [x.std_dev_score for x in feature_list]
    cv_axis = [x.cv for x in feature_list]

    plt.errorbar(x=cv_axis, y=mean_scores, yerr=std_scores, ls='none', marker='o', color='black', markersize=params_obj.plot_14_dot_size, markeredgecolor='black')
    plt.axhline(y=0.0, color='black', ls='--')

    # plot titles, labels, and legends
    if params_obj.plot_12_custom_title is not None:
        plot_title = params_obj.plot_12_custom_title
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    elif params_obj.plot_11_show_title:
        plot_title = scheme_name
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    if params_obj.plot_08_show_axes_titles:
        plt.xlabel(params_obj.plot_09_x_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
        plt.ylabel('-Log10(p-value)', fontsize=params_obj.plot_13_font_size, fontweight='bold')
    plt.xticks(fontsize=params_obj.plot_13_font_size)
    plt.yticks(fontsize=params_obj.plot_13_font_size)

    output_name = os.path.join(output_path, scheme_name + '_UFS' + params_obj.plot_02_extension)
    try:
        plt.savefig(output_name)
    except PermissionError:
        messagebox.showerror('Please Close the File Before Saving', 'The file {} is being used by another process! Please close it, THEN press the OK button to retry saving'.format(output_name))
        plt.savefig(output_name)
    plt.close()


def plot_feature_scores_subclass(features_by_subclass, params_obj, scheme_name, output_path):
    """
    Plot feature score by collision voltage
    :param features_by_subclass: list of CFeatures
    :type features_by_subclass: list[list[CFeature]]
    :param params_obj: Parameters object with plot information
    :type params_obj: Parameters
    :param scheme_name: (string) name of scheme for labeling purposes
    :param output_path: directory in which to save output
    :return: void
    """
    plt.figure(figsize=(params_obj.plot_03_figwidth, params_obj.plot_04_figheight), dpi=params_obj.plot_05_dpi)

    colors = ['black', 'blue', 'green', 'orange', 'red', 'purple', 'yellow', 'pink', 'lightblue']
    for index, feature_list in enumerate(features_by_subclass):
        mean_scores = [x.mean_score for x in feature_list]
        std_scores = [x.std_dev_score for x in feature_list]
        cv_axis = [x.cv for x in feature_list]

        try:
            color = colors[index]
        except IndexError:
            color = 'black'
        plt.errorbar(x=cv_axis, y=mean_scores, yerr=std_scores, ls='none', marker='o', color=color, markersize=params_obj.plot_14_dot_size, markeredgecolor='black', alpha=0.8, label=feature_list[0].subclass_label)
        plt.axhline(y=0.0, color='black', ls='--')

    # plot titles, labels, and legends
    if params_obj.plot_07_show_legend:
        plt.legend(loc='best', fontsize=params_obj.plot_13_font_size)
    if params_obj.plot_12_custom_title is not None:
        plot_title = params_obj.plot_12_custom_title
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    elif params_obj.plot_11_show_title:
        plot_title = scheme_name
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    if params_obj.plot_08_show_axes_titles:
        plt.xlabel(params_obj.plot_09_x_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
        plt.ylabel('-Log10(p-value)', fontsize=params_obj.plot_13_font_size, fontweight='bold')
    plt.xticks(fontsize=params_obj.plot_13_font_size)
    plt.yticks(fontsize=params_obj.plot_13_font_size)

    output_name = os.path.join(output_path, scheme_name + '_UFS' + params_obj.plot_02_extension)
    try:
        plt.savefig(output_name)
    except PermissionError:
        messagebox.showerror('Please Close the File Before Saving', 'The file {} is being used by another process! Please close it, THEN press the OK button to retry saving'.format(output_name))
        plt.savefig(output_name)
    plt.close()


def save_crossval_score(crossval_data, scheme_name, outputpath):
    """
    Save crossvalidation data output to file at path provided
    :param crossval_data: tuple of (training means, training stds, test means, test stds) lists
    :param scheme_name: (string) name of scheme for labeling purposes
    :param outputpath: directory in which to save output
    :return: void
    """
    train_score_means = crossval_data[0]
    train_score_stds = crossval_data[1]
    test_score_means = crossval_data[2]
    test_score_stds = crossval_data[3]
    outfilename = os.path.join(outputpath, scheme_name + '_crossval.csv')
    output_string = ''

    lineheader = 'num_feats, train_score_mean, train_score_std, test_score_mean, test_score_std, \n'
    output_string += lineheader
    for ind in range(len(train_score_means)):
        line = '{}, {}, {}, {}, {}, \n'.format(ind+1, train_score_means[ind], train_score_stds[ind],
                                               test_score_means[ind], test_score_stds[ind])
        output_string += line

    try:
        with open(outfilename, 'w') as outfile:
            outfile.write(output_string)
    except PermissionError:
        messagebox.showerror('Please Close the File Before Saving', 'The file {} is being used by another process! Please close it, THEN press the OK button to retry saving'.format(outfilename))
        with open(outfilename, 'w') as outfile:
            outfile.write(output_string)


def plot_crossval_scores(crossval_data, scheme_name, params_obj, outputdir):
    """
    Make plots of mean and std dev scores for training and test data from cross validation.
    :param crossval_data: tuple of (training means, training stds, test means, test stds) lists
    :param params_obj: Parameters object with plot information
    :type params_obj: Parameters
    :param scheme_name: (string) name of scheme for labeling purposes
    :param outputdir: directory in which to save output
    :return: void
    """
    plt.clf()
    plt.figure(figsize=(params_obj.plot_03_figwidth, params_obj.plot_04_figheight), dpi=params_obj.plot_05_dpi)

    train_score_means = crossval_data[0]
    train_score_stds = crossval_data[1]
    test_score_means = crossval_data[2]
    test_score_stds = crossval_data[3]

    xax = np.arange(1, len(train_score_means) + 1)
    plt.plot(xax, train_score_means, color='blue', marker='s', label='train_score', markersize=params_obj.plot_14_dot_size, markeredgecolor='black')
    plt.fill_between(xax, train_score_means-train_score_stds, train_score_means+train_score_stds, color='blue', alpha=0.2)
    plt.plot(xax, test_score_means, color='green', marker='o', label='test_score', markersize=params_obj.plot_14_dot_size, markeredgecolor='black')
    plt.fill_between(xax, test_score_means-test_score_stds, test_score_means+test_score_stds, color='green', alpha=0.2)

    # plot titles, labels, and legends
    if params_obj.plot_12_custom_title is not None:
        plot_title = params_obj.plot_12_custom_title
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    elif params_obj.plot_11_show_title:
        plot_title = scheme_name
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    if params_obj.plot_08_show_axes_titles:
        plt.xlabel('Number of Features (Collision Voltages)', fontsize=params_obj.plot_13_font_size, fontweight='bold')
        plt.ylabel('Accuracy Ratio', fontsize=params_obj.plot_13_font_size, fontweight='bold')
    plt.xticks(fontsize=params_obj.plot_13_font_size)
    plt.yticks(fontsize=params_obj.plot_13_font_size)
    if params_obj.plot_07_show_legend:
        plt.legend(loc='best', fontsize=params_obj.plot_13_font_size)

    output_name = os.path.join(outputdir, scheme_name + '_crossval' + params_obj.plot_02_extension)
    try:
        plt.savefig(output_name)
    except PermissionError:
        messagebox.showerror('Please Close the File Before Saving', 'The file {} is being used by another process! Please close it, THEN press the OK button to retry saving'.format(output_name))
        plt.savefig(output_name)
    plt.close()


def save_lda_and_predictions(scheme, lda_transform_data_by_file, predicted_class_by_file, probs_by_file, filenames, output_path, unknowns_bool):
    """
    Unified CSV output method for saving classification results (data transformed along LD axes and predicted
    classes and probabilities from the SVM). Used both when building a classification scheme and when
    classifying unknowns.
    :param scheme: ClassificationScheme container
    :type scheme: ClassificationScheme
    :param lda_transform_data_by_file: list for each input file of transformed data for each LD axis (i.e. [file1=[LD1, LD2], file2=[LD1, LD2]])
    :param predicted_class_by_file: list of lists of predicted classes, same format as lda_transform data (except listed by each class, not each LD axis)
    :param probs_by_file: list of lists of probabilities for each class, same format as predicted class data
    :param filenames: list of input filenames
    :param output_path: directory in which to save output
    :param unknowns_bool: True if saving unknown data
    :return: void
    """
    if unknowns_bool:
        outputname = os.path.join(output_path, scheme.name + '_Results-unknowns.csv')
    else:
        outputname = os.path.join(output_path, scheme.name + '_Results.csv')

    # arrange header information
    predict_labels = ','.join(['{} Probability'.format(x) for x in scheme.unique_labels])
    ld_dim_list = np.arange(1, len(lda_transform_data_by_file[0][0]) + 1)
    lda_header = ','.join('LD {} (linear discriminant dimension {})'.format(x, x) for x in ld_dim_list)
    header = 'File,Feature,{},Class Prediction,{}\n'.format(lda_header, predict_labels)
    output_string = header

    # loop over each file and each CV ("feature")
    for file_index in range(len(lda_transform_data_by_file)):
        transforms_by_cv = lda_transform_data_by_file[file_index]
        classes_by_cv = predicted_class_by_file[file_index]
        probs_by_cv = probs_by_file[file_index]

        for feat_index in range(len(transforms_by_cv)):
            ld_data = ','.join(['{:.2f}'.format(x) for x in transforms_by_cv[feat_index]])
            class_pred = scheme.unique_labels[classes_by_cv[feat_index] - 1]
            probs_data = ','.join(['{:.3f}'.format(x) for x in probs_by_cv[feat_index]])

            line = '{},{},{},{},{}\n'.format(filenames[file_index], scheme.input_feats[feat_index], ld_data, class_pred, probs_data)
            output_string += line

        # Add 'combined' probabilities at end of each file
        class_mode = scheme.unique_labels[np.argmax(np.bincount(classes_by_cv)) - 1]
        avg_probs = ','.join(['{:.3f}'.format(x) for x in np.average(probs_by_file[file_index], axis=0)])
        buffer_string = ','.join(['' for _ in range(len(ld_dim_list))])     # don't write anything into LD columns
        output_string += '{},{},{},{},{}\n'.format(filenames[file_index], 'Combined', buffer_string, class_mode, avg_probs)

    # Add explained variance ratio at the end of all files
    if scheme.explained_variance_ratio is not None:
        joined_exp_var = ','.join(['{:.3f}'.format(x) for x in scheme.explained_variance_ratio])
        output_string += 'Explained_variance_ratio,{},{}\n'.format(' ', joined_exp_var)

    # save to file
    try:
        with open(outputname, 'w') as outfile:
            outfile.write(output_string)
    except PermissionError:
        messagebox.showerror('Please Close the File Before Saving', 'The file {} is being used by another process! Please close it, THEN press the OK button to retry saving'.format(outputname))
        with open(outputname, 'w') as outfile:
            outfile.write(output_string)


def plot_probabilities(params_obj, scheme, class_probs_by_file, output_path, unknown_bool):
    """
    Generate a stacked bar graph of classification output probabilities for crossval data or unknowns.
    Organized such that the most likely class is on top of each stacked bar, and that the probabilities
    sum to 100%.
    :param params_obj: parameter container with plot options
    :type params_obj: Parameters
    :param scheme: Classification scheme with label information
    :type scheme: ClassificationScheme
    :param class_probs_by_file: list of lists of probabilities for each class (i.e. [file 1=[cv1[class1, class2, ...], cv2, ...], class 2, ...], file2, ...])
    :param output_path: directory in which to save the plot
    :param unknown_bool: if saving unknown data or not (to label output)
    :return: void
    """
    plt.clf()
    plt.figure(figsize=(params_obj.plot_03_figwidth, params_obj.plot_04_figheight), dpi=params_obj.plot_05_dpi)

    # prepare colors and labels
    initial_colors = ['dodgerblue', 'orange', 'fuchsia', 'forestgreen', 'gray', 'cyan', 'lightgreen', 'magenta', 'yellow']
    colors, labels = [], []
    for index, label in enumerate(scheme.unique_labels):
        colors.append(initial_colors[index])
        labels.append(label)

    # Stack each class probability in the bar graph
    for file_index in range(len(class_probs_by_file)):
        avg_probs = np.average(class_probs_by_file[file_index], axis=0)
        prob_class_tups = [(ind, val) for ind, val in enumerate(avg_probs)]
        sorted_probs = sorted(prob_class_tups, key=lambda x: x[1])
        y_offset = 0
        bar_index = 0
        for class_index, class_prob in sorted_probs:
            if bar_index > 0:
                y_offset += sorted_probs[bar_index - 1][1]
            plt.bar(file_index, class_prob, bottom=y_offset, color=colors[class_index], width=0.6)
            bar_index += 1

    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax.legend(labels, loc='center left', bbox_to_anchor=(1.2, 0.5), fontsize=params_obj.plot_13_font_size)

    # plot titles, labels, and legends
    if params_obj.plot_12_custom_title is not None:
        plot_title = params_obj.plot_12_custom_title
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    elif params_obj.plot_11_show_title:
        plot_title = 'Probabilities by Class for {}'.format(scheme.name)
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    if params_obj.plot_08_show_axes_titles:
        plt.xlabel('Sample Number', fontsize=params_obj.plot_13_font_size, fontweight='bold')
        plt.ylabel('Probability', fontsize=params_obj.plot_13_font_size, fontweight='bold')
    if params_obj.plot_07_show_legend:
        handles = []
        for index, color in enumerate(colors):
            patch_artist = matplotlib.patches.Patch(color=color, label=labels[index])
            handles.append(patch_artist)
        plt.legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=params_obj.plot_13_font_size)
    num_files = len(class_probs_by_file)
    plt.xticks(np.arange(num_files), np.arange(1, num_files + 1), fontsize=params_obj.plot_13_font_size)
    plt.yticks(fontsize=params_obj.plot_13_font_size)

    if unknown_bool:
        output_name = os.path.join(output_path, scheme.name + '_probs-unknown' + params_obj.plot_02_extension)
    else:
        output_name = os.path.join(output_path, scheme.name + '_probs' + params_obj.plot_02_extension)

    try:
        plt.savefig(output_name, bbox_inches='tight')
    except PermissionError:
        messagebox.showerror('Please Close the File Before Saving', 'The file {} is being used by another process! Please close it, THEN press the OK button to retry saving'.format(output_name))
        plt.savefig(output_name, bbox_inches='tight')
    plt.close()


def plot_classification_decision_regions(class_scheme, params_obj, output_path, unknown_tups=None):
    """
    Make a plot of decision regions determined for this classification scheme and the locations
    of the input test (validation) data against the constructed scheme.
    :param class_scheme: Classification object
    :param output_path: directory in which to save plot
    :param params_obj: Parameters object with plot information
    :type params_obj: Parameters
    :param unknown_tups: tuples of unknown (data, label). data = transformed lda data for unknown
    :return: void
    """
    shape_lda = np.shape(class_scheme.transformed_test_data)
    if shape_lda[1] > 3:
        # do not plot images with more than 3 dimensions
        return
    fig = plt.figure(figsize=(params_obj.plot_03_figwidth, params_obj.plot_04_figheight), dpi=params_obj.plot_05_dpi)

    # markers = ('s', 'o')
    markers = ('s', 'o', '^', 'v', 'D', '<', '>', '4', '8', 'h', 'H', '1', '2', '3', '+', '*', 'p', 'P', 'x')
    # colors = ['deepskyblue', 'orange', 'fuchsia', 'mediumspringgreen', 'gray', 'cyan', 'lightgreen', 'magenta', 'yellow']
    colors = ['dodgerblue', 'orange', 'fuchsia', 'forestgreen', 'gray', 'cyan', 'lightgreen', 'magenta', 'yellow']

    cmap = ListedColormap(colors[:len(class_scheme.unique_labels)])
    # decide whether the data has 1d or nds
    ax = plt.subplot(111)

    # plot 1D or 2D decision regions
    if shape_lda[1] == 1:
        x1_min, x1_max = np.floor(class_scheme.transformed_test_data.min()), np.ceil(class_scheme.transformed_test_data.max())
        if unknown_tups is not None:
            for unknown_tup in unknown_tups:
                min_unk = unknown_tup[0].min()
                max_unk = unknown_tup[0].max()
                x1_min = min([min_unk, x1_min])
                x1_max = max([max_unk, x1_max])

        # create a grid to evaluate model
        x2_min, x2_max = -1, 1
        x_grid, y_grid = np.mgrid[x1_min:x1_max:100j, x2_min:x2_max:100j]
        z = class_scheme.classifier.predict(x_grid.ravel().reshape(-1, 1))
        z = z.reshape(x_grid.shape)

        plt.contourf(x_grid, y_grid, z, alpha=0.4, cmap=cmap)
        plot_sklearn_lda_1ld(class_scheme, markers, colors, params_obj.plot_14_dot_size)
        if params_obj.plot_08_show_axes_titles:
            plt.xlabel('LD1', fontsize=params_obj.plot_13_font_size, fontweight='bold')
        plt.xticks(fontsize=params_obj.plot_13_font_size)
        plt.yticks([])
        if unknown_tups is not None:
            for ind, unknown_tup in enumerate(unknown_tups):
                if ind >= len(markers):
                    marker = 'x'
                else:
                    marker = markers[ind]
                plt.scatter(unknown_tup[0], np.zeros(np.shape(unknown_tup[0])), marker=marker, color='black',
                            alpha=0.5, label=unknown_tup[1], s=params_obj.plot_14_dot_size ** 2 * 2, edgecolors='black')

    if shape_lda[1] == 2:
        x1_min, x1_max = np.floor(class_scheme.transformed_test_data[:, 0].min()), np.ceil(class_scheme.transformed_test_data[:, 0].max())
        x2_min, x2_max = np.floor(class_scheme.transformed_test_data[:, 1].min()), np.ceil(class_scheme.transformed_test_data[:, 1].max())
        if unknown_tups is not None:
            for unknown_tup in unknown_tups:
                x1_min_unk, x1_max_unk = np.floor(unknown_tup[0][:, 0].min()), np.ceil(unknown_tup[0][:, 0].max())
                x2_min_unk, x2_max_unk = np.floor(unknown_tup[0][:, 1].min()), np.ceil(unknown_tup[0][:, 1].max())
                x1_min = min([x1_min_unk, x1_min])
                x1_max = max([x1_max_unk, x1_max])
                x2_min = min([x2_min_unk, x2_min])
                x2_max = max([x2_max_unk, x2_max])

        num_grid_bins = 1000
        x_grid_1, x_grid_2 = np.meshgrid(np.arange(x1_min, x1_max, abs(x1_max - x1_min) / num_grid_bins), np.arange(x2_min, x2_max, abs(x2_max - x2_min) / num_grid_bins))
        z = class_scheme.classifier.predict(np.array([x_grid_1.ravel(), x_grid_2.ravel()]).T)
        z = z.reshape(x_grid_1.shape)

        plt.contourf(x_grid_1, x_grid_2, z, alpha=0.4, cmap=cmap)
        plot_sklearn_lda_2ld(class_scheme, markers, colors, params_obj.plot_08_show_axes_titles, params_obj.plot_14_dot_size)
        if params_obj.plot_08_show_axes_titles:
            plt.xlabel('LD1', fontsize=params_obj.plot_13_font_size, fontweight='bold')
            plt.ylabel('LD2', fontsize=params_obj.plot_13_font_size, fontweight='bold')
        plt.xticks(fontsize=params_obj.plot_13_font_size)
        plt.yticks(fontsize=params_obj.plot_13_font_size)
        if unknown_tups is not None:
            for ind, unknown_tup in enumerate(unknown_tups):
                if ind >= len(markers):
                    marker = 'x'
                else:
                    marker = markers[ind]
                plt.scatter(x=unknown_tup[0][:, 0], y=unknown_tup[0][:, 1], marker=marker, color='black',
                            alpha=0.5, label=unknown_tup[1], s=params_obj.plot_14_dot_size ** 2, edgecolors='black')

    if shape_lda[1] == 3:
        print('NOTE: 3D plots are not fully optimized. Labels and font sizes may not be perfect.')
        ax = Axes3D(fig)
        plot_data = class_scheme.transformed_test_data
        y_values = class_scheme.numeric_labels
        unique_labels = class_scheme.unique_labels

        for label, marker, color in zip(range(0, len(unique_labels)), markers, colors):
            ax.scatter(xs=plot_data[:, 0][y_values == label + 1],
                       ys=plot_data[:, 1][y_values == label + 1],
                       zs=plot_data[:, 2][y_values == label + 1],
                       # marker=marker,
                       c=color, s=params_obj.plot_14_dot_size ** 2, edgecolors='black',
                       label=unique_labels[label])

        if unknown_tups is not None:
            for ind, unknown_tup in enumerate(unknown_tups):
                if ind >= len(markers):
                    marker = 'x'
                else:
                    marker = markers[ind]
                ax.scatter(xs=unknown_tup[0][:, 0],
                           ys=unknown_tup[0][:, 1],
                           zs=unknown_tup[0][:, 2],
                           marker=marker, c='black',
                           s=params_obj.plot_14_dot_size ** 2, edgecolors='black', alpha=0.9, label=unknown_tup[1])

        if params_obj.plot_08_show_axes_titles:
            ax.set_xlabel('LD1')
            ax.set_ylabel('LD2')
            ax.set_zlabel('LD3')
        # EDIT THIS IF YOU WANT TO CHANGE THE ANGLE OF THE PLOT
        # ax.view_init(elev=20., azim=30.)

    # plot titles, labels, and legends
    if params_obj.plot_12_custom_title is not None:
        plot_title = params_obj.plot_12_custom_title
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    elif params_obj.plot_11_show_title:
        cv_string = ', '.join([str(np.round(x.cv, 1)) for x in class_scheme.selected_features])
        plot_title = 'From CVs: {}'.format(cv_string)
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    if params_obj.plot_07_show_legend:
        ax.legend(loc='best', fontsize=params_obj.plot_13_font_size)

    if unknown_tups is not None:
        output_name = os.path.join(output_path, class_scheme.name + '_Results-unknowns' + params_obj.plot_02_extension)
    else:
        output_name = os.path.join(output_path, class_scheme.name + '_Results' + params_obj.plot_02_extension)
    try:
        plt.savefig(output_name)
    except PermissionError:
        messagebox.showerror('Please Close the File Before Saving', 'The file {} is being used by another process! Please close it, THEN press the OK button to retry saving'.format(output_name))
        plt.savefig(output_name)
    plt.close()


def plot_sklearn_lda_2ld(class_scheme, marker, color, label_axes, dot_size):
    """
    Scatter plot the class data onto a decision regions plot using provided colors/markers/etc
    :param class_scheme: scheme containing data and label information
    :param marker: list of marker styles to plot for each class
    :param color: list of colors to plot for each class
    :param dot_size: marker size in standard plot units (will be squared for scatter plot)
    :param label_axes: boolean: whether to label axes
    :return: void
    """
    x_data = class_scheme.transformed_test_data
    y_values = class_scheme.numeric_labels
    unique_labels = class_scheme.unique_labels

    for label, marker, color in zip(range(0, len(unique_labels)), marker, color):
        plt.scatter(x=x_data[:, 0][y_values == label + 1], y=x_data[:, 1][y_values == label + 1], marker=marker, color=color,
                    alpha=0.9, label=unique_labels[label], s=dot_size ** 2, edgecolors='black')
    if label_axes:
        plt.xlabel('LD1')
        plt.ylabel('LD2')


def plot_sklearn_lda_1ld(class_scheme, marker, color, dot_size):
    """
    Scatter plot the class data onto a decision regions plot using provided colors/markers/etc
    :param class_scheme: scheme containing data and label information
    :param marker: list of marker styles to plot for each class
    :param color: list of colors to plot for each class
    :param dot_size: marker size in standard plot units (will be squared for scatter plot)
    :return: void
    """
    x_data = class_scheme.transformed_test_data
    y_values = class_scheme.numeric_labels
    unique_labels = class_scheme.unique_labels

    for label, marker, color in zip(range(0, len(unique_labels)), marker, color):
        plt.scatter(x=x_data[:, 0][y_values == label + 1], y=np.zeros(np.shape(x_data[:, 0][y_values == label + 1])), marker=marker, color=color,
                    alpha=0.9, label=unique_labels[label], s=dot_size ** 2 * 2, edgecolors='black')


def save_scheme(scheme, outputdir):
    """
    Save a ClassificationScheme object into the provided output directory using pickle
    :param scheme: classification object to save
    :type scheme: ClassificationScheme
    :param outputdir: directory in which to save output
    :return: void
    """
    unique_labels = get_unique_labels(scheme.class_labels)
    save_name = 'Classifier_' + '_'.join(unique_labels)
    output_path = os.path.join(outputdir, save_name + '.clf')

    with open(output_path, 'wb') as save_file:
        pickle.dump(scheme, save_file)


def load_scheme(filepath):
    """
    Load a classification scheme object previously saved using pickle
    :param filepath: full system path to file location to load
    :return: loaded scheme object
    :rtype: ClassificationScheme
    """
    with open(filepath, 'rb') as loadfile:
        scheme = pickle.load(loadfile)
    return scheme


def multi_subclass_ufs(subclass_input_list, params_obj, output_path, subclass_labels):
    """
    Perform univariate feature selection across classes using multiple subclasses (e.g. different charge states
    of CIU fingerprints). Essentially performs standard UFS on each subclass separately and combines
    all output features/scores into a single list, from which the best features can be chosen for
    LDA/SVM classification. Inputs are structured like in standard UFS method, except provided
    as a list of inputs for each subclass rather than a single input (as the standard method takes only
    1 "subclass")
    :param subclass_input_list: list of ClassifInput containers for each subclass
    :type subclass_input_list: list[ClassifInput]
    :param params_obj: parameters information
    :type params_obj: Parameters
    :param output_path: directory in which to save plot
    :param subclass_labels: list of strings for scheme name
    :return: list of all features, sorted in decreasing order of score from ALL subclasses
    :rtype: list[CFeature]
    """
    # Iterate over all subclass lists to generate feature score information
    features = []
    features_by_subclass = []
    for subclass_input in subclass_input_list:
        # generate all combinations of replicate datasets within the labels
        shaped_label_list = subclass_input.shaped_label_list
        products = generate_products_for_ufs(subclass_input.analysis_objs_by_label, shaped_label_list, params_obj)

        # Create a CFeature object to hold the information for this CV (feature)
        scores = [product.fit_sc for product in products]
        mean_score = np.mean(scores, axis=0)
        std_score = np.std(scores, axis=0)

        cv_axis = subclass_input.analysis_objs_by_label[0][0].axes[1]
        subclass_features = []
        for cv_index, cv in enumerate(cv_axis):
            feature = CFeature(cv, cv_index, mean_score[cv_index], std_score[cv_index], subclass_label=subclass_input.subclass_label)
            features.append(feature)
            subclass_features.append(feature)
        features_by_subclass.append(subclass_features)

    # sort feature scores either by mean - stdev ("error mode") or just mean alone.
    if params_obj.classif_6_ufs_use_error_mode:
        sorted_features = sorted(features, key=lambda x: (x.mean_score - x.std_dev_score), reverse=True)
    else:
        sorted_features = sorted(features, key=lambda x: x.mean_score, reverse=True)

    unique_labels = get_unique_labels([x for label_list in subclass_input_list[0].shaped_label_list for x in label_list])
    # scheme_name = '_'.join(unique_labels)
    scheme_name = generate_scheme_name(unique_labels, subclass_labels)
    # plot_feature_scores(sorted_features, params_obj, scheme_name, output_path)
    plot_feature_scores_subclass(features_by_subclass, params_obj, scheme_name, output_path)
    return sorted_features


def get_subclass_index(subclass_label, list_classif_inputs):
    """
    Return the index of a particular subclass in a list of ClassifInput containers
    :param subclass_label: string label
    :param list_classif_inputs: list of ClassifInput containers
    :type list_classif_inputs: list[ClassifInput]
    :return: int (index)
    """
    indices = [index for index, input_obj in enumerate(list_classif_inputs) if input_obj.subclass_label == subclass_label]
    subclass_index = indices[0]
    return subclass_index


def main_build_classification_subclass(list_classif_inputs, params_obj, output_dir, known_feats=None):
    """
    Main method for classification. Performs feature selection followed by LDA and classification
    and generates output and plots. Returns a ClassificationScheme object to be saved for future
    classification of unknowns.
    :param list_classif_inputs: list of ClassifInput containers
    :type list_classif_inputs: list[ClassifInput]
    :param output_dir: directory in which to save plots/output
    :param params_obj: Parameters object with classification parameter information
    :type params_obj: Parameters
    :param known_feats: list of
    :return: ClassificationScheme object with the generated scheme
    :rtype: ClassificationScheme
    """
    if known_feats is None:
        # run feature selection and crossvalidation to select best features automatically
        all_features = multi_subclass_ufs(list_classif_inputs, params_obj, output_dir, [])

        # assess all features to determine which to use in the final scheme
        best_features, crossval_score, all_crossval_data = crossval_main_subclass(list_classif_inputs, output_dir, params_obj, all_features)
    else:
        # Manual mode: use the provided features and run limited crossvalidation
        best_features, crossval_score, all_crossval_data = crossval_main_subclass(list_classif_inputs, output_dir, params_obj, known_feats)
        best_features = known_feats

    # perform LDA and classification on the selected/best features
    constructed_scheme = lda_best_features_subclass(best_features, list_classif_inputs, params_obj, output_dir)
    constructed_scheme.crossval_test_score = crossval_score
    constructed_scheme.all_crossval_data = all_crossval_data

    # plot output here for now, will probably move eventually
    plot_classification_decision_regions(constructed_scheme, params_obj, output_dir)
    return constructed_scheme


def crossval_main_subclass(classif_inputs, outputdir, params_obj, features_list):
    """
    Updated crossval method to allow multiple subclasses. Same method as original crossval, but
    chooses data from the correct subclass if present.
    :param classif_inputs: list of ClassifInput containers
    :type classif_inputs: list[ClassifInput]
    :param outputdir:
    :param params_obj:
    :type params_obj: Parameters
    :param features_list: List of CFeatures, sorted in decreasing order of score
    :type features_list: list[CFeature]
    :return: list of selected features, test score for that # features, and all cross validation data
    """
    # determine training size as size of the smallest class - 1 (1 test file at a time)
    min_class_size = np.inf
    for subclass_input in classif_inputs:
        min_subclass_size = np.min([len(x) for x in subclass_input.analysis_objs_by_label])
        if min_subclass_size < min_class_size:
            min_class_size = min_subclass_size
    training_size = min_class_size - 1

    current_features_list = []
    train_score_means = []
    train_score_stds = []
    test_score_means = []
    test_score_stds = []
    results = []

    time_start = time.time()
    # optional max number of features to consider
    if params_obj.classif_7_max_feats_for_crossval > 0:
        max_features = params_obj.classif_7_max_feats_for_crossval
        if max_features > len(features_list):
            max_features = len(features_list) + 1
    else:
        max_features = len(features_list) + 1

    current_time = time_start
    for ind, feature in enumerate(features_list[:max_features]):
        # Generate all combinations - NOTE: assumes that if subclasses are present, features know their subclass (should always be true)
        print('\nPerforming cross validation for {} features, last took {:.1f}s'.format(ind+1, time.time() - current_time))
        current_time = time.time()

        current_features_list.append(feature)

        if len(classif_inputs) > 1:
            # multiple subclasses
            subclass_index = get_subclass_index(feature.subclass_label, classif_inputs)
        else:
            subclass_index = 0
        obj_list = classif_inputs[subclass_index].analysis_objs_by_label
        label_list = classif_inputs[subclass_index].class_labels

        # perform the cross validation for this feature combination
        crossval_obj = CrossValProduct(obj_list, label_list, training_size, current_features_list)
        crossval_combos = crossval_obj.assemble_class_combinations(params_obj)
        result = assemble_products(crossval_combos)
        results.append(result)

    for result in results:
        # get scores and plot and stuff
        train_score_means.append(result[0])
        train_score_stds.append(result[1])
        test_score_means.append(result[2])
        test_score_stds.append(result[3])
    print('classification done in {:.2f}'.format(time.time() - time_start))

    train_score_means = np.array(train_score_means)
    train_score_stds = np.array(train_score_stds)
    test_score_means = np.array(test_score_means)
    test_score_stds = np.array(test_score_stds)

    # save and plot crossvalidation score information
    crossval_data = (train_score_means, train_score_stds, test_score_means, test_score_stds)
    unique_labels = get_unique_labels(classif_inputs[0].class_labels)
    scheme_name = '_'.join(unique_labels)
    save_crossval_score(crossval_data, scheme_name, outputdir)
    plot_crossval_scores(crossval_data, scheme_name, params_obj, outputdir)

    # determine best features list from crossval scores
    best_num_feats, best_score = peak_crossval_score_detect(test_score_means, params_obj.classif_2_score_dif_tol)
    output_features = features_list[0: best_num_feats]
    return output_features, best_score, crossval_data


def lda_best_features_subclass(features_list, classif_inputs, param_obj, output_dir):
    """
    Construct final classification scheme using a support vector machine in the optimal linear
    discriminant space for the provided data. Only uses data from the specified features (activation
    voltages). Updated to handle data from multiple subclasses if necessary
    :param features_list: list of selected features from feature selection
    :type features_list: list[CFeature]
    :param classif_inputs: list of ClassifInput containers
    :type classif_inputs: list[ClassifInput]
    :param param_obj: parameters information
    :type param_obj: Parameters
    :param output_dir: directory in which to save output plot
    :return: generated classification scheme object with LDA and SVC performed
    :rtype: ClassificationScheme
    """
    # loop over features, assembling correct data for LDA for all classes/replicates for each
    x_data, label_data, filenames, cvs, file_ids = [], [], [], [], []
    for feature in features_list:
        x_data, label_data, filenames, cvs, file_ids = lda_prep_subclass(feature, classif_inputs, param_obj, x_data, label_data, filenames, cvs, file_ids)

    # finalize input data for LDA
    x_data = np.asarray(x_data)
    input_y_labels, target_label = createtargetarray_featureselect(label_data)

    # run LDA
    lda = LinearDiscriminantAnalysis(solver='svd', n_components=5)
    lda.fit(x_data, input_y_labels)
    x_lda = lda.transform(x_data)
    expl_var_r = lda.explained_variance_ratio_

    # build classification scheme
    clf = SVC(kernel='linear', C=1, probability=True, max_iter=1000)
    clf.fit(x_lda, input_y_labels)
    y_pred = clf.predict(x_lda)
    prec_score = precision_score(input_y_labels, y_pred, pos_label=1, average='weighted')
    probs = clf.predict_proba(x_lda)

    # initialize classification scheme object and return it
    scheme = ClassificationScheme()
    scheme.selected_features = features_list
    scheme.classifier = clf
    scheme.classifier_type = 'SVC'
    scheme.classif_prec_score = prec_score
    scheme.lda = lda
    scheme.explained_variance_ratio = expl_var_r
    scheme.numeric_labels = input_y_labels
    scheme.class_labels = target_label
    scheme.unique_labels = get_unique_labels(target_label)
    scheme.transformed_test_data = x_lda
    scheme.test_filenames = filenames
    scheme.params = clf.get_params()
    scheme.input_feats = cvs
    # scheme.set_name()
    scheme.name = generate_scheme_name(label_data, [])

    # Organize outputs by input file for easier viewing, then save
    x_lda_by_file, y_pred_by_file, probs_by_file, filenames_by_file = prep_outputs_by_file(x_lda, y_pred, probs, filenames, file_ids)

    save_lda_and_predictions(scheme, x_lda_by_file, y_pred_by_file, probs_by_file, filenames_by_file, output_dir, unknowns_bool=False)
    plot_probabilities(param_obj, scheme, probs_by_file, output_dir, unknown_bool=False)
    return scheme


def prep_outputs_by_file(x_lda, y_pred, probs, filenames, file_ids):
    """
    Organize outputs from LDA into groups sorted by input raw file for easy output formatting.
    All 4 input lists are same shape and would be made into a nice container object except that
    they come out all together from classification.
    :param x_lda: transformed LDA result data
    :param y_pred: predicted class outputs
    :param probs: predicted class probabilities
    :param filenames: raw filenames
    :param file_ids: list of strings to keep track of class/replicate ID to order correctly
    :return: sublist of each input list that is all the entries corresponding to the specific file
    """
    x_lda_by_file, y_pred_by_file, probs_by_file, filenames_by_file = [], [], [], []

    # Get unique IDs only
    unique_ids = []
    for file_id in file_ids:
        if file_id not in unique_ids:
            unique_ids.append(file_id)

    for unique_id in unique_ids:
        x_lda_out, y_pred_out, probs_out = [], [], []
        file_indices = [index for index, x in enumerate(file_ids) if x == unique_id]

        for index in file_indices:
            x_lda_out.append(x_lda[index])
            y_pred_out.append(y_pred[index])
            probs_out.append(probs[index])

        x_lda_by_file.append(x_lda_out)
        y_pred_by_file.append(y_pred_out)
        probs_by_file.append(probs_out)
        filenames_by_file.append(filenames[file_indices[0]])

    return x_lda_by_file, y_pred_by_file, probs_by_file, filenames_by_file


def lda_prep_subclass(feature, classif_inputs, param_obj, x_data, label_data, filenames, cvs, file_ids):
    """
    Helper method to append the correct input data for LDA to the provided lists for a given
    feature. x_data, labels, filenames, and cvs are ongoing lists that get added to, feature
    and classif_inputs are the data getting assembled. Supports subclasses since the feature
    knows its subclass. Gets the data for the Feature at the correct CV/subclass for every
    class and replicate and add to the lists.
    :param feature: CFeature specifying data to add
    :type feature: CFeature
    :param classif_inputs: list of ClassifInput containers with class data/info
    :type classif_inputs: list[ClassifInput]
    :param x_data: ongoing list of actual data for LDA
    :param param_obj: container with parameter information
    :type param_obj: Parameters
    :param label_data: ongoing list of labels for LDA
    :param filenames: ongoing list of filenames for outputs
    :param cvs: ongoing list of CVs chosen for outputs
    :param file_ids: ongoing list of file IDs (class/rep) for later sorting
    :return: x_data, label_data, filenames, cvs updated with values from this feature
    """
    # Get the correct subclass if more than one present
    if feature.subclass_label is not None:
        subclass_index = get_subclass_index(feature.subclass_label, classif_inputs)
    else:
        subclass_index = 0
    classif_input = classif_inputs[subclass_index]
    flat_obj_list = classif_input.get_flat_list(get_type='obj')

    # Assemble flattened lists to draw correct values from to append
    flat_raws = [get_classif_data(obj, param_obj) for obj in flat_obj_list]
    flat_labels = classif_input.get_flat_list(get_type='label')
    flat_filenames = [obj.short_filename for obj in flat_obj_list]
    flat_cv_axes = [obj.axes[1] for obj in flat_obj_list]
    flat_ids = classif_inputs[subclass_index].get_flat_list(get_type='rep')

    # Append data from each class/replicate to the output lists in order
    for data_index in range(len(flat_raws)):
        # Determine the correct CV column to append
        current_cv_axis = flat_cv_axes[data_index]
        this_cv_index = (np.abs(np.asarray(current_cv_axis) - feature.cv)).argmin()
        transposed_raw = flat_raws[data_index].T
        cv_col = transposed_raw[this_cv_index]

        x_data.append(cv_col)
        label_data.append(flat_labels[data_index])
        if flat_filenames is not None:
            filenames.append(flat_filenames[data_index])
        file_ids.append('{}_{}'.format(label_data[data_index], flat_ids[data_index]))
    cvs.append(feature.cv)

    return x_data, label_data, filenames, cvs, file_ids


def unk_subclass_dict_to_list(unk_subclass_obj):
    """
    Convert a dictionary of unknown inputs {subclass label: [list_analysis_objs]} to a list of
    ClassifInput containers by subclass for standard input into classification methods
    :param unk_subclass_obj: SubclassUnknown container
    :type unk_subclass_obj: SubclassUnknown
    :rtype: list[ClassifInput]
    """
    classif_inputs = []
    for subclass_label, obj_list in unk_subclass_obj.subclass_dict.items():
        classif_input = ClassifInput([unk_subclass_obj.label], [obj_list], subclass_label)
        classif_inputs.append(classif_input)
    return classif_inputs


def rearrange_ciu_by_feats(shaped_inputs_list, features_list, params_obj):
    """
    For each CIU dataset in the original input, generate a rearranged (and possibly shrunken)
    matrix of data in order of features in the provided features list. Designed to allow easy
    access to the CV subset data of interest prior to doing crossval/LDA/etc. Handles subclasses.
    :param features_list: list of features in descending order of score from UFS
    :type features_list: list[CFeature]
    :param shaped_inputs_list: lists by class of SubclassUnknown containers with classification input info
    :type shaped_inputs_list: list[list[ClInput]]
    :param params_obj: parameters container for getting classif data (raw data or gaussian mode)
    :type params_obj: Parameters
    :return: shaped output data with selected feature data in order
    :rtype: list[list[DataSubset]]
    """
    # Loop over the shaped input list, extracting data and maintaining the same organization
    shaped_output_list = []
    class_numeric_label = 1
    for class_list in shaped_inputs_list:
        class_outputs = []

        for rep_obj in class_list:
            data_output = []
            file_id = ''
            for feature in features_list:
                if feature.subclass_label is not None:
                    subclass_obj = rep_obj.get_subclass_obj(feature.subclass_label)
                    # multiple subclasses, so file_id is a combination of each ID
                    file_id += subclass_obj.short_filename
                    file_id += ';'
                else:
                    subclass_obj = rep_obj.get_subclass_obj()
                    file_id = subclass_obj.short_filename   # only one object, so just use its filename

                # Determine the correct CV column to append to the growing matrix and do so
                current_cv_axis = subclass_obj.axes[1]
                this_cv_index = (np.abs(np.asarray(current_cv_axis) - feature.cv)).argmin()
                raw_data = get_classif_data(subclass_obj, params_obj)
                cv_col = raw_data.T[this_cv_index]
                data_output.append(cv_col)

            # generate a subset container to hold the extracted data and associated metadata
            rep_subset = DataSubset(data_output, rep_obj.class_label, class_numeric_label, file_id, features_list)
            class_outputs.append(rep_subset)
        shaped_output_list.append(class_outputs)
        class_numeric_label += 1

    return shaped_output_list


def arrange_lda_new(subset_list):
    """
    Prepare data and label arrays for input to LDA methods given an input list of DataSubset containers
    with formatted data and labels
    :param subset_list: list of DataSubset containers with formatted data and labels
    :type subset_list: list[DataSubset]
    :return: x_data, numeric labels, string labels for direct input into LDA
    """
    x_data = []
    string_labels, numeric_labels = [], []
    # assemble each dataset into a single list by combining all columns of the input matrices
    for subset in subset_list:
        for column in subset.data:
            x_data.append(column)
            string_labels.append(subset.class_label)
            numeric_labels.append(subset.numeric_label)

    # convert to numpy arrays for SKLearn analyses
    x_data = np.asarray(x_data)
    string_labels = np.asarray(string_labels)
    numeric_labels = np.asarray(numeric_labels)
    return x_data, numeric_labels, string_labels


def run_lda_svc(x_data, label_data):
    """
    Run LDA and SVC analysis of a set of data and return the resulting classifier
    :param x_data: input x data
    :param label_data: input label data
    :return: sklearn.svm.SVC classifier object and LDA object
    :rtype: SVC, LinearDiscriminantAnalysis
    """
    lda = LinearDiscriminantAnalysis(solver='svd', n_components=5)
    lda.fit(x_data, label_data)
    train_lda = lda.transform(x_data)

    # max_iter=1000 needed to prevent occasional (and unpredictable) freezes with ridiculous iteration numbers
    svm = SVC(kernel='linear', C=1, probability=True, max_iter=1000)
    try:
        svm.fit(train_lda, label_data)
    except ValueError:
        print('Error in SVM fitting. This should not be reached - check your input data for duplicates (same input used in multiple classes)')

    return svm, lda


def main_build_classification_new(cl_inputs_by_label, subclass_labels, params_obj, output_dir, known_feats=None):
    """
    Main method for classification. Performs feature selection followed by LDA and classification
    and generates output and plots. Returns a ClassificationScheme object to be saved for future
    classification of unknowns. Allows use of subclasses.
    :param cl_inputs_by_label: lists of ClInput containers sorted by class label
    :type cl_inputs_by_label: list[list[ClInput]]
    :param subclass_labels: list of subclass labels (strings). If no subclasses present, default to ['0']
    :param output_dir: directory in which to save plots/output
    :param params_obj: Parameters object with classification parameter information
    :type params_obj: Parameters
    :param known_feats: list of features (optional, allows manual mode)
    :return: ClassificationScheme object with the generated scheme
    :rtype: ClassificationScheme
    """
    if known_feats is None:
        # convert to subclass oriented data (if no subclasses, will be a single entry in a list)
        class_labels = [class_list[0].class_label for class_list in cl_inputs_by_label]
        list_classif_inputs = subclass_inputs_from_class_inputs(cl_inputs_by_label, subclass_labels, class_labels)

        # run feature selection and crossvalidation to select best features automatically
        all_features = multi_subclass_ufs(list_classif_inputs, params_obj, output_dir, subclass_labels)

        # assess all features to determine which to use in the final scheme
        best_features, crossval_score, all_crossval_data, best_crossval = crossval_main_new(cl_inputs_by_label, output_dir, params_obj, all_features, subclass_labels)
    else:
        # Manual mode: use the provided features and run limited crossvalidation
        best_features, crossval_score, all_crossval_data, best_crossval = crossval_main_new(cl_inputs_by_label, output_dir, params_obj, known_feats, subclass_labels)
        best_features = known_feats

    # perform LDA and classification on the selected/best features
    shaped_subsets = rearrange_ciu_by_feats(cl_inputs_by_label, best_features, params_obj)
    flat_subsets = [x for class_list in shaped_subsets for x in class_list]
    constructed_scheme = lda_svc_best_feats(flat_subsets, best_features, params_obj, output_dir, subclass_labels)

    # constructed_scheme = lda_best_features_subclass(best_features, list_classif_inputs, params_obj, output_dir)
    constructed_scheme.crossval_test_score = crossval_score
    constructed_scheme.all_crossval_data = all_crossval_data

    # plot output here for now, will probably move eventually
    plot_classification_decision_regions(constructed_scheme, params_obj, output_dir)
    return constructed_scheme


def subclass_inputs_from_class_inputs(cl_inputs_by_label, subclass_label_list, class_labels):
    """
    Generate and return a list of ClassifInput containers (organized by subclass) from the primary
    input of replicates organized by class label.
    :param cl_inputs_by_label: list of lists of ClInput containers with any subclasses present
    :type cl_inputs_by_label: list[list[ClInput]]
    :param subclass_label_list: list of subclass label strings to search for (strings)
    :param class_labels: list of class labels (strings)
    :return: list of ClassifInputs
    :rtype: list[ClassifInput]
    """
    # Generate subclass oriented files for UFS from the primary input
    classif_inputs = []
    for subclass_label in subclass_label_list:
        objs_by_label = []
        for class_list in cl_inputs_by_label:
            obj_list_this_label = []
            for cl_input in class_list:
                subclass_obj = cl_input.subclass_dict[subclass_label]
                obj_list_this_label.append(subclass_obj)
            objs_by_label.append(obj_list_this_label)
        classif_input = ClassifInput(class_labels, objs_by_label, subclass_label)
        classif_inputs.append(classif_input)
    return classif_inputs


def lda_svc_best_feats(flat_subset_list, features_list, params_obj, output_dir, subclass_labels):
    """
    Generate a Scheme container by performing final LDA/SVM analysis on the provided data.
    :param flat_subset_list: list of DataSubset containers
    :type flat_subset_list: list[DataSubset]
    :param features_list: selected (best) features to use in scheme construction
    :type features_list: list[CFeature]
    :param params_obj: parameters
    :type params_obj: Parameters
    :param output_dir: directory in which to save output
    :param subclass_labels: list of strings for scheme output naming
    :return: Scheme container
    :rtype: ClassificationScheme
    """
    train_data, train_numeric_labels, train_string_labels = arrange_lda_new(flat_subset_list)
    svc, lda = run_lda_svc(train_data, train_numeric_labels)

    x_lda = lda.transform(train_data)
    expl_var_r = lda.explained_variance_ratio_

    # build classification scheme
    clf = SVC(kernel='linear', C=1, probability=True, max_iter=1000)
    clf.fit(x_lda, train_numeric_labels)
    y_pred = clf.predict(x_lda)
    prec_score = precision_score(train_numeric_labels, y_pred, pos_label=1, average='weighted')
    probs = clf.predict_proba(x_lda)

    # initialize classification scheme object and return it
    scheme = ClassificationScheme()
    scheme.selected_features = features_list
    scheme.classifier = clf
    scheme.classifier_type = 'SVC'
    scheme.classif_prec_score = prec_score
    scheme.lda = lda
    scheme.explained_variance_ratio = expl_var_r
    scheme.numeric_labels = train_numeric_labels
    scheme.class_labels = train_string_labels
    scheme.unique_labels = get_unique_labels(train_string_labels)
    scheme.transformed_test_data = x_lda
    scheme.params = clf.get_params()
    scheme.input_feats = [feature.cv for feature in features_list]
    scheme.name = generate_scheme_name(train_string_labels, subclass_labels)

    # Organize outputs by input file for easier viewing, then save
    x_lda_by_file, y_pred_by_file, probs_by_file, filenames_by_file = prep_outputs_by_file_new(x_lda, y_pred, probs, flat_subset_list)
    save_lda_and_predictions(scheme, x_lda_by_file, y_pred_by_file, probs_by_file, filenames_by_file, output_dir, unknowns_bool=False)
    plot_probabilities(params_obj, scheme, probs_by_file, output_dir, unknown_bool=False)
    return scheme


def prep_outputs_by_file_new(x_lda, y_pred, probs, flat_subset_list):
    """
    Organize outputs from LDA into groups sorted by input raw file for easy output formatting.
    All 4 input lists are same shape and would be made into a nice container object except that
    they come out all together from classification.
    :param x_lda: transformed LDA result data
    :param y_pred: predicted class outputs
    :param probs: predicted class probabilities
    :param flat_subset_list: list input data from DataSubset containers
    :type flat_subset_list: list[DataSubset]
    :return: sublist of each input list that is all the entries corresponding to the specific file
    """
    x_lda_by_file, y_pred_by_file, probs_by_file, filenames_by_file = [], [], [], []

    num_features = len(flat_subset_list[0].features)

    # loop over each input file (subset) and organize data accordingly
    output_column_index = 0
    for subset in flat_subset_list:
        # each file has num_features columns of data in the output matrix - get those
        file_xlda, file_ypred, file_probs = [], [], []
        for feat_index in range(num_features):
            file_xlda.append(x_lda[output_column_index])
            file_ypred.append(y_pred[output_column_index])
            file_probs.append(probs[output_column_index])
            output_column_index += 1
        x_lda_by_file.append(file_xlda)
        y_pred_by_file.append(file_ypred)
        probs_by_file.append(file_probs)
        filenames_by_file.append(subset.file_id)

    return x_lda_by_file, y_pred_by_file, probs_by_file, filenames_by_file


def crossval_main_new(cl_inputs_by_label, outputdir, params_obj, features_list, subclass_labels):
    """
    Updated crossval method to allow multiple subclasses. Updated to reduce datasets to selected
    features first, then perform crossval with modular methods.
    :param cl_inputs_by_label: lists of ClInput containers sorted by class label
    :type cl_inputs_by_label: list[list[ClInput]]
    :param outputdir: directory in which to save output plots
    :param params_obj: parameters container
    :type params_obj: Parameters
    :param features_list: List of CFeatures, sorted in decreasing order of score
    :type features_list: list[CFeature]
    :param subclass_labels: list of strings for scheme naming purposes
    :return: list of selected features, test score for that # features, and all cross validation data
    """
    # determine training size as size of the smallest class - 1 (1 test file at a time)
    min_class_size = np.min([len(x) for x in cl_inputs_by_label])
    training_size = min_class_size - 1
    label_list = [class_list[0].class_label for class_list in cl_inputs_by_label]

    # optional max number of features to consider
    if params_obj.classif_7_max_feats_for_crossval > 0:
        max_features = params_obj.classif_7_max_feats_for_crossval
        if max_features > len(features_list):
            max_features = len(features_list) + 1
    else:
        max_features = len(features_list) + 1

    current_features_list = []
    train_score_means, train_score_stds, test_score_means, test_score_stds, crossvals = [], [], [], [], []
    for ind, feature in enumerate(features_list[:max_features]):
        # Generate all combinations - NOTE: assumes that if subclasses are present, features know their subclass (should always be true)
        print('\nPerforming cross validation for {} features'.format(ind + 1))
        current_features_list.append(feature)

        # format all data
        shaped_data_list = rearrange_ciu_by_feats(cl_inputs_by_label, current_features_list, params_obj)

        # perform the cross validation for this feature combination
        crossval_obj = CrossValRun(shaped_data_list, label_list, training_size, current_features_list)
        crossval_obj.divide_data_and_run_lda()
        crossval_obj.assemble_class_products()

        # save results
        train_score_means.append(crossval_obj.train_scores_mean)
        train_score_stds.append(crossval_obj.train_scores_std)
        test_score_means.append(crossval_obj.test_scores_mean)
        test_score_stds.append(crossval_obj.test_scores_std)
        crossvals.append(crossval_obj)

    train_score_means = np.asarray(train_score_means)
    train_score_stds = np.asarray(train_score_stds)
    test_score_means = np.asarray(test_score_means)
    test_score_stds = np.asarray(test_score_stds)

    # save and plot crossvalidation score information
    crossval_data = (train_score_means, train_score_stds, test_score_means, test_score_stds)
    unique_labels = get_unique_labels(label_list)
    scheme_name = generate_scheme_name(unique_labels, subclass_labels)
    save_crossval_score(crossval_data, scheme_name, outputdir)
    plot_crossval_scores(crossval_data, scheme_name, params_obj, outputdir)

    # determine best features list from crossval scores
    best_num_feats, best_score = peak_crossval_score_detect(test_score_means, params_obj.classif_2_score_dif_tol)
    output_features = features_list[0: best_num_feats]
    best_crossval = crossvals[best_num_feats - 1]   # 0 indexed vs 1 indexed
    return output_features, best_score, crossval_data, best_crossval


def generate_scheme_name(class_labels, subclass_labels):
    """
    Generate the name of a classifying scheme to be a combination of the provided class labels
    :param class_labels: list of strings
    :param subclass_labels: list of strings
    :return: string name
    """
    # Get class labels
    unique_labels = get_unique_labels(class_labels)

    # add number of subclasses if applicable
    len_sublabels = len(subclass_labels)
    if len_sublabels > 1:
        name = '_'.join(unique_labels) + '_{}SubCl'.format(len_sublabels)
    else:
        name = '_'.join(unique_labels)
    return name


class ClInput(object):
    """
    Container for all classification inputs. Essentially just a dictionary of {subclass label: analysis object}
    for each subclass with associated name and space to save output/classification predictions/probabilities
    Allows for all methods to handle subclass or non-subclass data. Represents a SINGLE replicate of a
    single class (and all subclasses if using subclass data)
    """

    def __init__(self, label, dict_subclass_objs):
        """
        Initialize new container. NOTE: if no subclasses, set entry key = '0'
        :param label: string used to label outputs from this combined unknown
        :param dict_subclass_objs: dict of key = subclass label, value = CIUAnalysisObj
        :type dict_subclass_objs: dict[str, CIUAnalysisObj]
        """
        self.class_label = label
        self.subclass_dict = dict_subclass_objs

        # output storage
        self.predicted_label = None
        self.probs_by_cv = None
        self.probs_avg = None
        self.transformed_data = None

    def get_subclass_obj(self, subclass_label=None):
        """
        Return the list of CIUAnalysisObj's corresponding to the provided subclass label string
        :param subclass_label: string to search for in the dict. If None, there should be only one entry - return it or warn
        :return: list of analysis objs for this subclass
        :rtype: CIUAnalysisObj
        """
        try:
            if subclass_label is None:
                # no label provided - this is for non-subclass mode, where the dict will have only 1 entry. Return it (and warn if >1 entry)
                if len(self.subclass_dict) > 1:
                    print('WARNING: single subclass requested but there are more than one. Erratic returns possible')
                for key, obj in self.subclass_dict.items():
                    return obj
            else:
                # return the requested object
                return self.subclass_dict[subclass_label]
        except KeyError:
            return None

    def __str__(self):
        """
        string rep
        :return: string
        """
        return '<ClInput> {}'.format(self.class_label)


class ClassifInput(object):
    """
    Container for classification input information (labels and CIUAnalysisObj containers)
    to keep track of labels/replicates and subclasses.
    """
    def __init__(self, class_labels, obj_list_by_label, subclass_label=None):
        """
        Initialize a new container with provided information, including optional subclass label
        :param class_labels: list of strings corresponding to labels for each class
        :param obj_list_by_label: list of lists of analysis containers corresponding to all replicates of each class (sorted by class)
        :param subclass_label: (optional) if this is a subclass analysis, the label corresponding to this subclass
        """
        # prepare labels for all replicates in each class
        self.class_labels = class_labels
        self.shaped_label_list = []
        self.shaped_replicates = []
        for index, label in enumerate(class_labels):
            self.shaped_label_list.append([label for _ in range(len(obj_list_by_label[index]))])
            self.shaped_replicates.append([x for x in range(len(obj_list_by_label[index]))])
        self.analysis_objs_by_label = obj_list_by_label
        self.subclass_label = subclass_label

    def get_flat_list(self, get_type):
        """
        Get a flattened list of labels or analysis objects
        :param get_type: whether to get labels ('label') or analysis objects ('obj')
        :return: flattened list
        :rtype: list[CIUAnalysisObj]    #  not always, but this resolves some erroneous type checking in PyCharm
        """
        if get_type == 'label':
            flat_label_list = [x for label_list in self.shaped_label_list for x in label_list]
            return flat_label_list
        elif get_type == 'rep':
            flat_reps = [x for rep_list in self.shaped_replicates for x in rep_list]
            return flat_reps
        elif get_type == 'obj':
            flat_obj_list = [x for analysis_obj_list in self.analysis_objs_by_label for x in analysis_obj_list]
            return flat_obj_list
        else:
            print('Invalid type request. Valid values are "label" or "obj"')

    def __str__(self):
        """
        Return string representation - subclass label
        :return: string
        """
        if self.subclass_label is not None:
            return '<ClassInput> {}'.format(self.subclass_label)
        else:
            return '<ClassInput>'
    __repr__ = __str__


class ClassificationScheme(object):
    """
    Container for final classification information, to be saved and used in future for unknown
    analyses, etc.
    """
    def __init__(self):
        self.lda = None     # type: LinearDiscriminantAnalysis
        self.classifier = None      # type: SVC
        self.classifier_type = None     # string - holds classifier type information for diagnostics
        self.name = None    # string - classifier name, usually assembled from input class names
        self.params = None  # return of self.classier.get_params()
        self.final_axis_cropvals = None     # final axes used in the scheme
        self.classif_mode = None    # standard or Gaussian mode, to ensure unknowns are fit in correct mode
        self.classif_prec_score = None  # sklearn.metrics.precision_score for the classification scheme
        self.explained_variance_ratio = None    # list of ratios of variance explained by each LD axis

        self.selected_features = []     # type: List[CFeature]
        self.all_features = None    # type: List[CFeature]

        self.numeric_labels = None  # list of class labels converted to ints
        self.class_labels = None    # list of original string class labels, with replicates
        self.unique_labels = None   # list of original string class labels, but only 1 per class
        self.transformed_test_data = None   # input data transformed by the final LDA
        # self.test_filenames = None  # list of filenames from input data
        self.input_feats = None     # list of CVs used in the scheme. NOT Feature objects, just CVs

        # cross validation information
        self.crossval_test_score = None     # best test accuracy score (score of the feature set chosen for final classification)
        self.all_crossval_data = None   # list of (train_score_means, train_score_stds, test_score_means, test_score_stds) for each feature combination

        self.num_gaussians = None       # stores the max number of Gaussian found in a voltage for assembling matrices for unknowns

    def __str__(self):
        label_string = ','.join(self.unique_labels)
        return '<Classif_Scheme> type: {}, data: {}'.format(self.classifier_type, label_string)
    __repr__ = __str__

    def get_subclass_labels(self):
        """
        Return a list of subclass labels from all selected features
        :return: list of subclass labels (strings)
        """
        subclass_labels = []
        for feature in self.selected_features:
            if feature.subclass_label is not None:
                subclass_labels.append(feature.subclass_label)
        return subclass_labels

    def classify_unknown(self, unk_ciu_obj, params_obj, unk_label='Unknown'):
        """
        Classify a test dataset according to this classification scheme. Selects features from
        the test dataset (ciudata), classifies, and returns output and score metrics.
        :param unk_ciu_obj: CIUAnalysisObj containing the ciu_data from unknown to be fitted
        :type unk_ciu_obj: CIUAnalysisObj
        :param unk_label: label for the unknown data
        :param params_obj: parameters information
        :type params_obj: Parameters
        :return: updated analysis object with prediction data saved
        :rtype: CIUAnalysisObj
        """
        feature_cvs = [x.cv for x in self.selected_features]
        unk_ciudata = get_classif_data(unk_ciu_obj, params_obj, ufs_mode=False, num_gauss_override=self.num_gaussians, selected_cvs=feature_cvs)

        # Assemble feature data for fitting
        unk_input_x, fake_labels, filenames, input_feats = arrange_data_for_lda([unk_ciudata], unk_label, self.selected_features, [unk_ciu_obj.axes[1]], [unk_ciu_obj.short_filename])

        # Fit/classify data according to scheme LDA and classifier
        unknown_transformed_lda = self.lda.transform(unk_input_x)
        pred_class_label = self.classifier.predict(unknown_transformed_lda)
        pred_probs_by_cv = self.classifier.predict_proba(unknown_transformed_lda)
        pred_probs_avg = np.average(pred_probs_by_cv, axis=0)

        # create plots and save information to object
        unk_ciu_obj.classif_predicted_label = pred_class_label
        unk_ciu_obj.classif_probs_by_cv = pred_probs_by_cv
        unk_ciu_obj.classif_probs_avg = pred_probs_avg
        unk_ciu_obj.classif_transformed_data = unknown_transformed_lda

        return unk_ciu_obj

    def classify_unknown_subclass(self, unk_subclass_obj, params_obj):
        """
        Classify a test dataset according to this classification scheme with subclasses. Selects features from
        the test dataset (ciudata), classifies, and returns output and score metrics. Each container has only 1 (or more) replicates of a particular unknown rather than a shaped list by known classes
        :param unk_subclass_obj: A single subclass unknown container with all subclasses/replicates for a single analysis
        :type unk_subclass_obj: SubclassUnknown
        :param params_obj: parameters information
        :type params_obj: Parameters
        :return: updated analysis object with prediction data saved
        :rtype: SubclassUnknown
        """
        # convert subclass object to standard ClassifInput
        list_classif_input = unk_subclass_dict_to_list(unk_subclass_obj)

        # Assemble feature data for fitting
        unk_input_x, label_data, filenames, cvs, file_ids = [], [], [], [], []
        for feature in self.selected_features:
            unk_input_x, label_data, filenames, cvs, file_ids = lda_prep_subclass(feature, list_classif_input, params_obj, unk_input_x,
                                                                                  label_data, filenames, cvs, file_ids)

        # Fit/classify data according to scheme LDA and classifier
        unknown_transformed_lda = self.lda.transform(unk_input_x)
        pred_class_label = self.classifier.predict(unknown_transformed_lda)
        pred_probs_by_cv = self.classifier.predict_proba(unknown_transformed_lda)
        pred_probs_avg = np.average(pred_probs_by_cv, axis=0)

        # save information to original container and return for plotting/etc
        unk_subclass_obj.predicted_label = pred_class_label
        unk_subclass_obj.probs_by_cv = pred_probs_by_cv
        unk_subclass_obj.probs_avg = pred_probs_avg
        unk_subclass_obj.transformed_data = unknown_transformed_lda
        return unk_subclass_obj

    def plot_all_unknowns(self, unk_ciuobj_list, params_obj, output_path):
        """
        Same as classify unknown, except that all unknowns are plotted on a single output plot
        and labeled by filename
        :param unk_ciuobj_list: list of CIUAnalysis containers for each unknown - MUST have transformed data already set
        :type unk_ciuobj_list: list[CIUAnalysisObj]
        :param params_obj: parameters information
        :type params_obj: Parameters
        :param output_path: directory in which to save output
        :return: void
        """
        all_plot_tups = [(x.classif_transformed_data, x.short_filename) for x in unk_ciuobj_list]
        plot_classification_decision_regions(self, params_obj, output_path, unknown_tups=all_plot_tups)

    def plot_all_unknowns_subclass(self, unk_subclass_obj_list, params_obj, output_path):
        """
        Same as classify unknown, except that all unknowns are plotted on a single output plot
        and labeled by filename
        :param unk_subclass_obj_list: list of SubclassUnknown containers for each unknown - MUST have transformed data already set
        :type unk_subclass_obj_list: list[SubclassUnknown]
        :param params_obj: parameters information
        :type params_obj: Parameters
        :param output_path: directory in which to save output
        :return: void
        """
        all_plot_tups = [(x.transformed_data, x.label) for x in unk_subclass_obj_list]
        plot_classification_decision_regions(self, params_obj, output_path, unknown_tups=all_plot_tups)


class SubclassUnknown(object):
    """
    Container for unknowns in subclass analyses. Essentially just a dictionary of {subclass label: [list of analysis objects]}
    for each subclass with associated name and space to save output/classification predictions/probabilities
    """
    def __init__(self, label, dict_unk_subclasses):
        """
        Initialize new container. NOTE: if no subclasses, set entry key = '0'
        :param label: string used to label outputs from this combined unknown
        :param dict_unk_subclasses: dict of key = subclass label, value = list of CIUAnalysisObjs
        :type dict_unk_subclasses: dict[str, list[CIUAnalysisObj]]
        """
        self.label = label
        self.subclass_dict = dict_unk_subclasses

        # output storage
        self.predicted_label = None
        self.probs_by_cv = None
        self.probs_avg = None
        self.transformed_data = None

    def get_subclass_data_list(self, subclass_label):
        """
        Return the list of CIUAnalysisObj's corresponding to the provided subclass label string
        :param subclass_label: string to search for in the dict
        :return: list of analysis objs for this subclass
        :rtype: list[CIUAnalysisObj]
        """
        try:
            return self.subclass_dict[subclass_label]
        except KeyError:
            return None

    def __str__(self):
        """
        string rep
        :return: string
        """
        return '<SubclassUnknown> {}'.format(self.label)


class CFeature(object):
    """
    Container for classification feature information in feature selection.
    """
    def __init__(self, cv, cv_index, mean_score, std_dev_score, subclass_label=None):
        self.cv = cv
        self.cv_index = cv_index
        # self.score_list = score_list
        self.mean_score = mean_score
        self.std_dev_score = std_dev_score
        self.subclass_label = subclass_label

    def __str__(self):
        if self.subclass_label is not None:
            return '<CFeature> cv: {}, sub: {}, score: {:.2f}'.format(self.cv, self.subclass_label, self.mean_score)
        else:
            return '<CFeature> cv: {}, score: {:.2f}'.format(self.cv, self.mean_score)
    __repr__ = __str__


class CrossValRun(object):
    """
    Method to generate combinations of training and test datasets and run LDA + classification to generate training
    score and test score on selected features
    """
    def __init__(self, data_list_by_label, label_list, training_size, features):
        """
        Initialize a CrossValProduct method with data, label, training size, and features
        :param data_list_by_label: formatted data (sorted by feats, subclasses handled, etc) list for all classes, sorted by class label
        :type data_list_by_label: list[list[DataSubset]]
        :param label_list: label list for data
        :param training_size: num of ciu data in each class to be used as training set
        :param features: cv features
        :type features: list[CFeature]
        """
        self.data_list_by_label = data_list_by_label
        self.label_list = label_list
        self.training_size = training_size
        self.features = features

        # assemble training and test data and label
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None

        # results info
        self.train_scores = []
        self.train_scores_mean = None
        self.train_scores_std = None
        self.test_scores = []
        self.test_scores_mean = None
        self.test_scores_std = None
        self.probs = []
        self.probs_mean = None
        self.probs_std = None

        self.all_classes_combo_lists = []   # assembled data (combinations within one class)
        self.all_products = []      # assembled data (products of all combinations across all classes)

    def divide_data_and_run_lda(self):
        """
        Divide the data into each possible training and test arrangement and run LDA/SVC classification
        to assess accuracy
        :return: void (set results to self)
        """
        shaped_label_list = []
        for index, label in enumerate(self.label_list):
            shaped_label_list.append([label for _ in range(len(self.data_list_by_label[index]))])

        # generate all combinations of each class input datalist
        for class_index, class_subset_list in enumerate(self.data_list_by_label):
            class_traintest_combo_list = []
            # create all combinations of training and test data
            for training_subset_list in itertools.combinations(class_subset_list, self.training_size):
                # test data is anything not in the training list
                test_subset_list = [x for x in class_subset_list if x not in training_subset_list]

                # assemble LDA data here for train_subsets, train_labels (and test_subsets, test_labels)
                train_lda_data, train_lda_labels, train_string_labels = arrange_lda_new(training_subset_list)
                test_lda_data, test_lda_labels, test_string_labels = arrange_lda_new(test_subset_list)

                # initialize container to hold this combination of datasets for eventual product against other class(es)
                data_combo = CrossValData(train_lda_data, train_lda_labels, train_string_labels, test_lda_data, test_lda_labels, test_string_labels)
                class_traintest_combo_list.append(data_combo)
            self.all_classes_combo_lists.append(class_traintest_combo_list)

    def assemble_class_products(self):
        """
        Take divided train/test data combinations from self.divide_data and generate all products between
        classes. (e.g. for 2 classes, 3 replicates each, divide data assigns reps 1, 2 to train and 3 to test for
        one class (and all other combinations), and this method takes class 1, train 1 2 * class 2 train 1 2 to generate
        all combinations of training and test subsets for all classes).
        :return: void (saves to self.all_products)
        """
        for class_combo_pair in itertools.product(*self.all_classes_combo_lists):
            final_train_data, final_train_labels = [], []
            final_test_data, final_test_labels = [], []

            # assemble training and test data into a single matrix (each) with a corresponding label matrix for each
            for crossval_data in class_combo_pair:
                # crossval_data contains training and test data, each of which may have multiple columns. Combined into single matrix each
                for index in range(len(crossval_data.train_data)):
                    final_train_data.append(crossval_data.train_data[index])
                    final_train_labels.append(crossval_data.train_labels_num[index])
                for index in range(len(crossval_data.test_data)):
                    final_test_data.append(crossval_data.test_data[index])
                    final_test_labels.append(crossval_data.test_labels_num[index])

            # Run LDA/SVC crossvalidation using the assembled data
            svc, lda = run_lda_svc(final_train_data, final_train_labels)

            # Generate an accuracy score for the training and test data against the classifier and save
            train_lda = lda.transform(final_train_data)
            test_lda = lda.transform(final_test_data)
            train_score = svc.score(train_lda, final_train_labels)
            test_score = svc.score(test_lda, final_test_labels)
            self.train_scores.append(train_score)
            self.test_scores.append(test_score)

        # Save final scores and means to container
        self.train_scores_mean = np.mean(self.train_scores)
        self.train_scores_std = np.std(self.train_scores)
        self.test_scores_mean = np.mean(self.test_scores)
        self.test_scores_std = np.std(self.test_scores)
        self.probs_mean = np.mean(self.probs)
        self.probs_std = np.std(self.probs)


class DataSubset(object):
    """
    Container for a subset of CIU data selected from selected features.
    """
    def __init__(self, data, class_label, numeric_label, file_id, features):
        """
        Initialize new subset container
        :param data: dataset arranged using rearrange CIU by feats method (2D numpy array)
        :param class_label: class label
        :param numeric_label: (int) numeric label corresponding to the class label of this subset
        :param file_id: (string) file identifier (CIUAnalysisObj.short_filename). In subclass mode, this should be a combination of filenames for each subclass obj.
        :param features: list of features used in arranging data
        :type features: list[CFeature]
        """
        self.data = data
        self.class_label = class_label
        self.numeric_label = numeric_label
        self.features = features
        self.file_id = file_id


class CrossValProduct(object):
    """
    Method to generate combinations of training and test datasets and run LDA + classification to generate training
    score and test score on selected features
    """
    def __init__(self, data_list_by_label, label_list, training_size, features):
        """
        Initialize a CrossValProduct method with data, label, training size, and features
        :param data_list_by_label: ciu_data list for all classes, sorted by class label
        :param label_list: label list for data
        :param training_size: num of ciu data in each class to be used as training set
        :param features: cv features
        """
        self.data_list_by_label = data_list_by_label
        self.label_list = label_list
        self.training_size = training_size
        self.features = features

        # assemble training and test data and label
        self.train_data = None
        self.train_label = None
        self.test_data = None
        self.test_label = None

        # results info
        self.train_scores = []
        self.train_scores_mean = None
        self.train_scores_std = None
        self.test_scores = []
        self.test_scores_mean = None
        self.test_scores_std = None
        self.probabs = []

        # assembled data
        self.all_class_combination_lists = None
        self.all_products = None

    def assemble_class_combinations(self, params_obj):
        """
        Generate training and test datasets with labels with the given training size
        :param params_obj: parameters information
        :type params_obj: Parameters
        :return: List of class combination lists for all classes. Each class combo list is a list of DataCombination
        objects. Intended to go directly into itertools.product
        :rtype: list[list[DataCombination]]
        """
        shaped_label_list = []
        for index, label in enumerate(self.label_list):
            shaped_label_list.append([label for _ in range(len(self.data_list_by_label[index]))])

        # generate all combinations of each class input datalist (products)
        all_class_combo_lists = []
        for class_index, class_ciu_list in enumerate(self.data_list_by_label):
            # create all combinations of specified training/test sizes from this class's data
            class_combo_list = []

            # create all combinations of training and test data
            for training_data_tuple, training_label_tuple in zip(itertools.combinations(class_ciu_list, self.training_size),
                                                                 itertools.combinations(shaped_label_list[class_index], self.training_size)):
                # training_data_list = [x.ciu_data for x in training_data_tuple]
                training_data_list = [get_classif_data(x, params_obj) for x in training_data_tuple]
                training_cv_axes = [x.axes[1] for x in training_data_tuple]

                test_data_list = [x for x in class_ciu_list if x not in training_data_tuple]
                test_cv_axes = [x.axes[1] for x in test_data_list]
                test_data_list = [get_classif_data(x, params_obj) for x in test_data_list]
                # [self.test_labels_tup[test_index] for _ in range(len((test_data[0])))]
                test_label_list = [shaped_label_list[class_index][0] for _ in range(len(test_data_list))]

                # create Train/Test DataCombo for this combination
                current_combo = DataCombination(training_data_list, training_label_tuple, training_cv_axes, test_data_list, test_label_list, test_cv_axes)
                current_combo.prepare_data(self.features)
                class_combo_list.append(current_combo)

            all_class_combo_lists.append(class_combo_list)
        return all_class_combo_lists


class CrossValData(object):
    """
    Container object for holding training and test data for a single way of combining the datasets in a class.
    Training and test datasets can be of arbitrary size. Holds corresponding label arrays of same shape as
    train/test data.
    """
    def __init__(self, train_data, train_num_labels, train_str_labels, test_data, test_num_labels, test_str_labels):
        """
        Initialize new container
        :param train_data: list containing ciu datasets for training
        :param train_num_labels: numeric labels (ints) list, same length as train data
        :param train_str_labels: same length list as train_data with labels (strings)
        :param test_data: list containing ciu datasets for testing
        :param test_num_labels: numeric labels (ints) list, same length as test data
        :param test_str_labels: same length list as test_data with labels (strings)
        """
        self.train_data = train_data
        self.train_labels_num = train_num_labels
        self.train_labels_str = train_str_labels
        self.test_data = test_data
        self.test_labels_num = test_num_labels
        self.test_labels_str = test_str_labels


class DataCombination(object):
    """
    Container object for holding training and test data for a single way of combining the datasets in a class.
    Training and test datasets can be of arbitrary size. Holds corresponding label arrays of same shape as
    train/test data.
    """
    def __init__(self, train_data, train_labels, train_cv_axes, test_data, test_labels, test_cv_axes):
        """

        :param train_data: n-length tuple containing ciu datasets for training
        :param train_labels: same shape tuple with labels
        :param train_cv_axes: CV axes for all ciu training datasets (same shape as train_data)
        :param test_data: n-length tuple containing ciu datasets for testing
        :param test_labels: same shape tuple with labels
        :param test_cv_axes: CV axes for all ciu test datasets (same shape as test_data)
        """
        self.training_data_tup = train_data
        self.training_labels_tup = train_labels
        self.training_cv_axes = train_cv_axes
        self.test_data_tup = test_data
        self.test_labels_tup = test_labels
        self.test_cv_axes = test_cv_axes

        self.training_data_final = []
        self.training_labels_string = []
        self.test_data_final = []
        self.test_labels_string = []

    def prepare_data(self, features_list):
        """
        Assemble concatenated data and label arrays for the specified slices of the input data (CV columns/features)
        :param features_list: list of selected Features with cv data
        :return: void
        """
        train_data_final, train_labels_final, empty_filenames, final_cvs = arrange_data_for_lda(self.training_data_tup, self.training_labels_tup, features_list, self.training_cv_axes)
        test_data_final, test_labels_final, empty_filenames2, final_cvs = arrange_data_for_lda(self.test_data_tup, self.test_labels_tup, features_list, self.test_cv_axes)

        self.training_data_final = train_data_final
        self.training_labels_string = train_labels_final
        self.test_data_final = test_data_final
        self.test_labels_string = test_labels_final


class UFSResult(object):
    """
    Container for label and product information for data combinations used in feature selection.
    """
    def __init__(self, data_list, label_list):
        """
        Initialize a new UFSResult with associated label and data
        :param data_list: list of ciu_data matrices (normalized)
        :param label_list: list of class labels (same length as data_list)
        """
        self.data_input = data_list
        self.labels_input = label_list

        self.combined_data = None
        self.combined_label_arr = None
        self.numeric_label_arr = None

        # results information
        self.fit_scores = None
        self.fit_pvalues = None
        self.fit_sc = None

        self.train_scores = None
        self.test_scores = None

        # run data preparation
        self.prepare_data()

    def prepare_data(self):
        """
        Manipulate data into concatenated arrays suitable for input into feature selection algorithm.
        :return: void
        """
        self.combined_data = np.concatenate(self.data_input)

        label_array_list = []
        index = 0
        for label in self.labels_input:
            label_array = [label for _ in range(len(self.data_input[index]))]
            label_array_list.append(label_array)
            index += 1
        self.combined_label_arr = np.concatenate(label_array_list)

        # use the label encoder to generate a numeric label list (could just do this manually??)
        # output is just the class numbers in an array the shape of the input
        encoder = LabelEncoder()
        label_list = encoder.fit(self.combined_label_arr)
        self.numeric_label_arr = label_list.transform(self.combined_label_arr) + 1

    def __str__(self):
        label_string = ','.join(self.labels_input)
        return '<UFSResult> labels: {}'.format(label_string)
    __repr__ = __str__


# if __name__ == '__main__':
    # Read the data
    # import tkinter
    # from tkinter import filedialog
    # from tkinter import simpledialog
    # import CIU_Params
    # import Raw_Processing

    #
    # root = tkinter.Tk()
    # root.withdraw()

    # num_classes = simpledialog.askinteger('Class Number', 'Into how many classes do you want to group?')
    # num_classes = 4
    # data_labels = []
    # obj_list_by_label = []
    # # main_dir = 'C:\\'
    # main_dir = r"C:\Users\dpolasky\Desktop\CIUSuite2\CIU2 test data\Classify_Sarah\apo_cdl_pi_3way"

    # class_labels = ['Igg1', 'Igg2', 'Igg4', 'Igg4']
    # f1 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG1_1.ciu'
    # f2 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG1_2.ciu'
    # f3 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG1_3.ciu'
    # f4 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG2_1.ciu'
    # f5 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG2_2.ciu'
    # f6 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG2_3.ciu'
    # f7 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG3_1.ciu'
    # f8 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG3_2.ciu'
    # f9 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG3_3.ciu'
    # f10 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG4_1.ciu'
    # f11 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG4_2.ciu'
    # f12 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG4_3.ciu'

    # class_labels = ['cdl', 'pi', 'apo']
    # class1_files = [os.path.join(main_dir, x) for x in os.listdir(main_dir) if x.endswith('.ciu') and class_labels[0] in x.lower()]
    # class2_files = [os.path.join(main_dir, x) for x in os.listdir(main_dir) if x.endswith('.ciu') and class_labels[1] in x.lower()]
    # class3_files = [os.path.join(main_dir, x) for x in os.listdir(main_dir) if x.endswith('.ciu') and os.path.join(main_dir, x) not in class1_files and os.path.join(main_dir, x) not in class2_files]
    # fs = [class1_files, class2_files, class3_files]
    # f_class1 = [f1, f2, f3]
    # f_class2 = [f4, f5, f6]
    # f_class3 = [f7, f8, f9]
    # f_class4 = [f10, f11, f12]
    # fs = [f_class1, f_class2, f_class4, f_class4]
    # fs= [f_class1, f_class2]

    # sarahfile = r"C:\Users\sugyan\Documents\CIUSuite\Classification\IgGdata\Iggs_datalist.csv"
    # files = np.genfromtxt(sarahfile, skip_header=1, delimiter=',', dtype='str')
    # class_sarahfiles = np.unique(files[:, 0])
    # # CDL, PA, PC, PE, PG, PI, PPIX, PS [in order]
    # filelist = [[] for i in range(len(class_sarahfiles))]
    # for index, classtype in enumerate(files[:, 0]):
    #     for class_ind, class_unique in enumerate(class_sarahfiles):
    #         if class_unique == classtype:
    #             filelist[class_ind].append((files[:, 1][index]))
    #
    # class_labels = [class_sarahfiles[0], class_sarahfiles[1], class_sarahfiles[3], class_sarahfiles[3]]
    # fs = [filelist[0], filelist[1], filelist[3], filelist[3]]
    #
    # for class_index in range(0, num_classes):
    #     # Read in the .CIU files and labels for each class
    #     # label = simpledialog.askstring('Class Name', 'What is the name of this class?')
    #     class_label = class_labels[class_index]
    #     # files = filedialog.askopenfilenames(filetypes=[('CIU', '.ciu')])
    #     files = fs[class_index]
    #     main_dir = os.path.dirname(files[0])
    #
    #     obj_list = []
    #     for file in files:
    #         with open(file, 'rb') as analysis_file:
    #             obj = pickle.load(analysis_file)
    #         obj_list.append(obj)
    #     data_labels.append(class_label)
    #     obj_list_by_label.append(obj_list)
    #
    # unkdata_name = open(r"C:\Users\sugyan\Documents\CIUSuite\Classification\IgGdata\IgG1_3.ciu", 'rb')
    # unkdata_ = pickle.load(unkdata_name)
    #
    # params = CIU_Params.Parameters()
    # params.set_params(CIU_Params.parse_params_file(CIU_Params.hard_descripts_file))
    # obj_list_by_label, equalized_axes_list = Raw_Processing.equalize_axes_2d_list(obj_list_by_label)
    #
    # scheme = main_build_classification(data_labels, obj_list_by_label, params, main_dir)
    # scheme.classify_unknown(unkdata_, params, main_dir, unk_label='Unknown')

    # featurescaling_lda(data_labels, obj_list_by_label, main_dir)
    # class_comparison_lda(data_labels, obj_list_by_label, main_dir)
    # univariate_feature_selection_datacv(data_labels, obj_list_by_label, main_dir)
    # univariate_feature_selection_datacv_withprediction(data_labels, obj_list_by_label, main_dir)
    # univariate_feature_selection_datacv_lda_withprediction(data_labels, obj_list_by_label, main_dir)

    # univariate_feature_selection(data_labels, obj_list_by_label)
    # output_scheme = main_build_classification(data_labels, obj_list_by_label, main_dir)
    # univariate_feature_selection_datacv_runldaonfeats_suggie(data_labels, obj_list_by_label, main_dir)
