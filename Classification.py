"""
Module for classification schemes for CIU data groups
Authors: Dan Polasky, Sugyan Dixit
Date: 1/11/2018
"""

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from matplotlib.backends.backend_pdf import PdfPages as pdfpage
from sklearn.metrics import accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.linear_model import LogisticRegression


def class_comparison_lda(labels, analysis_obj_list_by_label, output_dir):
    """
    Modularized version of class comparision algorithm from Suggie. Takes a list of labels and a
    list of AnalysisObjs for each label and performs a 2-stage LDA based classification. Output
    is saved with the voltages of greatest difference and their scores for assay development.
    :param labels = list of strings corresponding to class labels
    :param analysis_obj_list_by_label: list of lists of CIUAnalysisObj's. Same length as labels list, with
    arbitrary number of Analysis objects for each label.
    :param output_dir: directory in which to save output
    :return: void
    """
    # CVs for plot titles later
    cv1 = analysis_obj_list_by_label[0][0].axes[1]

    inarray_list = []
    inlabel_list = []
    label_index = 0
    for label in labels:
        # Get the list of datasets corresponding to this label
        label_obj_list = analysis_obj_list_by_label[label_index]
        data_index = 1

        for analysis_obj in label_obj_list:
            # Add the formatted label and dataset to their input lists for the feature selection
            inlabel_list.append('{}_{}'.format(label, data_index))
            ciu_data = analysis_obj.ciu_data
            inarray_list.append(ciu_data.T)     # append transposed CIU data matrix
            data_index += 1
        label_index += 1
    run_lda_raw(inarray_list, inlabel_list, labels, output_dir, cv1)


def classification_lda_maxvals(labels, analysis_obj_list_by_label, output_dir):
    """
    Analogous to classification_lda_raw, except uses only the max value in each CV column (set to 1,
    with all other DTs set to 0) as the input data.
    :param labels = list of strings corresponding to class labels
    :param analysis_obj_list_by_label: list of lists of CIUAnalysisObj's. Same length as labels list, with
    arbitrary number of Analysis objects for each label.
    :param output_dir: directory in which to save output
    :return: void
    """
    cv1 = analysis_obj_list_by_label[0][0].axes[1]

    inarray_list = []
    inlabel_list = []
    label_index = 0
    for label in labels:
        # Get the list of datasets corresponding to this label
        label_obj_list = analysis_obj_list_by_label[label_index]
        data_index = 1

        for analysis_obj in label_obj_list:
            # Add the formatted label and dataset to their input lists for the feature selection
            inlabel_list.append('{}_{}'.format(label, data_index))
            ciu_data_col = analysis_obj.ciu_data.T

            # reduce ciu_data to max values only (normalized to 1.0)
            new_ciu = np.asarray([[np.argmax(col)] for col in ciu_data_col])
            # new_ciu = np.zeros(np.shape(ciu_data_col))
            # for i in range(len(ciu_data_col)):
            #     for j in range(len(ciu_data_col[0])):
            #         if ciu_data_col[i][j] == np.max(ciu_data_col[i]):
            #             new_ciu[i][j] = 1.0

            # inarray_list.append(ciu_data_col)  # append transposed CIU data matrix
            inarray_list.append(new_ciu)
            data_index += 1
        label_index += 1
    run_lda_raw(inarray_list, inlabel_list, labels, output_dir, cv1)


def run_lda_raw(inarray_list, inlabel_list, label_list, output_dir, cv_labels):
    # create the dictionary needed to create the data and target array
    dict_init = createdict(inarray_list, inlabel_list)  # dictionary
    X_ = transformrawdataforclf.transformdata_rawdata(dict_init)  # data
    y_, target_label = transformrawdataforclf.createtargetarray_rawdata(dict_init)  # target

    print(np.shape(X_))

    # initiate LDA from sklearn
    # lda = LDA(solver='eigen', n_components=5)
    lda = LDA(solver='svd', n_components=2)


    # initiate feature selection from mlxtend package
    # 4 different algorithms are available. See
    # rabst.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector
    # to toggle between different selection algorithms, change the bool values for forward and floating attributes
    # below is set up for all 4 feature selection module

    # SFFS and SBFS is computationally more expensive because it either excludes after inclusion or includes
    # after exclusion to maximize the score

    # Sequential Forward Selection (sfs)
    # sfs = SFS(lda, k_features=(1, np.shape(X_)[1]), forward=True, floating=False, scoring='accuracy', cv=5, n_jobs=-1)
    sfs = SFS(lda, k_features=(1, np.shape(X_)[1]), forward=True, floating=False, scoring='accuracy', cv=0, n_jobs=-1)

    # Sequential Backward Selection (sbs)
    # sfs = SFS(lda, k_features = (1, np.shape(X_)[1]), forward = False, floating = False, scoring = 'accuracy', cv = 5, n_jobs = -1)

    # Sequential Forward Floating Selection (SFFS)
    # sfs = SFS(lda, k_features = (1, np.shape(X_)[1]), forward = True, floating = True, scoring = 'accuracy', cv = 5, n_jobs = -1)

    # Sequential Backward Floating Selection (SBFS)
    # sfs = SFS(lda, k_features = (1, np.shape(X_)[1]), forward = False, floating = True, scoring = 'accuracy', cv = 5, n_jobs = -1)

    ############

    # fit the feature selection for the data
    sfs.fit(X_, y_)

    print('best combination (ACCURACY: %.3f): %s\n' % (sfs.k_score_, sfs.k_feature_idx_))
    # print ('\nAll Subsets:\n', sfs.subsets_)

    labels_name = 'LDA_' + '_'.join(label_list) + '_'
    output_path = os.path.join(output_dir, labels_name)

    # write the avg score and features in a text file for reference
    write_features_scores(dict=sfs.subsets_, fname=output_path + 'avg_score_features_sfs_unevenclassnum.txt')

    # plot the accuracy vs features using plot_sfs module from mlxtend
    plot_sfs(sfs.get_metric_dict(), kind='std_err')
    plt.savefig(output_path + 'performance_vs_features_sfs.pdf')
    plt.close()

    # with the best k features from the feature selection algorithm, we now compare the raw data for each k feature

    # create pdf to store figures of lda transformed data for each feature
    pdf_fig = pdfpage(output_path + 'ldatransform_features_sfs.pdf')

    # create X and y input array
    for i in sfs.k_feature_idx_:
        arr = []
        for k in inarray_list:
            arr.append(k[i])
        arr = np.array(arr)  # X array
        y_, target_label = transformrawdataforclf.createtargetarray_featureselect(inlabel_list)  # target array

        # initiate LDA. The solver is now changed to SVD from eigen. For some reason, scikitlearn cannot compute eigen values. Not sure exactly what causes it. There are some posts online refering to internal bug. Anyway, svd creates very similar results. I personally like eigen solver because I understand that better.
        # n_components is n_classes - 1. Since we have only 2 classes, we'll only obtain 1 component. However, with classes > 2 its necessary to evaluate the explained variance ratio for each component in order to properly assess the transformation.
        lda2 = LDA(solver='svd', n_components=5)

        # fit LDA
        lda2.fit(arr, y_)
        print('Explained_variance_ratio: ',
              lda2.explained_variance_ratio_)  # important to evaluate, especially when more than one LDs.
        # transform the dataset with lda
        x_lda = lda2.transform(arr)

        # evaluate score. This is a harsh metric especially for multi class classification. The performance of such
        # transformation and classification needs to be evaluated in better ways, for example, creating training and
        # test datasets, cross-validation, assessing other performance metrics etc. Will be working on that soon.
        print('Score: ', lda2.score(arr, y_))

        if len(label_list) == 2:
            plot_sklearn_lda_1ld(x_lda, y_, target_label)
        elif len(label_list) == 3:
            lr = LogisticRegression()
            lr = lr.fit(x_lda, y_)

            plot_decision_regions(x_lda, y_, target_label, classifier=lr)
        else:
            lr = LogisticRegression()
            lr = lr.fit(x_lda, y_)

            plot_decision_regions(x_lda, y_, target_label, classifier=lr)
        plt.title(str(cv_labels[i]))
        pdf_fig.savefig()
        plt.close()

    pdf_fig.close()


class transformrawdataforclf():

    def transformdata_rawdata(dict):
        arr = np.hstack((dict.values())).T
        return arr

    def createtargetarray_rawdata(dict):
        arr = []
        for i, (keys, values) in enumerate(zip(dict.keys(), dict.values())):
            arr.append(np.repeat(str(keys).split('_')[0], len(values[0])))
        arr = np.concatenate(arr)
        enc = LabelEncoder()
        label = enc.fit(arr)
        arr = label.transform(arr) + 1
        arr_decode = label.inverse_transform((arr - 1))
        return arr, arr_decode

    def createtargetarray_featureselect(inputlabel):
        arr = []
        for i in inputlabel:
            arr.append(i.split('_')[0])
        enc = LabelEncoder()
        label = enc.fit(arr)
        arr = label.transform(arr) + 1
        arr_decode = label.inverse_transform((arr - 1))
        return arr, arr_decode


def createdict(list, labels):
    dict = {e: list[i] for i, e in enumerate(labels)}
    return dict


def curatedata_bestfeature_combined(array, feature_idx):
    newarr = []
    for i in feature_idx:
        newarr.append(array[i][:])
    return newarr


def write_features_scores(dict, fname='avg_score_features_sfs.txt'):
    with open(fname, 'w') as f:
        for key, val in sorted(dict.items()):
            feature_idx = val['feature_idx']
            avg_score = val['avg_score']
            f.write(str(avg_score) + '    ' + str(feature_idx) + '\n')
    f.close()


def plot_sklearn_lda_3ld(X, y, target_label, title='LDA: Projection of first 3 LDs'):
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=9, azim=142)
    for label, marker, color in zip(range(0, len(np.unique(target_label))), ('^', 's', 'o', 'd'),
                                    ('blue', 'red', 'green', 'black')):
        ax.scatter(xs=X[:, 0][y == label + 1], ys=X[:, 1][y == label + 1], zs=X[:, 2][y == label + 1], color=color,
                   edgecolor='k', marker=marker, s=40, label=np.unique(target_label)[label])
    ax.set_xlabel('LD1')
    ax.set_ylabel('LD2')
    ax.set_zlabel('LD3')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)


def plot_sklearn_lda_2ld(X, y, target_label, title='LDA: Projection of first 2 LDs'):
    ax = plt.subplot(111)
    for label, marker, color in zip(range(0, len(np.unique(target_label))), ('^', 's', 'o', 'd'),
                                    ('blue', 'red', 'green', 'black')):
        plt.scatter(x=X[:, 0][y == label + 1], y=X[:, 1][y == label + 1], marker=marker, color=color, alpha=0.5,
                    label=np.unique(target_label)[label])

    plt.xlabel('LD1')
    plt.ylabel('LD2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.grid()


def plot_sklearn_lda_1ld(X, y, target_label):
    for label, marker, color in zip(range(0, len(np.unique(target_label))), ('^', 's', 'o', 'd', 'x'),
                                    ('red', 'blue', 'green', 'gray', 'cyan')):
        plt.scatter(X[:, 0][y == label + 1], np.zeros(np.shape(X[:, 0][y == label + 1])), marker=marker, color=color,
                    alpha=0.5, label=np.unique(target_label)[label])
    plt.legend(loc='best')
    plt.xlabel('LD1')
    plt.ylim((1, -1))


def plot_decision_regions(X, y, target_label, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ['blue', 'red', 'lightgreen', 'gray', 'cyan']
    # cmap = ListedColormap(['blue', 'red', 'lightgreen'])
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    x2_min, x2_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx],
                    label=np.unique(target_label)[idx])

    plt.legend(loc='best', fontsize='small')


def load_analysis_obj(analysis_filename):
    """
    Load a pickled analysis object back into program memory
    :param analysis_filename: full path to file location to load
    :return: CIUAnalysisObj
    """
    with open(analysis_filename, 'rb') as analysis_file:
        analysis_obj = pickle.load(analysis_file)
    return analysis_obj


if __name__ == '__main__':
    # Read the data
    import tkinter
    from tkinter import filedialog
    from tkinter import simpledialog
    root = tkinter.Tk()
    root.withdraw()

    num_classes = simpledialog.askinteger('Class Number', 'Into how many classes do you want to group?')

    data_labels = []
    obj_list_by_label = []
    main_dir = 'C:\\'
    for index in range(0, num_classes):
        # Read in the .CIU files and labels for each class
        label = simpledialog.askstring('Class Name', 'What is the name of this class?')
        files = filedialog.askopenfilenames(filetypes=[('CIU', '.ciu')])
        main_dir = os.path.dirname(files[0])    # TODO: adjust to actual obj outputdir once synched

        obj_list = []
        for file in files:
            with open(file, 'rb') as analysis_file:
                obj = pickle.load(analysis_file)
            obj_list.append(obj)
        data_labels.append(label)
        obj_list_by_label.append(obj_list)

    class_comparison_lda(data_labels, obj_list_by_label, main_dir)
    # classification_lda_maxvals(data_labels, obj_list_by_label, main_dir)
