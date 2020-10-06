import os

import numpy as np
#import pandas as pd

from Utils import Utils


class DataLoader:
    def preprocess_data_from_csv(self, dataset_path, split_size):
        print(".. Data Loading ..")

        # data load
        np_arr = np.loadtxt(dataset_path)
        np_X = np_arr[:, :9]
        np_Y = np_arr[:, 9]
        print("ps_np_covariates_X: {0}".format(np_X.shape))
        print("ps_np_treatment_Y: {0}".format(np_Y.shape))

        np_X_train, np_X_test, np_Y_train, np_Y_test = \
            Utils.test_train_split(np_X, np_Y, split_size)
        print("np_covariates_X_train: {0}".format(np_X_train.shape))
        print("np_covariates_Y_train: {0}".format(np_Y_train.shape))

        print("np_covariates_X_test: {0}".format(np_X_test.shape))
        print("np_covariates_Y_test: {0}".format(np_Y_test.shape))
        return np_X_train, np_X_test, np_Y_train, np_Y_test

    def preprocess_data_from_csv_multi(self, dataset_path, split_size):
        print(".. Data Loading ..")

        # data load
        np_arr = np.loadtxt(dataset_path)
        np.random.shuffle(np_arr)
        np_X = np_arr[:, :9]
        np_Y = np_arr[:, 9:]

        print("ps_np_covariates_X: {0}".format(np_X.shape))
        print("ps_np_treatment_Y: {0}".format(np_Y.shape))

        np_X_train, np_X_test, np_Y_train, np_Y_test = \
            Utils.test_train_split(np_X, np_Y, split_size)

        print("np_covariates_X_train: {0}".format(np_X_train.shape))
        print("np_covariates_Y_train: {0}".format(np_Y_train.shape))

        print("np_covariates_X_test: {0}".format(np_X_test.shape))
        print("np_covariates_Y_test: {0}".format(np_Y_test.shape))

        return np_X_train, np_X_test, np_Y_train, np_Y_test