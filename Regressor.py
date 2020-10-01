import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from Utils import Utils


class Regressor:
    def train_knn(self, np_x_train, np_y_train, k_list):
        knn_regressors = []

        for k_value in k_list:
            regressor = KNeighborsRegressor(n_neighbors=k_value)
            # Train the model
            regressor.fit(np_x_train, np_y_train)
            knn_regressors.append(regressor)

        # return the k_value and  corresponding regressor.
        return dict(zip(k_list, knn_regressors))

    def regression_using_knn(self, np_x_train, np_x_test, np_y_train, np_y_test):
        k_list = list(range(3, 100))
        y_pred_list = []
        accuracy_threshold_fixed = []
        accuracy_threshold_mean = []

        regressor_dict = self.train_knn(np_x_train, np_y_train, k_list)

        for k_value in regressor_dict:
            print("k value in multiclass knn -> " + str(k_value))
            knn_regressor = regressor_dict[k_value]
            y_pred = knn_regressor.predict(np_x_test) # 1311 x 9
            y_pred_list.append(y_pred)

            y_pred_fixed = np.where(y_pred > 0.5, 1, 0)
            y_pred_mean = np.where(y_pred > np.mean(y_pred), 1, 0)

            accuracy_fixed = Utils.get_accuracy_score(np_y_test, y_pred_fixed)
            accuracy_threshold_fixed.append(accuracy_fixed)
            print(accuracy_fixed)

            accuracy_mean = Utils.get_accuracy_score(np_y_test, y_pred_mean)
            accuracy_threshold_mean.append(accuracy_mean)
            print(accuracy_mean)
            print("---" * 5)

        # Get the k for which there is maximum accuracy
        best_k_fixed = k_list[accuracy_threshold_fixed.index(max(accuracy_threshold_fixed))]
        best_k_mean = k_list[accuracy_threshold_mean.index(max(accuracy_threshold_mean))]

        print("Optimal K for fixed threshold: {0}".format(best_k_fixed))
        print(accuracy_threshold_fixed[accuracy_threshold_fixed.index(max(accuracy_threshold_fixed))])

        print("Optimal K for mean as threshold: {0}".format(best_k_mean))
        print(accuracy_threshold_mean[accuracy_threshold_mean.index(max(accuracy_threshold_mean))])

    def regression_using_mlp(self, np_x_train, np_x_test, np_y_train, np_y_test):
        folds = KFold(n_splits=5, shuffle=True, random_state=1)
        param_grid = [
            {
                "learning_rate_init": [1e-04, 1e-05, 1e-02],
                'max_iter': [500, 1000],
                'hidden_layer_sizes': [
                    (200, 200, 9), (250, 250, 9), (100, 100, 9)
                ]
            }
        ]

        clf = GridSearchCV(MLPRegressor(random_state=1,
                                        activation='relu',
                                        solver='adam',
                                        alpha=1e-5,
                                        learning_rate_init=1e-04),
                           param_grid, cv=folds)
        clf.fit(np_x_train, np_y_train)
        best_score = clf.best_score_
        print(best_score)
        print("Best parameters set found on development set:")
        print(clf.best_params_)

        best_hyperparams = clf.best_params_
        # print(best_hyperparams)
        # clf = MLPRegressor(solver='adam', alpha=1e-5,
        #                    hidden_layer_sizes=(200, 200, 9),
        #                    random_state=1,
        #                    max_iter=1000)
        clf.fit(np_x_train, np_y_train)
        # y_pred = clf.predict(np_x_test)
        # y_pred_fixed = np.where(y_pred > 0.5, 1, 0)
        # y_pred_mean = np.where(y_pred > np.mean(y_pred), 1, 0)
        #
        # accuracy_fixed = Utils.get_accuracy_score(np_y_test, y_pred_fixed)
        # print(accuracy_fixed)
        #
        # accuracy_mean = Utils.get_accuracy_score(np_y_test, y_pred_mean)
        # print(accuracy_mean)

        # best_score = clf.best_score_
        # print("Best parameters set found on development set:")
        # print(clf.best_params_)
        #
        # best_hyperparams = clf.best_params_
        # print("The best test score is {0} corresponding to hyperparameters {1}"
        #       .format(best_score, best_hyperparams))
        # activation = best_hyperparams['activation']
        # hidden_layer_sizes = best_hyperparams['hidden_layer_sizes']
        #
        # final_clf = MLPRegressor(max_iter=100, activation=activation,
        #                           hidden_layer_sizes=hidden_layer_sizes)
        # final_clf.fit(np_x_train, np_y_train)
        #
        # y_pred = final_clf.predict(np_x_test)
        #
        # y_pred_fixed = np.where(y_pred > 0.5, 1, 0)
        # y_pred_mean = np.where(y_pred > np.mean(y_pred), 1, 0)
        #
        # accuracy_fixed = Utils.get_accuracy_score(np_y_test, y_pred_fixed)
        # print("Accuracy linear MLP with fixed threshold: {0}".format(accuracy_fixed))
        #
        # accuracy_mean = Utils.get_accuracy_score(np_y_test, y_pred_mean)
        # print("Accuracy linear MLP with mean threshold: {0}".format(accuracy_mean))
        # print("---" * 5)