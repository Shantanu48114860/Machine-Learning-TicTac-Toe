import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from Utils import Utils


class Regressor:
    def regression_using_knn(self, np_x_train, np_x_test, np_y_train, np_y_test):
        print("Knn Regression")
        param_grid = {"n_neighbors": np.arange(3, 100)}
        knn_gscv = GridSearchCV(KNeighborsRegressor(), param_grid, cv=10)
        knn_gscv.fit(np_x_train, np_y_train)

        best_hyperparams = knn_gscv.best_params_
        optimal_k = best_hyperparams["n_neighbors"]
        print(optimal_k)

        regressor = KNeighborsRegressor(n_neighbors=optimal_k)
        regressor.fit(np_x_train, np_y_train)

        Y_pred = regressor.predict(np_x_test)
        Y_pred = np.where(Y_pred > 0.5, 1, 0)

        #TO DO : Invoke confusion matrix
        total_acc = np.empty(9)
        for i in range(9):
            total_acc[i] = Utils.get_accuracy_score(np_y_test[:, i],
                                                    Y_pred[:, i], normalized=False)

        acc = np.sum(total_acc) / np.shape(np_y_test)[0] * 9
        print("Accuracy knn: {0}".format(acc))

    def regression_using_mlp(self, np_x_train, np_x_test, np_y_train, np_y_test):
        print(" --->>> MLP Regression")
        # folds = KFold(n_splits=10, shuffle=True, random_state=1)
        param_grid = [
            {
                'max_iter': [1000],
                'hidden_layer_sizes': [
                    (200, 200, 9), (300, 300, 9), (100, 50, 25, 9)
                ],
                'activation':['tanh','relu','sigmoid'],
                'solver': ['adam', 'sgd'],
                'alpha': [0.0001, 0.05],
                'learning_rate': ['constant', 'adaptive', 'invscaling'],
            }
        ]

        clf = GridSearchCV(MLPRegressor(random_state=1), param_grid, cv=10, n_jobs=-1)
        clf.fit(np_x_train, np_y_train)
        best_score = clf.best_score_
        print(best_score)
        print("Best parameters set found on development set:")
        print(clf.best_params_)

        best_hyperparams = clf.best_params_
        best_solver = best_hyperparams["solver"]
        best_learning_rate = best_hyperparams["learning_rate"]
        max_iter = best_hyperparams["max_iter"]
        best_layer_size = best_hyperparams["hidden_layer_sizes"]
        best_alpha = best_hyperparams["alpha"]
        best_activation = best_hyperparams["activation"]

        final_clf = MLPRegressor(random_state=1,
                                 max_iter=max_iter, activation=best_activation,
                                 hidden_layer_sizes=best_layer_size,
                                 learning_rate=best_learning_rate,
                                 solver=best_solver,
                                 alpha=best_alpha)

        # final_clf = MLPRegressor(random_state=1, max_iter=1000, activation='relu',
        #                          # learning_rate_init=1e-04, learning_rate='adaptive',
        #                          hidden_layer_sizes=(300, 300, 200, 100, 9))

        final_clf.fit(np_x_train, np_y_train)
        y_pred = final_clf.predict(np_x_test)

        y_pred_fixed = np.where(y_pred > 0.5, 1, 0)
        total_acc = np.empty(9)
        for i in range(9):
            total_acc[i] = Utils.get_accuracy_score(np_y_test[:, i],
                                                    y_pred_fixed[:, i], normalized=False)
        acc = np.sum(total_acc) / np.shape(np_y_test)[0] * 9
        print("Accuracy MLP: {0}".format(np.max(acc)))

    def linear_reg(self, np_x_train, np_x_test, np_y_train, np_y_test):
        print("Linear Regression")
        Y_pred = np.empty((np.shape(np_y_test)[0], np.shape(np_y_test)[1]))

        for i in range(9):
            y = np_y_train[:, i]
            W = np.linalg.inv(np_x_train.T @ np_x_train) @ np_x_train.T @ y
            y_pred = np_x_test @ W
            Y_pred[:, i] = y_pred

        Y_pred = (Y_pred == Y_pred.max(axis=1)[:, None]).astype(int)

        total_acc = np.empty(9)
        for i in range(9):
            total_acc[i] = Utils.get_accuracy_score(np_y_test[:, i],
                                                    Y_pred[:, i], normalized=False)

        acc = np.sum(total_acc) / np.shape(np_y_test)[0] * 9
        print("Accuracy LR: {0}".format(acc))

    @staticmethod
    def to_labels(Y_pred, t):
        return (Y_pred >= t).astype("int")
