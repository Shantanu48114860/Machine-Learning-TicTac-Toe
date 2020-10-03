import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, KFold
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

        regressor_dict = self.train_knn(np_x_train, np_y_train, k_list)

        for k_value in regressor_dict:
            print("k value in multiclass knn -> " + str(k_value))
            knn_regressor = regressor_dict[k_value]
            y_pred = knn_regressor.predict(np_x_test)  # 1311 x 9
            y_pred_list.append(y_pred)

            y_pred_fixed = np.where(y_pred > 0.5, 1, 0)

            accuracy_fixed = Utils.get_accuracy_score(np_y_test, y_pred_fixed)
            accuracy_threshold_fixed.append(accuracy_fixed)
            print(accuracy_fixed)

            print("---" * 5)

        # Get the k for which there is maximum accuracy
        best_k_fixed = k_list[accuracy_threshold_fixed.index(max(accuracy_threshold_fixed))]

        print("Optimal K for fixed threshold: {0}".format(best_k_fixed))
        print(accuracy_threshold_fixed[accuracy_threshold_fixed.index(max(accuracy_threshold_fixed))])

    def regression_using_mlp(self, np_x_train, np_x_test, np_y_train, np_y_test):
        folds = KFold(n_splits=5, shuffle=True, random_state=1)
        param_grid = [
            {
                'max_iter': [1000],
                'hidden_layer_sizes': [
                    (200, 200, 9), (300, 300, 9)
                ]
            }
        ]

        clf = GridSearchCV(MLPRegressor(random_state=1,
                                        activation='relu',
                                        solver='adam'),
                           param_grid, cv=folds)
        clf.fit(np_x_train, np_y_train)
        best_score = clf.best_score_
        print(best_score)
        print("Best parameters set found on development set:")
        print(clf.best_params_)

        best_hyperparams = clf.best_params_
        # learning_rate_init = best_hyperparams["learning_rate_init"]
        max_iter = best_hyperparams["max_iter"]
        hidden_layer_sizes = best_hyperparams["hidden_layer_sizes"]

        final_clf = MLPRegressor(random_state=1,
                                 max_iter=1000, activation='relu',
                                 hidden_layer_sizes=hidden_layer_sizes,
                                 solver='adam')

        # final_clf = MLPRegressor(random_state=1, max_iter=1000, activation='relu',
        #                          # learning_rate_init=learning_rate_init,
        #                         hidden_layer_sizes=(200, 200, 9))

        final_clf.fit(np_x_train, np_y_train)
        y_pred = final_clf.predict(np_x_test)

        y_pred_fixed = np.where(y_pred > 0.5, 1, 0)

        accuracy_fixed = Utils.get_accuracy_score(np_y_test, y_pred_fixed)
        print("Accuracy linear MLP with fixed threshold: {0}".format(accuracy_fixed))

    def linear_reg(self, np_x_train, np_x_test, np_y_train, np_y_test):
        # X = np.c_[np.ones(np.shape(np_x_train)[0]), np_x_train]
        # x_t = X.T
        # W = np.linalg.inv(x_t.dot(X)).dot(x_t).dot(np_y_train)
        #
        # x_t = np.c_[np.ones(np.shape(np_x_test)[0]), np_x_test]
        # Y_pred = np.dot(x_t, W)

        Y_pred = np.empty((np.shape(np_y_test)[0], np.shape(np_y_test)[1]))
        print(np.shape(Y_pred))

        for i in range(9):
            y = np_y_train[:, i]
            W = np.linalg.inv(np_x_train.T @ np_x_train) @ np_x_train.T @ y
            y_pred = np_x_test @ W
            Y_pred[:, i] = y_pred

        print(np.shape(Y_pred))
        print(Y_pred)
        print(np.shape(np_y_test))
        print(np_y_test)

        # threshold = np.arange(0, 1, 0.001)
        #
        # scores = [f1_score(np_y_test, self.to_labels(Y_pred, t), average="weighted")
        #           for t in threshold]
        # ix = np.argmax(scores)
        #
        # print(ix)
        # print("Threshold: {0}, F1: {1}".format(threshold[ix], scores[ix]))
        # Y_thrs = np.where(Y_pred < threshold[ix], 1, 0)
        # print(Y_thrs)
        #
        # print(np_y_test[0])



        print("Accuracy: {0}".format(Utils.get_accuracy_score(np_y_test, Y_pred)))

    @staticmethod
    def to_labels(Y_pred, t):
        return (Y_pred >= t).astype("int")
