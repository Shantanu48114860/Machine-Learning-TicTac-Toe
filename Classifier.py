import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from Utils import Utils


class Classifier:
    def classify_using_MLP(self, np_X_train, np_X_test, np_Y_train, np_Y_test):
        folds = KFold(n_splits=5, shuffle=True, random_state=10)

        param_grid = [
            {
                'activation': ['relu'],
                'hidden_layer_sizes': [
                    (100, 100), (120, 120), (150, 150), (200, 200)
                ]
            }
        ]
        clf = GridSearchCV(MLPClassifier(max_iter=100), param_grid, cv=folds,
                           scoring='accuracy')
        clf.fit(np_X_train, np_Y_train)
        best_score = clf.best_score_
        print("Best parameters set found on development set:")
        print(clf.best_params_)

        best_hyperparams = clf.best_params_
        print("The best test score is {0} corresponding to hyperparameters {1}"
              .format(best_score, best_hyperparams))
        activation = best_hyperparams['activation']
        hidden_layer_sizes = best_hyperparams['hidden_layer_sizes']

        final_clf = MLPClassifier(max_iter=100, activation=activation,
                                  hidden_layer_sizes=hidden_layer_sizes)
        final_clf.fit(np_X_train, np_Y_train)
        y_pred = self.test_knn(np_X_test, final_clf)

        print("Accuracy linear MLP: {0}".format(Utils.get_accuracy_score(np_Y_test, y_pred)))

    def classify_using_lin_SVM(self, np_X_train, np_X_test, np_Y_train, np_Y_test):
        folds = KFold(n_splits=5, shuffle=True, random_state=10)

        hyper_params = [{'gamma': [1, 0.1, 0.5],
                         'C': [10, 12]}]

        # specify model
        model = svm.SVC(kernel="linear")

        # set up GridSearchCV()
        model_cv = GridSearchCV(estimator=model,
                                param_grid=hyper_params,
                                scoring='accuracy',
                                cv=folds,
                                verbose=1,
                                return_train_score=True)

        model_cv.fit(np_X_train, np_Y_train)
        best_score = model_cv.best_score_
        best_hyperparams = model_cv.best_params_
        gamma = best_hyperparams['gamma']
        C = best_hyperparams['C']
        print("The best test score is {0} corresponding to hyperparameters {1}"
              .format(best_score, best_hyperparams))

        clf = svm.SVC(kernel="linear", gamma=gamma, C=C)
        clf.fit(np_X_train, np_Y_train)
        y_pred = self.test_knn(np_X_test, clf)
        print("Accuracy linear SVM: {0}".format(Utils.get_accuracy_score(np_Y_test, y_pred)))

        clf = svm.SVC(kernel="rbf", gamma=gamma, C=C)
        clf.fit(np_X_train, np_Y_train)
        y_pred = self.test_knn(np_X_test, clf)
        print("Accuracy rbf SVM: {0}".format(Utils.get_accuracy_score(np_Y_test, y_pred)))

    def classify_using_knn(self, np_X_train, np_X_test, np_Y_train, np_Y_test):
        k_list = list(range(3, 100, 2))
        classifier_dict, k_cv_score_dict = self.train_knn(np_X_train, np_Y_train, k_list)
        knn_score_list = []

        print("K value      cv_score")
        for items in k_cv_score_dict:
            print(str(items) + " ----- " + str(k_cv_score_dict[items]))

        print("Test Accuracies:")
        for knn_classifier_key in classifier_dict:
            print("knn -> " + str(knn_classifier_key))
            knn_classifier = classifier_dict[knn_classifier_key]
            y_pred = self.test_knn(np_X_test, knn_classifier)
            knn_score = knn_classifier.score(np_X_test, np_Y_test)
            knn_score_list.append(knn_score)
            print("knn_score: " +
                  str(knn_score))
            # confusion_mat, pred_accu = knn.calculate_decision_metric(y_true, y_pred)

        optimal_k = k_list[knn_score_list.index(max(knn_score_list))]
        optimal_knn_classifier = classifier_dict[optimal_k]
        print("Optimal K: {0}".format(optimal_k))
        # print(optimal_knn_classifier)
        print("Optimal knn test accuracy: {0}"
              .format(optimal_knn_classifier.score(np_X_test, np_Y_test)))

        self.__plot_knn_accuracy(k_list, knn_score_list, "Knn - final_move dataset",
                                 "./Plots/knn_final_move_plot.jpg")

    def train_knn(self, np_X_train, np_Y_train, k_list):
        cv_score = []
        knn_classifiers = []

        for k_value in k_list:
            classifier = KNeighborsClassifier(n_neighbors = k_value)
            # Train the model
            classifier.fit(np_X_train, np_Y_train)

            # scores is an array after running 10 times
            scores = cross_val_score(classifier, np_X_train, np_Y_train, cv=10,
                                     scoring="accuracy")
            cv_score.append(np.mean(scores))
            knn_classifiers.append(classifier)

        # return a corresponding classifers and cv_score for each of the kValues
        return dict(zip(k_list, knn_classifiers)), dict(zip(k_list, cv_score))

    def multiClassify_using_knn(self, np_X_train, np_X_test, np_Y_train, np_Y_test):
        k_list = list(range(3, 100, 2))

        # call the training method
        # Gets two dictionaries mapping classifier object and cv_score for each k_value
        classifier_dict, k_cv_score_dict = self.train_knn(np_X_train, np_Y_train, k_list)

        knn_score_list = []

        # Print the Train Accuracy (cross validation score)
        print("K value      cv_score")
        for items in k_cv_score_dict:
            print(str(items) + " ----- " + str(k_cv_score_dict[items]))

        print("Test Accuracies:")
        for k_value in classifier_dict:
            print("K value in knn -> " + str(k_value))
            knn_classifier = classifier_dict[k_value]
            # Get the test accuracy for the test data
            knn_score = knn_classifier.score(np_X_test, np_Y_test)
            knn_score_list.append(knn_score)
            print("knn_score: " + str(knn_score))
            # confusion_mat, pred_accu = knn.calculate_decision_metric(y_true, y_pred)

        # Get the k for which there is maximum accuracy
        optimal_k = k_list[knn_score_list.index(max(knn_score_list))]
        optimal_knn_classifier = classifier_dict[optimal_k]

        print("Optimal K: {0}".format(optimal_k))
        print("Optimal knn test accuracy: {0}"
              .format(optimal_knn_classifier.score(np_X_test, np_Y_test)))

        self.__plot_knn_accuracy(k_list, knn_score_list, "Knn - final_move dataset",
                                 "./Plots/knn_final_move_plot.jpg")



    @staticmethod
    def test_knn(X_test, classifier):
        y_pred = classifier.predict(X_test)
        return y_pred

    @staticmethod
    def __plot_knn_accuracy(k_list, knn_score, title, fig_name):
        plt.title(title)
        plt.plot(k_list, knn_score)
        plt.xlabel("Value of K for KNN")
        plt.ylabel('Testing Accuracy')
        plt.draw()
        plt.savefig(fig_name, dpi=220)
        plt.clf()
