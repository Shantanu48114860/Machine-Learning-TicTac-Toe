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
        print("Knn classifier")
        param_grid = {"n_neighbors": np.arange(1, 100, 2)}
        knn_gscv = GridSearchCV(KNeighborsClassifier(), param_grid, cv=10)
        knn_gscv.fit(np_X_train, np_Y_train)

        best_hyperparams = knn_gscv.best_params_
        optimal_k = best_hyperparams["n_neighbors"]
        print(optimal_k)

        clf = KNeighborsClassifier(n_neighbors=optimal_k)
        clf.fit(np_X_train, np_Y_train)

        Y_pred = clf.predict(np_X_test)
        acc = Utils.get_accuracy_score(np_Y_test, Y_pred)
        print(acc)

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
