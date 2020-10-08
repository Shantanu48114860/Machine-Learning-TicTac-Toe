import numpy as np
import sklearn.model_selection as sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


class DataLoader:
    @staticmethod
    def preprocess_data_from_csv(dataset_path, split_size):
        print(".. Data Loading ..")

        # data load
        np_arr = np.loadtxt(dataset_path)
        np_x = np_arr[:, :9]
        np_y = np_arr[:, 9]
        print("ps_np_covariates_X: {0}".format(np_x.shape))
        print("ps_np_treatment_Y: {0}".format(np_y.shape))

        np_x_train, np_x_test, np_y_train, np_y_test = \
            Utils.test_train_split(np_x, np_y, split_size)
        print("np_covariates_X_train: {0}".format(np_x_train.shape))
        print("np_covariates_Y_train: {0}".format(np_y_train.shape))

        print("np_covariates_X_test: {0}".format(np_x_test.shape))
        print("np_covariates_Y_test: {0}".format(np_y_test.shape))

        return np_x_train, np_x_test, np_y_train, np_y_test

class Utils:
    @staticmethod
    def test_train_split(covariates_x, treatment_y, split_size=0.8):
        return sklearn.train_test_split(covariates_x, treatment_y, train_size=split_size)

    @staticmethod
    def get_accuracy_score(y_true, y_pred, normalized=True):
        pred_accu = accuracy_score(y_true, y_pred, normalize=normalized)
        return pred_accu


class Classifier:
    def classify_using_MLP(self, np_X_train, np_X_test, np_Y_train, np_Y_test):
        folds = KFold(n_splits=10, shuffle=True, random_state=10)

        param_grid = [
            {
                'activation': ['relu'],
                'hidden_layer_sizes': [
                    (100, 100), (120, 120), (150, 150), (200, 200)
                ]
            }
        ]
        clf = GridSearchCV(MLPClassifier(max_iter=1000), param_grid, cv=folds,
                           scoring='accuracy', n_jobs=-1)
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

        confusion_matrix(np_Y_test, y_pred)
        disp = plot_confusion_matrix(final_clf, np_X_test, np_Y_test)
        disp.ax_.set_title("Normalized Confusion Matrix")
        print(disp.confusion_matrix)
        plt.show()


        print("Accuracy linear MLP: {0}".format(Utils.get_accuracy_score(np_Y_test, y_pred)))

    def classify_using_lin_SVM(self, np_X_train, np_X_test, np_Y_train, np_Y_test):
        folds = KFold(n_splits=10, shuffle=True, random_state=10)

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
                                return_train_score=True,
                                n_jobs=-1)

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

        confusion_matrix(np_Y_test, y_pred)
        disp = plot_confusion_matrix(clf, np_X_test, np_Y_test)
        disp.ax_.set_title("Normalized Confusion Matrix")
        print(disp.confusion_matrix)
        plt.show()


        print("Accuracy linear SVM: {0}".format(Utils.get_accuracy_score(np_Y_test, y_pred)))

        # clf = svm.SVC(kernel="rbf", gamma=gamma, C=C)
        # clf.fit(np_X_train, np_Y_train)
        # y_pred = self.test_knn(np_X_test, clf)
        #
        # confusion_matrix(np_Y_test, y_pred)
        # disp = plot_confusion_matrix(clf, np_X_test, np_Y_test)
        # disp.ax_.set_title("Normalized Confusion Matrix")
        # print(disp.confusion_matrix)
        # plt.show()
        #
        # print("Accuracy rbf SVM: {0}".format(Utils.get_accuracy_score(np_Y_test, y_pred)))

    def classify_using_knn(self, np_X_train, np_X_test, np_Y_train, np_Y_test):
        print("Knn classifier")
        param_grid = {"n_neighbors": np.arange(1, 100, 2)}
        knn_gscv = GridSearchCV(KNeighborsClassifier(), param_grid, cv=10, n_jobs=-1)
        knn_gscv.fit(np_X_train, np_Y_train)

        best_hyperparams = knn_gscv.best_params_
        optimal_k = best_hyperparams["n_neighbors"]
        print(optimal_k)

        clf = KNeighborsClassifier(n_neighbors=optimal_k)
        clf.fit(np_X_train, np_Y_train)

        Y_pred = clf.predict(np_X_test)
        acc = Utils.get_accuracy_score(np_Y_test, Y_pred)
        print(acc)

        confusion_matrix(np_Y_test, Y_pred)
        disp = plot_confusion_matrix(clf, np_X_test, np_Y_test)
        disp.ax_.set_title("Normalized Confusion Matrix")
        print(disp.confusion_matrix)
        plt.show()

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


def execute_classifier(final_dataset_path, split_size):
    dL = DataLoader()

    np_x_train, np_x_test, np_Y_train, np_Y_test = \
        dL.preprocess_data_from_csv(final_dataset_path, split_size)

    classifier = Classifier()

    print("---" * 20)
    print("1. Model: KNN")
    classifier.classify_using_knn(np_x_train, np_x_test, np_Y_train, np_Y_test)

    print("---" * 20)
    print("2. Model: SVM")
    classifier.classify_using_lin_SVM(np_x_train, np_x_test, np_Y_train, np_Y_test)

    print("---" * 20)
    print("3. Model: MLP")
    classifier.classify_using_MLP(np_x_train, np_x_test, np_Y_train, np_Y_test)


if __name__ == '__main__':
    print("--> Final move dataset: <--")
    final_dataset_path = "datasets-part1/tictac_final.txt"
    split_size = 0.8
    execute_classifier(final_dataset_path, split_size)

    # print("####" * 20)
    # print("--> Single class classification move dataset: <--")
    # final_dataset_path = "datasets-part1/tictac_single.txt"
    # split_size = 0.8
    # execute_classifier(final_dataset_path, split_size)