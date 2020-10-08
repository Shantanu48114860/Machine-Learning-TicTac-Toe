import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection as sklearn
from mlxtend.plotting import plot_confusion_matrix
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


class DataLoader:
    @staticmethod
    def preprocess_data_from_csv(dataset_path, split_size, fraction_10th):
        print(".. Data Loading ..")

        # data load
        np_arr = np.loadtxt(dataset_path)
        if fraction_10th:
            size_10th = int(np.round(np_arr.shape[0] / 10))
            random_indices = np.random.choice(np_arr.shape[0], size=size_10th, replace=False)
            # print("Size:" + str(random_indices))
            np_arr = np_arr[random_indices, :]
            print(np_arr.shape)

            # np_arr = np_arr[:size_10th, :]

        np_x = np_arr[:, :9]
        np_y = np_arr[:, 9]

        print("ps_np_X: {0}".format(np_x.shape))
        print("ps_np_Y: {0}".format(np_y.shape))
        np_x_train, np_x_test, np_y_train, np_y_test = \
            Utils.test_train_split(np_x, np_y, split_size)

        print("np_X_train: {0}".format(np_x_train.shape))
        print("np_Y_train: {0}".format(np_y_train.shape))

        print("np_X_test: {0}".format(np_x_test.shape))
        print("np_Y_test: {0}".format(np_y_test.shape))

        return np_x_train, np_x_test, np_y_train, np_y_test


class Utils:
    @staticmethod
    def test_train_split(covariates_x, treatment_y, split_size=0.8):
        return sklearn.train_test_split(covariates_x, treatment_y, train_size=split_size)

    @staticmethod
    def get_accuracy_score(y_true, y_pred, normalized=True):
        pred_accu = accuracy_score(y_true, y_pred, normalize=normalized)
        return pred_accu

    @staticmethod
    def plot_confusion_matrix(confusion_matrix, fig_title):
        print("Confusion Matrix:")
        print(confusion_matrix)
        fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix,
                                        show_absolute=True,
                                        show_normed=True,
                                        colorbar=True)
        plt.title(label=fig_title)
        plt.draw()
        plt.show()
        plt.savefig("./Plots/" + fig_title, dpi=220)
        plt.clf()


    @staticmethod
    def plot_knn_accuracy(k_list, knn_score, title):
        plt.title(title)
        plt.plot(k_list, knn_score)
        plt.xlabel("Value of K for KNN")
        plt.ylabel('Validation Accuracy')
        plt.draw()
        plt.savefig("./Plots/" + title, dpi=220)
        plt.clf()


class Classifier:
    def classify_using_MLP(self, np_X_train, np_X_test, np_Y_train, np_Y_test, fig_title,
                           fraction_10th):
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

        # print(clf.cv_results_['params'])
        # print(clf.cv_results_['mean_test_score'])

        final_clf = MLPClassifier(max_iter=100, activation=activation,
                                  hidden_layer_sizes=hidden_layer_sizes)
        final_clf.fit(np_X_train, np_Y_train)
        y_pred = self.test_knn(np_X_test, final_clf)

        confusion_mat = confusion_matrix(np_Y_test, y_pred)
        Utils.plot_confusion_matrix(confusion_mat, fig_title)

        # if not fraction_10th:
        #     confusion_mat = confusion_matrix(np_Y_test, y_pred)
        #     Utils.plot_confusion_matrix(confusion_mat, fig_title)

        print("Accuracy linear MLP: {0}".format(Utils.get_accuracy_score(np_Y_test, y_pred)))

    def classify_using_lin_SVM(self, np_X_train, np_X_test, np_Y_train, np_Y_test, fig_title,
                               fraction_10th):
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

        # print(model_cv.cv_results_['params'])
        # print(model_cv.cv_results_['mean_test_score'])

        print("The best test score is {0} corresponding to hyperparameters {1}"
              .format(best_score, best_hyperparams))

        clf = svm.SVC(kernel="linear", gamma=gamma, C=C)
        clf.fit(np_X_train, np_Y_train)
        y_pred = self.test_knn(np_X_test, clf)

        confusion_mat = confusion_matrix(np_Y_test, y_pred)
        Utils.plot_confusion_matrix(confusion_mat, fig_title)

        # if not fraction_10th:
        #     confusion_mat = confusion_matrix(np_Y_test, y_pred)
        #     Utils.plot_confusion_matrix(confusion_mat, fig_title)

        print("Accuracy linear SVM: {0}".format(Utils.get_accuracy_score(np_Y_test, y_pred)))

    def classify_using_knn(self, np_X_train, np_X_test, np_Y_train, np_Y_test, fig_title,
                           k_range, fraction_10th):
        print("Knn classifier")
        param_grid = {"n_neighbors": np.arange(1, k_range, 2)}
        knn_gscv = GridSearchCV(KNeighborsClassifier(), param_grid, cv=10, n_jobs=-1)
        knn_gscv.fit(np_X_train, np_Y_train)

        best_hyperparams = knn_gscv.best_params_
        optimal_k = best_hyperparams["n_neighbors"]
        k_ = np.arange(1, k_range, 2)

        scores = knn_gscv.cv_results_['mean_test_score']

        clf = KNeighborsClassifier(n_neighbors=optimal_k)
        clf.fit(np_X_train, np_Y_train)

        Y_pred = clf.predict(np_X_test)
        acc = Utils.get_accuracy_score(np_Y_test, Y_pred)
        print("Accuracy Knn: {0}".format(acc))
        # Utils.plot_knn_accuracy(k_, scores, fig_title + " (knn_Plot)")

        confusion_mat = confusion_matrix(np_Y_test, Y_pred)
        Utils.plot_confusion_matrix(confusion_mat, fig_title)

        # if not fraction_10th:
        #     confusion_mat = confusion_matrix(np_Y_test, Y_pred)
        #     Utils.plot_confusion_matrix(confusion_mat, fig_title)

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


def execute_classifier(final_dataset_path, split_size, title, k_range, fraction_10th=False):
    dL = DataLoader()

    np_x_train, np_x_test, np_Y_train, np_Y_test = \
        dL.preprocess_data_from_csv(final_dataset_path, split_size, fraction_10th)

    classifier = Classifier()

    print("---" * 20)
    print("1. Model: KNN")
    classifier.classify_using_knn(np_x_train, np_x_test, np_Y_train, np_Y_test, title + "Knn",
                                  k_range, fraction_10th)

    print("---" * 20)
    print("2. Model: SVM")
    classifier.classify_using_lin_SVM(np_x_train, np_x_test, np_Y_train, np_Y_test, title + "SVM",
                                      fraction_10th)

    # print("---" * 20)
    print("3. Model: MLP")
    classifier.classify_using_MLP(np_x_train, np_x_test, np_Y_train, np_Y_test, title + "MLP",
                                  fraction_10th)


if __name__ == '__main__':
    print("--> Final move dataset: <--")
    final_dataset_path = "datasets-part1/tictac_final.txt"
    split_size = 0.8
    execute_classifier(final_dataset_path, split_size, "Final move dataset - ",
                       k_range=100, fraction_10th=False)
    print("---" * 20)
    print("####" * 20)
    print("---" * 20)
    print("--> Single class classification move dataset: <--")
    final_dataset_path = "datasets-part1/tictac_single.txt"
    split_size = 0.8
    execute_classifier(final_dataset_path, split_size, "Single Classifier dataset - ",
                       k_range=100,fraction_10th=False)

    print("---" * 20)
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("---" * 20)
    print("--> 0.10 Fraction - Final move dataset: <--")
    final_dataset_path = "datasets-part1/tictac_final.txt"
    split_size = 0.8
    execute_classifier(final_dataset_path, split_size, "Final move dataset(10_th_Fraction) - ",
                       k_range=10, fraction_10th=True)

    print("####" * 20)
    print("--> 0.10 Fraction - Single class classification move dataset: <--")
    final_dataset_path = "datasets-part1/tictac_single.txt"
    split_size = 0.8
    execute_classifier(final_dataset_path, split_size, "Single Classifier dataset(10_th_Fraction) - ",
                       k_range=10, fraction_10th=True)
