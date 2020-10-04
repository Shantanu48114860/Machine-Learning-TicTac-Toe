from Classifier import Classifier
from Regressor import Regressor
from dataloader import DataLoader


class ML_Algo:
    @staticmethod
    def execute_classifier(final_dataset_path, split_size):
        dL = DataLoader()

        np_X_train, np_X_test, np_Y_train, np_Y_test = \
            dL.preprocess_data_from_csv(final_dataset_path, split_size)

        classifier = Classifier()

        print("---" * 20)
        print("1. Model: KNN")
        classifier.classify_using_knn(np_X_train, np_X_test, np_Y_train, np_Y_test)

        print("---" * 20)
        print("2. Model: SVM")
        classifier.classify_using_lin_SVM(np_X_train, np_X_test, np_Y_train, np_Y_test)

        print("---" * 20)
        print("3. Model: MLP")
        classifier.classify_using_MLP(np_X_train, np_X_test, np_Y_train, np_Y_test)

    @staticmethod
    def multiclass_regression(final_dataset_path, split_size):
        dL = DataLoader()
        np_x_train, np_x_test, np_y_train, np_y_test = \
            dL.preprocess_data_from_csv_multi(final_dataset_path, split_size)

        startRegression = Regressor()
        print("---" * 20)
        print("1. Model: KNN")
        startRegression.regression_using_knn(np_x_train, np_x_test, np_y_train, np_y_test)

        print("---" * 20)
        print("2. Model: MLP")
        startRegression.regression_using_mlp(np_x_train, np_x_test,
                                             np_y_train, np_y_test)

        print("---" * 20)
        print("2. Model: Linear Regression")
        startRegression.linear_reg(np_x_train, np_x_test,
                                   np_y_train, np_y_test)
