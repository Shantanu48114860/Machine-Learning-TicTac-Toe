from sklearn.model_selection import StratifiedKFold

from Classifier import Classifier
from Linear_regressor import Linear_regressor
from dataloader import DataLoader


class ML_Algo:
    def execute_classifier(self, final_dataset_path, split_size):
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

    def execute_regressor(self, final_dataset_path, split_size):
        dL = DataLoader()
        np_X_train, np_X_test, np_Y_train, np_Y_test = \
            dL.preprocess_data_from_csv_multi(final_dataset_path, split_size)
        lin_reg = Linear_regressor()

        print("1. Linear regressor")
        lin_reg.do_lin_reg(np_X_train, np_X_test, np_Y_train, np_Y_test)


