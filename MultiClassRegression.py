from Regressor import Regressor
from dataloader import DataLoader

class RegressionAlgorithms:
    def multiclass_regression(self, final_dataset_path, split_size):
        dL = DataLoader()
        np_x_train, np_x_test, np_y_train, np_y_test = \
            dL.preprocess_data_from_csv_multi(final_dataset_path, split_size)

        startRegression = Regressor()
        startRegression.regression_using_knn(np_x_train, np_x_test, np_y_train, np_y_test)

        startRegression.regression_using_mlp(np_x_train, np_x_test,
                                             np_y_train, np_y_test)

        startRegression.linear_reg(np_x_train, np_x_test,
                                             np_y_train, np_y_test)