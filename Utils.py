import sklearn.model_selection as sklearn
from sklearn.metrics import accuracy_score


class Utils:
    @staticmethod
    def convert_df_to_np_arr(data):
        return data.to_numpy()

    @staticmethod
    def test_train_split(covariates_X, treatment_Y, split_size=0.8):
        return sklearn.train_test_split(covariates_X, treatment_Y, train_size=split_size)

    @staticmethod
    def get_accuracy_score(y_true, y_pred):
        pred_accu = accuracy_score(y_true, y_pred)
        return pred_accu
