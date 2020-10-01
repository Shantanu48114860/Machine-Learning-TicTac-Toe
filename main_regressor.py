from MultiClassRegression import RegressionAlgorithms

if __name__ == '__main__':
    print("####" * 20)
    print("--> Multi class Regression move dataset: <--")
    final_dataset_path = "datasets-part1/tictac_multi.txt"
    split_size = 0.8
    do_regression = RegressionAlgorithms()
    do_regression.multiclass_regression(final_dataset_path, split_size)
