from ML_Algo import ML_Algo

if __name__ == '__main__':
    print("--> Final move dataset: <--")
    final_dataset_path = "datasets-part1/tictac_final.txt"
    split_size = 0.8
    classifier_final_dataset = ML_Algo()
    classifier_final_dataset.execute_classifier(final_dataset_path, split_size)

    print("####" * 20)
    print("--> Single class classification move dataset: <--")
    final_dataset_path = "datasets-part1/tictac_single.txt"
    split_size = 0.8
    ML_Algo.execute_classifier(final_dataset_path, split_size)

