import os
import sys

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)

from utils.utils import load_json


def analysis_data(data_dict):
    data_types = list(data_dict.keys())
    dataset_num = 0
    data_num = 0
    for data_type in data_types:
        datasets = data_dict[data_type].keys()
        dataset_num += len(datasets)
        print("Data type {} has {} datasets: {}".format(data_type.upper(), len(datasets), datasets))
        for dataset in datasets:
            print("Dataset {} has {} samples, {} labels".format(
                dataset.upper(),
                len(data_dict[data_type][dataset]["data_list"]),
                len(data_dict[data_type][dataset]["label_mappings"])
            ))
            data_num += len(data_dict[data_type][dataset]["data_list"])
    print("Total: {} data types; {} datasets; {} data samples".format(
        len(data_types),
        dataset_num,
        data_num
    ))


if __name__ == "__main__":
    path_prefix = "../data/"
    opensource_file = path_prefix+"opensource_sample_500.json"
    competition_train = path_prefix+"train_data.json"
    competition_test = path_prefix+"test_data.json"
    opensource_data = load_json(opensource_file)
    analysis_data(opensource_data)
    train_data = load_json(competition_train)
    analysis_data(train_data)
    test_data = load_json(competition_test)
    analysis_data(test_data)
