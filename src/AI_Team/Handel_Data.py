import math, json, os, sys
def get_file_list_from_dir(datadir):
    all_files = os.listdir(os.path.abspath(datadir))
    data_files = list(filter(lambda file: file.endswith('.png'), all_files))
    return data_files

from random import shuffle

def randomize_files(file_list):
    shuffle(file_list)

from math import floor

def get_training_and_testing_sets(file_list):
    split = 0.7
    split_index = floor(len(file_list) * split)
    training = file_list[:split_index]
    testing = file_list[split_index:]
    return training, testing




def ml_function(datadir, num_folds):
    data_files = get_file_list_from_dir(datadir)
    randomize_files(data_files)
    for train_set, test_set in cross_validate(data_files, num_folds):
        do_ml_training(train_set)
        do_ml_testing(test_set)



l =get_file_list_from_dir("C:/Users/ali_a/Desktop/AUT-2018/Intact/meetup-ML-assurance-hackathon/data/data-train/data-train")
