import argparse
import numpy as np
import pandas as pd
import os
from scipy import sparse as sp
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from auxiliarymethods.kernel_evaluation import kernel_svm_evaluation, linear_svm_evaluation
from auxiliarymethods.auxiliary_methods import read_lib_svm, normalize_gram_matrix, normalize_feature_vector


if __name__ == "__main__":
    
    dirname = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gram_dir", type=str,
        default=os.path.join(dirname, "GM", "EXP"),
        help="the directory of gram matrices"
    )
    parser.add_argument(
        "--dataset_dir", type=str,
        default=os.path.join(os.path.dirname(dirname), "datasets"),
        help="the directory of gram matrices"
    )
    parser.add_argument(
        "--k", type=int,
        default=1,
        help="complexity of kernel functions"
    )
    parser.add_argument(
        "--kernel", type=str,
        default="WL",
        help="kernel function"
    )
    parser.add_argument(
        "--datasets", nargs="+",
        help="datasets"
    )
    parser.add_argument(
        "--n_iters", type=int,
        default=5,
        help="number of runs"
    )
    args = parser.parse_args()

    kernel = args.kernel
    if kernel == "WL" or args.k != 1:
        kernel = kernel + str(args.k)

    for dataset in args.datasets:
        class_file_name = os.path.join(args.dataset_dir, dataset, dataset + "_graph_labels.txt")
        if not os.path.exists(class_file_name):
            class_file_name = os.path.join(args.dataset_dir, dataset + "_graph_labels.txt")
        if not os.path.exists(class_file_name):
            # raise FileNotFoundError("%s and %s are not found." % (
            #     os.path.join(args.dataset_dir, dataset, dataset + "_graph_labels.txt"),
            #     os.path.join(args.dataset_dir, dataset + "_graph_labels.txt")
            # ))
            print("%s and %s are not found." % (
                os.path.join(args.dataset_dir, dataset, dataset + "_graph_labels.txt"),
                os.path.join(args.dataset_dir, dataset + "_graph_labels.txt")
            ))
            continue

        with open(class_file_name, "r") as f:
            classes = np.asarray(f.readlines(), dtype=np.int32)

        gram_matrices = []
        for i in range(0, args.n_iters+1):
            gram_file_name = os.path.join(args.gram_dir, dataset + "__" + kernel + "_" + str(i) + ".gram")
            if os.path.exists(gram_file_name):
                gram_matrix, _ = read_lib_svm(gram_file_name)
                gram_matrix = normalize_gram_matrix(gram_matrix)
                gram_matrices.append(gram_matrix)

        if len(gram_matrices) == 0:
            # raise FileNotFoundError("Gram matrices for %s are not found." % (dataset))
            print("Gram matrices for %s are not found." % (dataset))
            continue

        acc_train, std_train, acc, std = kernel_svm_evaluation(gram_matrices, classes, num_repetitions=10)
        print(kernel, dataset, acc_train, std_train, acc, std, sep="\t")
