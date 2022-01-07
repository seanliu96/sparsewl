import argparse
import numpy as np
import os
import torch
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
from auxiliarymethods.kernel_evaluation import kernel_svm_evaluation
from auxiliarymethods.auxiliary_methods import read_lib_svm, normalize_gram_matrix


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
        "--n_iters", type=int,
        default=5,
        help="number of runs"
    )
    parser.add_argument(
        "--seeds", nargs="+",
        default=0,
        help="seeds"
    )
    parser.add_argument(
        "--datasets", nargs="+",
        help="datasets"
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

        train_accuracies_all = np.zeros((len(args.seeds), len(gram_matrices)))
        valid_accuracies_all = np.zeros((len(args.seeds), len(gram_matrices)))
        test_accuracies_all = np.zeros((len(args.seeds), len(gram_matrices)))
        for i, seed in enumerate(args.seeds):
            seed = int(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            num_graphs = len(gram_matrices[0])
            num_train = int(num_graphs * 0.8)
            num_valid = int(num_graphs * 0.1)
            num_test = num_graphs - (num_train + num_valid)
            train_index, valid_index, test_index = random_split(np.arange(num_graphs), [num_train, num_valid, num_test])
            train_index = train_index.indices
            valid_index = valid_index.indices
            test_index = test_index.indices
            train_matrices = [gram_matrix[train_index][:, train_index] for gram_matrix in gram_matrices]
            valid_matrices = [gram_matrix[valid_index][:, train_index] for gram_matrix in gram_matrices]
            test_matrices = [gram_matrix[test_index][:, train_index] for gram_matrix in gram_matrices]

            # num_iterations
            train_accuracies, valid_accuracies, test_accuracies = \
                kernel_svm_evaluation(
                    train_matrices, valid_matrices, test_matrices,
                    classes[train_index], classes[valid_index], classes[test_index],
                    C=None, seed=seed
                )
            train_accuracies_all[i] = train_accuracies
            valid_accuracies_all[i] = valid_accuracies
            test_accuracies_all[i] = test_accuracies

        for k in range(len(gram_matrices)):
            print(
                kernel + "-" + str(k),
                dataset,
                round(train_accuracies_all[:, k].mean(), 1),
                round(train_accuracies_all[:, k].std(), 1),
                round(valid_accuracies_all[:, k].mean(), 1),
                round(valid_accuracies_all[:, k].std(), 1),
                round(test_accuracies_all[:, k].mean(), 1),
                round(test_accuracies_all[:, k].std(), 1),
                sep="\t"
            )
        
        best_k_ind = np.expand_dims(valid_accuracies_all.argmax(axis=1), axis=1)
        train_accuracies_avg = np.take_along_axis(train_accuracies_all, best_k_ind, axis=1).squeeze(axis=1)
        valid_accuracies_avg = np.take_along_axis(valid_accuracies_all, best_k_ind, axis=1).squeeze(axis=1)
        test_accuracies_avg = np.take_along_axis(test_accuracies_all, best_k_ind, axis=1).squeeze(axis=1)
        print(
            kernel + "-avg",
            dataset,
            round(train_accuracies_avg.mean(), 2),
            round(train_accuracies_avg.std(), 2),
            round(valid_accuracies_avg.mean(), 2),
            round(valid_accuracies_avg.std(), 2),
            round(test_accuracies_avg.mean(), 2),
            round(test_accuracies_avg.std(), 2),
            sep="\t"
        )
