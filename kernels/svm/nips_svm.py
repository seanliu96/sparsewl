import argparse
import numpy as np
import os
from auxiliarymethods.kernel_evaluation import kernel_svm_nips
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
        "--n_reps", type=int,
        default=10,
        help="number of repetitions"
    )
    parser.add_argument(
        "--n_folds", type=int,
        default=10,
        help="folds of cross-validation"
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

        # num_repetitions x num_folds x num_iterations
        train_accuracies_all, valid_accuracies_all, test_accuracies_all = \
            kernel_svm_nips(gram_matrices, classes, num_repetitions=args.n_reps, num_folds=args.n_folds)

        for k in range(len(gram_matrices)):
            print(
                kernel + "-" + str(k),
                dataset,
                round(train_accuracies_all[:, :, k].mean(), 2),
                round(train_accuracies_all[:, :, k].mean(axis=1).std(), 2),
                round(valid_accuracies_all[:, :, k].mean(), 2),
                round(valid_accuracies_all[:, :, k].mean(axis=1).std(), 2),
                round(test_accuracies_all[:, :, k].mean(), 2),
                round(test_accuracies_all[:, :, k].mean(axis=1).std(), 2),
                sep="\t"
            )
        
        best_k_ind = np.expand_dims(valid_accuracies_all.argmax(axis=2), axis=2)
        train_accuracies_avg = np.take_along_axis(train_accuracies_all, best_k_ind, axis=2).squeeze(axis=2)
        valid_accuracies_avg = np.take_along_axis(valid_accuracies_all, best_k_ind, axis=2).squeeze(axis=2)
        test_accuracies_avg = np.take_along_axis(test_accuracies_all, best_k_ind, axis=2).squeeze(axis=2)
        print(
            kernel + "-avg",
            dataset,
            round(train_accuracies_avg.mean(), 2),
            round(train_accuracies_avg.mean(axis=1).std(), 2),
            round(valid_accuracies_avg.mean(), 2),
            round(valid_accuracies_avg.mean(axis=1).std(), 2),
            round(test_accuracies_avg.mean(), 2),
            round(test_accuracies_avg.mean(axis=1).std(), 2),
            sep="\t"
        )
