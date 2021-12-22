import argparse
import numpy as np
import pandas as pd
import os
from scipy import sparse as sp
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


def read_classes(ds_name):
    with open("../datasets/" + ds_name + "/" + ds_name + "_graph_labels.txt", "r") as f:
        classes = [int(i) for i in list(f)]
    f.closed

    return classes


def main():
    path = "./svm/SVM/src/EXPSPARSE/"

    for name in ["Yeast", "YeastH", "UACC257", "UACC257H", "OVCAR-8", "OVCAR-8H"]:
        for algorithm in ["WL", "LWL2", "LWLP2"]:

            print(name)
            print(algorithm)

            # Collect feature matrices over all iterations
            all_feature_matrices = []
            classes = read_classes(name)
            classes = np.array(classes)
            for i in range(0, 6):
                # Load feature matrices.
                feature_vector = pd.read_csv(path + name + "__" + algorithm + "_" + str(i), header=1,
                                             delimiter=" ").to_numpy()

                feature_vector = feature_vector.astype(int)
                feature_vector[:, 0] = feature_vector[:, 0] - 1
                feature_vector[:, 1] = feature_vector[:, 1] - 1
                feature_vector[:, 2] = feature_vector[:, 2] + 1

                xmax = int(feature_vector[:, 0].max())
                ymax = int(feature_vector[:, 1].max())

                feature_vector = sp.coo_matrix(
                    (feature_vector[:, 2], (feature_vector[:, 0], feature_vector[:, 1])), shape=(xmax + 1, ymax + 1)
                )
                feature_vector = feature_vector.tocsr()

                all_feature_matrices.append(feature_vector)
            print("### Data loading done.")

            test_accuracies_all = []
            for _ in range(10):

                kf = KFold(n_splits=10, shuffle=True)
                test_accuracies = []
                for train_index, test_index in kf.split(list(range(len(classes)))):
                    best_f = None
                    best_m = None
                    best_val = 0.0
                    for f in all_feature_matrices:
                        train_index, val_index = train_test_split(train_index, test_size=0.1)
                        train = f[train_index]
                        val = f[val_index]
                        c_train = classes[train_index]
                        c_val = classes[val_index]
                        for c in [10**3, 10**2, 10**1, 10**0, 10**-1, 10**-2, 10**-3]:
                            clf = LinearSVC(C=c)
                            clf.fit(train, c_train)
                            p = clf.predict(val)
                            a = np.sum(np.equal(p, c_val)) / val.shape[0]

                            if a > best_val:
                                best_val = a
                                best_f = f
                                best_m = clf

                    test = best_f[test_index]
                    c_test = classes[test_index]
                    p = best_m.predict(test)
                    a = np.sum(np.equal(p, c_test)) / test.shape[0]
                    test_accuracies.append(a * 100.0)

                test_accuracies_all.append(np.mean(test_accuracies))

            print(np.mean(test_accuracies_all), np.std(test_accuracies_all))


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

        all_feature_matrices = []
        for i in range(0, args.n_iters+1):
            gram_file_name = os.path.join(args.gram_dir, dataset + "__" + kernel + "_" + str(i) + ".gram")
            if os.path.exists(gram_file_name):
                gram_matrix, _ = read_lib_svm(gram_file_name)
                gram_matrix = normalize_gram_matrix(gram_matrix)
                gram_matrices.append(gram_matrix)