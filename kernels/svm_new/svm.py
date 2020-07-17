import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from auxiliarymethods.kernel_evaluation import kernel_svm_evaluation, linear_svm_evaluation
from auxiliarymethods.auxiliary_methods import read_lib_svm, normalize_gram_matrix
import os.path
from os import path as pth

def read_classes(ds_name):
    with open("../../../WWW/graphkerneldatasets/DS_all/" + ds_name + "/" + ds_name + "_graph_labels.txt", "r") as f:
        classes = [int(i) for i in list(f)]
    f.closed

    return np.array(classes)


def main():


    path = "../svm/SVM/src/EXP/"
    dataset = [["IMDB-BINARY", False], ["IMDB-MULTI", False], ["NCI1", True], ["NCI109", True], ["PROTEINS", True],
               ["PTC_FM", True], ["REDDIT-BINARY", False], ["ENZYMES", True]]
    algorithms = ["WL1", "GR", "SP", "WLOA", "LWL2", "LWLP2", "WL2", "DWL2", "LWL3", "LWLP3", "WL3", "DWL3"]

    for a in algorithms:
        for d, use_labels in dataset:
            gram_matrices = []
            for i in range(0,6):
                if not pth.exists(path + d + "__" + a + "_" + str(i) + ".gram"):
                    continue
                else:
                    gram_matrix, _ = read_lib_svm(path + d + "__" + a + "_" + str(i) + ".gram")
                    gram_matrix = normalize_gram_matrix(gram_matrix)
                    classes = read_classes(d)
                    gram_matrices.append(gram_matrix)

            if gram_matrices != []:
                acc, acc_train, s_1 = kernel_svm_evaluation(gram_matrices, classes, num_repetitions=10, all_std=False)
                print(a, d, acc, acc_train, s_1)

    exit()

    path = "../svm/SVM/src/EXPSPARSE/"
    for name in ["Yeast", "YeastH", "UACC257", "UACC257H", "OVCAR-8", "OVCAR-8H"]:
        for algorithm in ["WL", "LWL2", "LWLP2"]:

            # Collect feature matrices over all iterations
            all_feature_matrices = []
            classes = read_classes(name)
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

                feature_vector = sp.coo_matrix((feature_vector[:, 2], (feature_vector[:, 0], feature_vector[:, 1])),
                                               shape=(xmax + 1, ymax + 1))
                feature_vector = feature_vector.tocsr()

                all_feature_matrices.append(feature_vector)

                acc, s_1 = linear_svm_evaluation(all_feature_matrices, classes, num_repetitions=3, all_std=False)
                print(name, algorithm, acc, s_1)









if __name__ == "__main__":
    main()
