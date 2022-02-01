import argparse
import os
import sklearn.datasets as ds
import numpy as np

def read_lib_svm(file_name):
    gram_matrix, labels = ds.load_svmlight_file(file_name, multilabel=False)
    return gram_matrix.toarray(), labels

def write_lib_svm(file_name, gram_matrix, labels):
    ds.dump_svmlight_file(gram_matrix, labels, file_name, zero_based=False, multilabel=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_feat_files", nargs="+", default="../GM/MUTAG")
    parser.add_argument("--save_feat_file", type=str, default="../GM/ENSEMBLE-MUTAG")
    args = parser.parse_args()
    assert len(args.load_feat_files) > 0
    gram_matrix, labels = read_lib_svm(args.load_feat_files[0])
    for file_name in args.load_feat_files[1:]:
        gram_matrix_, labels_ = read_lib_svm(file_name)
        assert np.equal(labels, labels_).all()
        gram_matrix += gram_matrix_
    write_lib_svm(args.save_feat_file, gram_matrix, labels)