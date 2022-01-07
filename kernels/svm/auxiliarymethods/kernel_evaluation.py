import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC

def kernel_svm_evaluation(train_matrices, valid_matrices, test_matrices, train_classes, valid_classes, test_classes, C=None, seed=None):
    if C is None:
        C = np.power(10.0, np.arange(3.0, -4.0, -1))
    num_iterations = len(train_matrices)
    # Acc. over all repetitions.
    train_accuracies = np.zeros((num_iterations,))
    valid_accuracies = np.zeros((num_iterations,))
    test_accuracies = np.zeros((num_iterations,))

    # Determine hyperparameters
    for k, (train, valid) in enumerate(zip(train_matrices, valid_matrices)):
        best_valid_acc = 0.0
        best_c = C[0]
        best_clf = None
        for c in C:
            clf = SVC(C=c, kernel="precomputed", tol=0.001, random_state=seed)
            clf.fit(train, train_classes)
            valid_acc = accuracy_score(valid_classes, clf.predict(valid)) * 100.0
            if best_valid_acc < valid_acc:
                best_valid_acc = valid_acc
                best_c = c
                best_clf = clf
        
        valid_accuracies[k] = best_valid_acc
        train_accuracies[k] = accuracy_score(train_classes, best_clf.predict(train)) * 100.0
        test_accuracies[k] = accuracy_score(test_classes, best_clf.predict(test_matrices[k])) * 100.0
    
    return train_accuracies, valid_accuracies, test_accuracies


# 10K-CV for kernel svm and hyperparameter selection.
def kernel_svm_cv(all_matrices, classes, num_folds=10, C=None, seed=None):
    if C is None:
        C = np.power(10.0, np.arange(3.0, -4.0, -1))
    num_iterations = len(all_matrices)
    # Acc. over all repetitions.
    train_accuracies = np.zeros((num_folds, num_iterations))
    valid_accuracies = np.zeros((num_folds, num_iterations))
    test_accuracies = np.zeros((num_folds, num_iterations))

    kf = KFold(n_splits=num_folds, random_state=seed, shuffle=True)

    for j, (train_index, test_index) in enumerate(kf.split(classes)):
        train_index, valid_index = train_test_split(train_index, test_size=0.1, random_state=seed, shuffle=True)
        
        test_matrices = [gram_matrix[test_index, :][:, train_index] for gram_matrix in all_matrices]
        test_classes = classes[test_index]

        # Determine hyperparameters
        for k, gram_matrix in enumerate(all_matrices):
            train = gram_matrix[train_index, :][:, train_index]
            valid = gram_matrix[valid_index, :][:, train_index]

            train_classes = classes[train_index]
            valid_classes = classes[valid_index]

            best_valid_acc = 0.0
            best_c = C[0]
            best_clf = None
            for c in C:
                clf = SVC(C=c, kernel="precomputed", tol=0.001, random_state=0)
                clf.fit(train, train_classes)
                valid_acc = accuracy_score(valid_classes, clf.predict(valid)) * 100.0
                if best_valid_acc < valid_acc:
                    best_valid_acc = valid_acc
                    best_c = c
                    best_clf = clf
            
            valid_accuracies[j, k] = best_valid_acc
            train_accuracies[j, k] = accuracy_score(train_classes, best_clf.predict(train)) * 100.0
            test_accuracies[j, k] = accuracy_score(test_classes, best_clf.predict(test_matrices[k])) * 100.0

    return train_accuracies, valid_accuracies, test_accuracies


# Weisfeiler and Leman go sparse: Towards scalable higher-order graph embeddings
def kernel_svm_nips(all_matrices, classes, num_repetitions=10, num_folds=10, C=None):
    if C is None:
        C = np.power(10.0, np.arange(3.0, -4.0, -1))
    num_iterations = len(all_matrices)
    # Acc. over all repetitions.
    train_accuracies = np.zeros((num_repetitions, num_folds, num_iterations))
    valid_accuracies = np.zeros((num_repetitions, num_folds, num_iterations))
    test_accuracies = np.zeros((num_repetitions, num_folds, num_iterations))

    for i in range(num_repetitions):
        kf = KFold(n_splits=num_folds, random_state=i, shuffle=True)

        for j, (train_index, test_index) in enumerate(kf.split(classes)):
            train_index, valid_index = train_test_split(train_index, test_size=0.1, random_state=i, shuffle=True)
            
            test_matrices = [gram_matrix[test_index, :][:, train_index] for gram_matrix in all_matrices]
            test_classes = classes[test_index]

            # Determine hyperparameters
            for k, gram_matrix in enumerate(all_matrices):
                train = gram_matrix[train_index, :][:, train_index]
                valid = gram_matrix[valid_index, :][:, train_index]

                train_classes = classes[train_index]
                valid_classes = classes[valid_index]

                best_valid_acc = 0.0
                best_c = C[0]
                best_clf = None
                for c in C:
                    clf = SVC(C=c, kernel="precomputed", tol=0.001, random_state=i)
                    clf.fit(train, train_classes)
                    valid_acc = accuracy_score(valid_classes, clf.predict(valid)) * 100.0
                    if best_valid_acc < valid_acc:
                        best_valid_acc = valid_acc
                        best_c = c
                        best_clf = clf
                
                valid_accuracies[i, j, k] = best_valid_acc
                train_accuracies[i, j, k] = accuracy_score(train_classes, best_clf.predict(train)) * 100.0
                test_accuracies[i, j, k] = accuracy_score(test_classes, best_clf.predict(test_matrices[k])) * 100.0

    return train_accuracies, valid_accuracies, test_accuracies
