import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC


# 10-CV for linear svm with sparse feature vectors and hyperparameter selection.
def linear_svm_evaluation(all_matrices, classes, num_repetitions=10,
                          C=[10 ** 3, 10 ** 2, 10 ** 1, 10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3], all_std=False):
    # Acc. over all repetitions.
    test_accuracies_all = []
    # All acc. over all folds and repetitions.
    test_accuracies_complete = []

    for i in range(num_repetitions):
        # Test acc. over all folds.
        test_accuracies = []
        kf = KFold(n_splits=10, random_state=i, shuffle=True)

        for train_index, test_index in kf.split(list(range(len(classes)))):
            # Sample 10% split from training split for validation.
            train_index, val_index = train_test_split(train_index, test_size=0.1)
            best_val_acc = 0.0
            best_gram_matrix = all_matrices[0]
            best_c = C[0]

            for gram_matrix in all_matrices:
                train = gram_matrix[train_index]
                val = gram_matrix[val_index]

                c_train = classes[train_index]
                c_val = classes[val_index]

                for c in C:
                    clf = LinearSVC(C=c, tol=0.01, dual=False)
                    clf.fit(train, c_train)
                    val_acc = accuracy_score(c_val, clf.predict(val)) * 100.0

                    if val_acc > best_val_acc:
                        # Get test acc.
                        best_val_acc = val_acc
                        best_c = c
                        best_gram_matrix = gram_matrix

            # Determine test accuracy.
            train = best_gram_matrix[train_index]
            test = best_gram_matrix[test_index]

            c_train = classes[train_index]
            c_test = classes[test_index]
            clf = LinearSVC(C=best_c, tol=0.01, dual=False)
            clf.fit(train, c_train)
            best_test = accuracy_score(c_test, clf.predict(test)) * 100.0

            test_accuracies.append(best_test)
            if all_std:
                test_accuracies_complete.append(best_test)
        test_accuracies_all.append(float(np.asarray(test_accuracies).mean()))

    test_accuracies_all = np.asarray(test_accuracies_all)
    test_accuracies_complete = np.asarray(test_accuracies_complete)

    if all_std:
        return (test_accuracies_all.mean(), test_accuracies_all.std(), test_accuracies_complete.std())
    else:
        return (test_accuracies_all.mean(), test_accuracies_all.std())

"""
# 10-CV for kernel svm and hyperparameter selection.
def kernel_svm_evaluation(all_matrices, classes, num_repetitions=10,
                          C=[10 ** 3, 10 ** 2, 10 ** 1, 10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3]):
    # Acc. over all repetitions.
    train_accuracies_all = []
    test_accuracies_all = []

    for i in range(num_repetitions):
        # Test acc. over all folds.
        train_accuracies = []
        test_accuracies = []
        kf = KFold(n_splits=10, shuffle=True)

        for train_index, test_index in kf.split(list(range(len(classes)))):
            # Determine hyperparameters
            train_index, val_index = train_test_split(train_index, test_size=0.1)
            best_val_acc = 0.0
            best_gram_matrix = all_matrices[0]
            best_c = C[0]

            for gram_matrix in all_matrices:
                train = gram_matrix[train_index, :]
                train = train[:, train_index]
                val = gram_matrix[val_index, :]
                val = val[:, train_index]

                c_train = classes[train_index]
                c_val = classes[val_index]

                for c in C:
                    clf = SVC(C=c, kernel="precomputed", tol=0.001)
                    clf.fit(train, c_train)
                    val_acc = accuracy_score(c_val, clf.predict(val)) * 100.0

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_c = c
                        best_gram_matrix = gram_matrix

            # Determine test accuracy.
            train = best_gram_matrix[train_index, :]
            train = train[:, train_index]
            test = best_gram_matrix[test_index, :]
            test = test[:, train_index]

            c_train = classes[train_index]
            c_test = classes[test_index]
            clf = SVC(C=best_c, kernel="precomputed", tol=0.001)
            clf.fit(train, c_train)
            best_test = accuracy_score(c_test, clf.predict(test)) * 100.0
            best_train = accuracy_score(c_train, clf.predict(train)) * 100.0

            test_accuracies.append(best_test)
            train_accuracies.append(best_train)

        test_accuracies_all.append(float(np.asarray(test_accuracies).mean()))
        train_accuracies_all.append(float(np.asarray(train_accuracies).mean()))
    
    train_accuracies_all = np.asarray(train_accuracies_all)
    test_accuracies_all = np.asarray(test_accuracies_all)

    return (train_accuracies_all.mean(), train_accuracies_all.std(), test_accuracies_all.mean(), test_accuracies_all.std())
"""

def kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, num_folds=10, C=None):
    if C is None:
        C = np.power(10.0, np.arange(3.0, -4.0, -1))
    num_iterations = len(all_matrices)
    # Acc. over all repetitions.
    train_accuracies_all = np.zeros((num_repetitions, num_folds, num_iterations))
    val_accuracies_all = np.zeros((num_repetitions, num_folds, num_iterations))
    test_accuracies_all = np.zeros((num_repetitions, num_folds, num_iterations))

    for i in range(num_repetitions):
        kf = KFold(n_splits=num_folds, random_state=i, shuffle=True)

        for j, (train_index, test_index) in enumerate(kf.split(classes)):
            train_index, val_index = train_test_split(train_index, test_size=0.1, random_state=i, shuffle=True)
            
            test_matrices = [gram_matrix[test_index, :][:, train_index] for gram_matrix in all_matrices]
            c_test = classes[test_index]

            # Determine hyperparameters
            for k, gram_matrix in enumerate(all_matrices):
                train = gram_matrix[train_index, :][:, train_index]
                val = gram_matrix[val_index, :][:, train_index]

                c_train = classes[train_index]
                c_val = classes[val_index]

                best_val_acc = 0.0
                best_c = C[0]
                best_clf = None
                for c in C:
                    clf = SVC(C=c, kernel="precomputed", tol=0.001, random_state=i)
                    clf.fit(train, c_train)
                    val_acc = accuracy_score(c_val, clf.predict(val)) * 100.0
                    # train_acc = accuracy_score(c_train, clf.predict(train)) * 100.0
                    # train_params_accs[k].append(train_acc)
                    # val_params_accs[k].append(val_acc)
                    # test_acc = accuracy_score(c_test, clf.predict(test)) * 100.0
                    # test_params_accs[k].append(test_acc)
                    if best_val_acc < val_acc:
                        best_val_acc = val_acc
                        best_c = c
                        best_clf = clf
                
                val_accuracies_all[i, j, k] = best_val_acc
                train_accuracies_all[i, j, k] = accuracy_score(c_train, best_clf.predict(train)) * 100.0
                test_accuracies_all[i, j, k] = accuracy_score(c_test, best_clf.predict(test_matrices[k])) * 100.0

    return train_accuracies_all, val_accuracies_all, test_accuracies_all
