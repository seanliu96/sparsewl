# Weisfeiler and Leman go sparse
Code for "_Weisfeiler and Leman go sparse: Towards scalable higher-order graph embeddings_" (NeurIPS 2020).

## Requirements
- `eigen3`
- `numpy`
- `pandas`
- `scipy`
- `sklearn`
- `torch>=1.5`
- `torch-geometric>=1.5`
- `libsvm`
- `pybind11`

All results in the paper and the appendix can be reproduced by the following the steps below. 

## Reproducing the kernel experiments (precomputed Gram matrices) (Tables 1, 2a, 3a, 5, 6, 8, 9)
- `cd kernels`
- Download datasets from `www.graphlearning.io`,  and place the unzipped folders into `kernels/datasets`
- Download `https://www.chrsmrrs.com/wl_goes_sparse_matrices/EXP.zip` and `https://www.chrsmrrs.com/wl_goes_sparse_matrices/EXPSPARSE.zip` and unzip them into `kernels/svm/GM`
- `cd svm`
- Run `python svm.py --dataset_dir ../datasets --gram_dir ../GM/EXP --k 1 --n_iters 5 --kernel WL --datasets ENZYMES`

## Reproducing the kernel experiments from scratch (Tables 1, 2a, 3a, 5, 6, 8, 9)
- `cd kernels`
- Download datasets from `www.graphlearning.io`,  and place the unzipped folders into `kernels/datasets`
- Run `g++ gram.cpp src/*cpp -std=c++11 -o gram -O2`
- Create a directory for gram matrices `mkdir ../GM/EXP -p`
- Run `./gram --dataset_dir ./datasets --gram_dir ../GM/EXP --k 1 --n_iters 5 --kernel WL --datasets ENZYMES` (running times will be outputted on the screen, too)
- `cd svm`
- Run `python svm.py --gram_dir ../GM/EXP --dataset_dir ../datasets --k 1 --n_iters 5 --kernel WL --datasets ENZYMES`

### New alternatives (faster)
- `cd kernels`
- Download datasets from `www.graphlearning.io`,  and place the unzipped folders into `kernels/datasets`
- Run `python kernel.py --gram_dir ./GM/EXP --dataset_dir ./datasets --k 1 --n_iters 5 --kernel WL --datasets ENZYMES`

## Reproducing the neural baselines (Tables 1, 5)
- `cd neural baselines`
- Run `python main_gnn.py`

## Reproducing the neural higher-order results (Table 2b, Figure 2abc, 3b, Table 7)
You first need to build the Python package:
- `cd neural_higher_order/preprocessing`
- You might need to adjust the path to `pybind` in `preprocessing.cpp`, then run 
    - MaxOS: c++ -O3 -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes`  preprocessing.cpp src/*cpp -o ../preprocessing`python3-config --extension-suffix` 
    - Linux: c++ -O3  -shared -std=c++11 -fPIC `python3 -m pybind11 --includes`  preprocessing.cpp src/*cpp -o ../preprocessing`python3-config --extension-suffix`

- Run the Python scripts in `Alchemy`, `QM9`, `ZINC` to reproduce the scores and running times
    - For example: `cd Alchemy`, `python local_2_FULL.py` to reproduce the scores for the \delta-2-LGNN on the  `Alchemy` dataset

