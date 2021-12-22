import argparse
import os
import sys
import datetime
import subprocess


class Logger(object):
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")
   
    def write(self, message):
        if isinstance(message, bytes):
            message = message.decode("utf-8")
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self, *args, **kw):
        self.terminal.flush()
        self.log.flush()

if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
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
        "--log_file", type=str,
        default=os.path.join(dirname, "log.txt"),
        help="the log file"
    )
    parser.add_argument(
        "--k", type=int,
        default=1,
        help="complexity of kernel functions"
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
        "--kernel", type=str,
        default="WL",
        help="kernel function"
    )
    parser.add_argument(
        "--add_dummy", type=str,
        default="false",
        help="whether to add dummy nodes and edges (true/false)"
    )
    parser.add_argument(
        "--datasets", nargs="+",
        help="datasets seperated by spaces (e.g., ENZYMES IMDB-BINARY MUTAG)"
    )
    args = parser.parse_args()
    args.gram_dir = os.path.abspath(args.gram_dir)
    args.dataset_dir = os.path.abspath(args.dataset_dir)
    args.log_file = os.path.abspath(args.log_file)

    if not os.path.exists(args.gram_dir):
        os.makedirs(args.gram_dir)

    ts = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    sys.stdout = Logger(args.log_file)
    print("-" * 80, flush=True)
    print(ts, flush=True)
    print(" ".join(sys.argv), flush=True)

    # run gram
    processes = []
    for dataset in args.datasets:
        proc = subprocess.Popen(
            [
                "./gram/gram.out",
                "--dataset_dir", args.dataset_dir, "--gram_dir", args.gram_dir,
                "--k", str(args.k), "--n_iters", str(args.n_iters),
                "--kernel", args.kernel,
                "--datasets", dataset, "--add_dummy", args.add_dummy
            ],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        for line in proc.stdout:
            print(line.decode("utf-8"), end="", flush=True)
        processes.append(proc)

    for proc in processes:
        proc.wait()
    processes.clear()

    print("-" * 80, flush=True)

    # run svm
    processes = []
    os.chdir("svm")
    for dataset in args.datasets:
        proc = subprocess.Popen(
            [
                "python",
                "new_svm.py",
                "--dataset_dir", args.dataset_dir, "--gram_dir", args.gram_dir,
                "--k", str(args.k), "--n_iters", str(args.n_iters),
                "--n_reps", str(args.n_reps), "--n_folds", str(args.n_folds),
                "--kernel", args.kernel,
                "--datasets", dataset
            ],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        for line in proc.stdout:
            print(line.decode("utf-8"), end="", flush=True)
        processes.append(proc)

    for proc in processes:
        proc.wait()
    processes.clear()

    print("=" * 80, flush=True)

    