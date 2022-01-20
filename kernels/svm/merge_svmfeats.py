import argparse
import os

def read_svmfeat_file(file_name):
    labels = []
    feats = []
    with open(file_name, "r") as f:
        for line in f:
            line = line.strip().split(" ", 1)
            labels.append(line[0])
            feat = dict()
            for x in line[1].strip().split(" "):
                x = x.split(":")
                feat[int(x[0])] = x[1]
            feats.append(feat)
    return labels, feats


def write_svmfeat_file(labels, feats, file_name):
    with open(file_name, "w") as f:
        for label, feat in zip(labels, feats):
            f.write(label)
            f.write(" ")
            f.write(" ".join(["%d:%s" % (x) for x in feat.items()]))
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_feat_files", nargs="+", default="../GM/MUTAG")
    parser.add_argument("--save_feat_file", type=str, default="../GM/ENSEMBLE-MUTAG")
    args = parser.parse_args()

    assert len(args.load_feat_files) > 0
    labels, feats = read_svmfeat_file(args.load_feat_files[0])
    for file_name in args.load_feat_files[1:]:
        labels_, feats_ = read_svmfeat_file(file_name)
        assert labels == labels_
        start_idx = len(feats)
        for i, feat in enumerate(feats_):
            for k, v in feat.items():
                if k + start_idx in feats[i]:
                    continue
                feats[i][k + start_idx] = v

    write_svmfeat_file(labels, feats, args.save_feat_file)