#include <chrono>
#include <cstdio>
#include <iostream>

#include "src/AuxiliaryMethods.h"
#include "src/ColorRefinementKernel.h"
#include "src/GenerateThree.h"
#include "src/GenerateTwo.h"
#include "src/Graph.h"
#include "src/GraphletKernel.h"
#include "src/ShortestPathKernel.h"

using namespace std::chrono;
using namespace GraphLibrary;
using namespace std;

// template<typename T>
// std::ostream &operator<<(std::ostream &out, const std::vector<T> &v) {
//    if (!v.empty()) {
//        out << '[';
//        std::copy(v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
//        out << "\b\b]";
//    }
//    return out;
//}

unordered_map<string, tuple<string, bool, bool>> all_datasets = {
    {"ENZYMES", make_tuple("ENZYMES", true, false)},
    {"IMDB-BINARY", make_tuple("IMDB-BINARY", false, false)},
    {"IMDB-MULTI", make_tuple("IMDB-MULTI", false, false)},
    // {"IMDB-BINARY", make_tuple("IMDB-BINARY", true, false)},
    // {"IMDB-MULTI", make_tuple("IMDB-MULTI", true, false)},
    {"NCI1", make_tuple("NCI1", true, false)},
    {"NCI109", make_tuple("NCI109", true, false)},
    {"PTC_FM", make_tuple("PTC_FM", true, false)},
    {"PROTEINS", make_tuple("PROTEINS", true, false)},
    {"REDDIT-BINARY", make_tuple("REDDIT-BINARY", false, false)},
    {"Yeast", make_tuple("Yeast", true, true)},
    {"YeastH", make_tuple("YeastH", true, true)},
    {"UACC257", make_tuple("UACC257", true, true)},
    {"UACC257H", make_tuple("UACC257H", true, true)},
    {"OVCAR-8", make_tuple("OVCAR-8", true, true)},
    {"OVCAR-8H", make_tuple("OVCAR-8H", true, true)}};

int main(int argc, char **argv) {
    string dataset_dir = "./datasets";
    string gram_dir = "./svm/GM/EXP";
    uint k = 1;
    string kernel = "WL";
    uint n_iters = 1;
    vector<tuple<string, bool, bool>> datasets;
    bool add_dummy = false;

    uint i = 1;
    while (i < argc) {
        if (strcmp(argv[i], "--dataset_dir") == 0) {
            dataset_dir = string(argv[i + 1]);
            i += 2;
        } else if (strcmp(argv[i], "--gram_dir") == 0) {
            gram_dir = string(argv[i + 1]);
            i += 2;
        } else if (strcmp(argv[i], "--k") == 0) {
            k = atoi(argv[i + 1]);
            i += 2;
        } else if (strcmp(argv[i], "--kernel") == 0) {
            kernel = string(argv[i + 1]);
            i += 2;
        } else if (strcmp(argv[i], "--n_iters") == 0) {
            n_iters = atoi(argv[i + 1]);
            i += 2;
        } else if (strcmp(argv[i], "--datasets") == 0) {
            uint j = i + 1;
            while (j < argc) {
                if (strlen(argv[j]) > 2 && strncmp(argv[j], "--", 2) == 0) {
                    break;
                }
                string dataset = string(argv[j]);
                auto it = all_datasets.find(dataset);
                if (it != all_datasets.end()) {
                    datasets.push_back(it->second);
                } else {
                    cout << "Warning: " << dataset << " is not a valid dataset." << endl;
                }
                ++j;
            }
            i = j;
        } else if (strcmp(argv[i], "--add_dummy") == 0) {
            if ((i != argc - 1) && (strlen(argv[i+1]) > 2 && strncmp(argv[i+1], "--", 2) != 0)) {
                if (strlen(argv[i+1]) == 4 && ((strcmp(argv[i+1], "true") == 0) || (strcmp(argv[i+1], "True") == 0))) {
                    add_dummy = true;
                } else if (strlen(argv[i+1]) == 5 && ((strcmp(argv[i+1], "false") == 0) || strcmp(argv[i+1], "False") == 0)) {
                    add_dummy = false;
                } else {
                    cout << "Unknown args: " << argv[i] << " " << argv[i+1] << endl;
                }
                i += 2;
            } else {
                add_dummy = true;
                ++i;
            }
        } else {
            cout << "Unknown args: " << argv[i];
            ++i;
        }
    }

    vector<GraphDatabase> gdbs;
    vector<vector<int>> classes;
    for (auto &d : datasets) {
        string ds = std::get<0>(d);
        GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(dataset_dir, ds);
        gdb.erase(gdb.begin() + 0);
        gdbs.push_back(gdb);
        classes.push_back(AuxiliaryMethods::read_classes(dataset_dir, ds));
    }

    // add dummy
    if (add_dummy) {
        for (auto &gdb : gdbs) {
            for (auto &g : gdb) {
                g.add_dummy();
            }
        }
    }

    for (uint d = 0; d < datasets.size(); ++d) {
        GraphDatabase &gdb = gdbs[d];
        string &ds = std::get<0>(datasets[d]);
        bool use_labels = std::get<1>(datasets[d]) || add_dummy;
        bool use_edge_labels = std::get<2>(datasets[d]) || add_dummy;

        // WL

        if (k == 1) {
            if (kernel.compare("WL") == 0) {
                ColorRefinement::ColorRefinementKernel graph_kernel(gdb);

                for (uint i = 0; i <= n_iters; ++i) {
                    cout << ds + "__" + kernel + to_string(k) + "_" + to_string(i) << endl;

                    GramMatrix gm;
                    if (i == n_iters) {
                        high_resolution_clock::time_point t1 = high_resolution_clock::now();
                        gm = graph_kernel.compute_gram_matrix(i, use_labels, use_edge_labels, true, false);
                        high_resolution_clock::time_point t2 = high_resolution_clock::now();
                        auto duration = duration_cast<seconds>(t2 - t1).count();
                        cout << duration << endl;
                    } else {
                        gm = graph_kernel.compute_gram_matrix(i, use_labels, use_edge_labels, true, false);
                    }

                    AuxiliaryMethods::write_libsvm(
                        gm, classes[d],
                        gram_dir + "/" + ds + "__" + kernel + to_string(k) + "_" + to_string(i) + ".gram");
                }
            } else if (kernel.compare("WLOA") == 0) {
                ColorRefinement::ColorRefinementKernel graph_kernel(gdb);

                // i starts from 1
                for (uint i = 1; i <= n_iters; ++i) {
                    cout << ds + "__" + kernel + "_" + to_string(i) << endl;

                    GramMatrix gm;
                    if (i == n_iters) {
                        high_resolution_clock::time_point t1 = high_resolution_clock::now();
                        gm = graph_kernel.compute_gram_matrix(i, use_labels, use_edge_labels, true, true);
                        high_resolution_clock::time_point t2 = high_resolution_clock::now();
                        auto duration = duration_cast<seconds>(t2 - t1).count();
                        cout << duration << endl;
                    } else {
                        gm = graph_kernel.compute_gram_matrix(i, use_labels, use_edge_labels, true, true);
                    }

                    AuxiliaryMethods::write_libsvm(gm, classes[d],
                                                   gram_dir + "/" + ds + "__" + kernel + "_" + to_string(i) + ".gram");
                }
            } else if (kernel.compare("SP") == 0) {
                ShortestPathKernel::ShortestPathKernel graph_kernel(gdb);

                cout << ds + "__" + kernel + "_" + to_string(0) << endl;

                GramMatrix gm;
                high_resolution_clock::time_point t1 = high_resolution_clock::now();
                gm = graph_kernel.compute_gram_matrix(use_labels, true);
                high_resolution_clock::time_point t2 = high_resolution_clock::now();
                auto duration = duration_cast<seconds>(t2 - t1).count();
                cout << duration << endl;

                AuxiliaryMethods::write_libsvm(gm, classes[d],
                                               gram_dir + "/" + ds + "__" + kernel + "_" + to_string(0) + ".gram");
            } else if (kernel.compare("GR") == 0) {
                GraphletKernel::GraphletKernel graph_kernel(gdb);

                cout << ds + "__" + kernel + "_" + to_string(0) << endl;

                GramMatrix gm;
                high_resolution_clock::time_point t1 = high_resolution_clock::now();
                gm = graph_kernel.compute_gram_matrix(use_labels, use_edge_labels, true);
                high_resolution_clock::time_point t2 = high_resolution_clock::now();
                auto duration = duration_cast<seconds>(t2 - t1).count();
                cout << duration << endl;

                AuxiliaryMethods::write_libsvm(gm, classes[d],
                                               gram_dir + "/" + ds + "__" + kernel + "_" + to_string(0) + ".gram");
            }
        } else if (k == 2) {
            if (kernel.find("WL") != string::npos) {
                string algorithm;
                if (kernel.compare("WL") == 0) {
                    algorithm = "wl";
                } else if (kernel.compare("DWL") == 0) {
                    algorithm = "malkin";
                } else if (kernel.compare("LWL") == 0) {
                    algorithm = "local";
                } else if (kernel.compare("LWLP") == 0) {
                    algorithm = "localp";
                } else {
                    throw kernel;
                }
                GenerateTwo::GenerateTwo graph_kernel(gdb);

                for (uint i = 0; i <= n_iters; ++i) {
                    cout << ds + "__" + kernel + to_string(k) + "_" + to_string(i) << endl;

                    GramMatrix gm;
                    if (i == n_iters) {
                        high_resolution_clock::time_point t1 = high_resolution_clock::now();
                        gm = graph_kernel.compute_gram_matrix(i, use_labels, use_edge_labels, algorithm, true, true);
                        high_resolution_clock::time_point t2 = high_resolution_clock::now();
                        auto duration = duration_cast<seconds>(t2 - t1).count();
                        cout << duration << endl;
                    } else {
                        gm = graph_kernel.compute_gram_matrix(i, use_labels, use_edge_labels, algorithm, true, true);
                    }

                    AuxiliaryMethods::write_libsvm(
                        gm, classes[d],
                        gram_dir + "/" + ds + "__" + kernel + to_string(k) + "_" + to_string(i) + ".gram");
                }
            }
        } else if (k == 3) {
            if (kernel.find("WL") != string::npos) {
                string algorithm;
                if (kernel.compare("WL") == 0) {
                    algorithm = "wl";
                } else if (kernel.compare("DWL") == 0) {
                    algorithm = "malkin";
                } else if (kernel.compare("LWL") == 0) {
                    algorithm = "local";
                } else if (kernel.compare("LWLP") == 0) {
                    algorithm = "localp";
                } else {
                    throw kernel;
                }
                GenerateThree::GenerateThree graph_kernel(gdb);

                for (uint i = 0; i <= n_iters; ++i) {
                    cout << ds + "__" + kernel + "_" + to_string(i) << endl;

                    GramMatrix gm;
                    if (i == n_iters) {
                        high_resolution_clock::time_point t1 = high_resolution_clock::now();
                        gm = graph_kernel.compute_gram_matrix(i, use_labels, use_edge_labels, algorithm, true, true);
                        high_resolution_clock::time_point t2 = high_resolution_clock::now();
                        auto duration = duration_cast<seconds>(t2 - t1).count();
                        cout << duration << endl;
                    } else {
                        gm = graph_kernel.compute_gram_matrix(i, use_labels, use_edge_labels, algorithm, true, true);
                    }

                    AuxiliaryMethods::write_libsvm(
                        gm, classes[d],
                        gram_dir + "/" + ds + "__" + kernel + to_string(k) + "_" + to_string(i) + ".gram");
                }
            }
        }
    }

    return 0;
}
