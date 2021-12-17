#include <cstdio>
#include "src/AuxiliaryMethods.h"
#include "src/ColorRefinementKernel.h"
#include "src/ShortestPathKernel.h"
#include "src/GraphletKernel.h"
#include "src/GenerateTwo.h"
#include "src/GenerateThree.h"
#include "src/Graph.h"
#include <iostream>
#include <chrono>

using namespace std::chrono;
using namespace GraphLibrary;
using namespace std;

//template<typename T>
//std::ostream &operator<<(std::ostream &out, const std::vector<T> &v) {
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

int main(int argc, char **argv)
{
    uint k = 1;
    string kernel = "WL";
    uint nfolds = 1;
    vector<tuple<string, bool, bool>> datasets;
    if (argc > 1)
    {
        k = stoi(argv[1]);
    }
    if (argc > 2)
    {
        kernel = string(argv[2]);
    }
    if (argc > 3)
    {
        nfolds = stoi(argv[3]);
    }
    if (argc > 4)
    {
        for (uint i = 4; i < argc; ++i)
        {
            string dataset = string(argv[i]);
            auto it = all_datasets.find(dataset);
            if (it != all_datasets.end())
            {
                datasets.push_back(it->second);
            }
        }
    }

    vector<GraphDatabase> gdbs;
    vector<vector<int>> classes;
    for (auto &d : datasets)
    {
        string ds = std::get<0>(d);
        GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
        gdb.erase(gdb.begin() + 0);
        gdbs.push_back(gdb);
        classes.push_back(AuxiliaryMethods::read_classes(ds));
    }

    // add dummy
    for (auto &gdb : gdbs)
    {
        for (auto &g : gdb)
        {
            g.add_dummy();
        }
    }

    for (uint d = 0; d < datasets.size(); ++d)
    {
        GraphDatabase &gdb = gdbs[d];
        string &ds = std::get<0>(datasets[d]);
        bool &use_node_labels = std::get<1>(datasets[d]);
        bool &use_edge_labels = std::get<2>(datasets[d]);

        // WL

        if (k == 1)
        {
            if (kernel.compare("WL") == 0)
            {
                ColorRefinement::ColorRefinementKernel graph_kernel(gdb);

                for (uint i = 0; i <= nfolds; ++i)
                {
                    cout << ds + "__" + kernel + to_string(k) + "_" + to_string(i) << endl;

                    GramMatrix gm;
                    if (i == nfolds)
                    {
                        high_resolution_clock::time_point t1 = high_resolution_clock::now();
                        gm = graph_kernel.compute_gram_matrix(i, use_node_labels, use_edge_labels, true, false);
                        high_resolution_clock::time_point t2 = high_resolution_clock::now();
                        auto duration = duration_cast<seconds>(t2 - t1).count();
                        cout << duration << endl;
                    }
                    else
                    {
                        gm = graph_kernel.compute_gram_matrix(i, use_node_labels, use_edge_labels, true, false);
                    }

                    AuxiliaryMethods::write_libsvm(gm, classes[d], "./svm/GM/EXP/" + ds + "__" + kernel + to_string(k) + "_" + to_string(i) + ".gram");
                }
            }
            else if (kernel.compare("WLOA") == 0)
            {
                ColorRefinement::ColorRefinementKernel graph_kernel(gdb);

                for (uint i = 0; i <= nfolds; ++i)
                {
                    cout << ds + "__" + kernel + "_" + to_string(i) << endl;

                    GramMatrix gm;
                    if (i == nfolds)
                    {
                        high_resolution_clock::time_point t1 = high_resolution_clock::now();
                        gm = graph_kernel.compute_gram_matrix(i, use_node_labels, use_edge_labels, true, true);
                        high_resolution_clock::time_point t2 = high_resolution_clock::now();
                        auto duration = duration_cast<seconds>(t2 - t1).count();
                        cout << duration << endl;
                    }
                    else
                    {
                        gm = graph_kernel.compute_gram_matrix(i, use_node_labels, use_edge_labels, true, true);
                    }

                    AuxiliaryMethods::write_libsvm(gm, classes[d], "./svm/GM/EXP/" + ds + "__" + kernel + "_" + to_string(i) + ".gram");
                }
            }
            else if (kernel.compare("SP") == 0)
            {
                ShortestPathKernel::ShortestPathKernel graph_kernel(gdb);

                cout << ds + "__" + kernel + "_" + to_string(0) << endl;

                GramMatrix gm;
                high_resolution_clock::time_point t1 = high_resolution_clock::now();
                gm = graph_kernel.compute_gram_matrix(use_node_labels, true);
                high_resolution_clock::time_point t2 = high_resolution_clock::now();
                auto duration = duration_cast<seconds>(t2 - t1).count();
                cout << duration << endl;

                AuxiliaryMethods::write_libsvm(gm, classes[d], "./svm/GM/EXP/" + ds + "__" + kernel + "_" + to_string(0) + ".gram");
            }
            else if (kernel.compare("GR") == 0)
            {
                GraphletKernel::GraphletKernel graph_kernel(gdb);

                cout << ds + "__" + kernel + "_" + to_string(0) << endl;

                GramMatrix gm;
                high_resolution_clock::time_point t1 = high_resolution_clock::now();
                gm = graph_kernel.compute_gram_matrix(use_node_labels, use_edge_labels, true);
                high_resolution_clock::time_point t2 = high_resolution_clock::now();
                auto duration = duration_cast<seconds>(t2 - t1).count();
                cout << duration << endl;

                AuxiliaryMethods::write_libsvm(gm, classes[d], "./svm/GM/EXP/" + ds + "__" + kernel + "_" + to_string(0) + ".gram");
            }
        }
        else if (k == 2)
        {
            if (kernel.find("WL") != std::string::npos)
            {
                string algorithm;
                if (kernel.compare("WL") == 0)
                {
                    algorithm = "wl";
                }
                else if (kernel.compare("DWL") == 0)
                {
                    algorithm = "malkin";
                }
                else if (kernel.compare("LWL") == 0)
                {
                    algorithm = "local";
                }
                else if (kernel.compare("LWLP") == 0)
                {
                    algorithm = "localp";
                }
                else
                {
                    throw kernel;
                }
                GenerateTwo::GenerateTwo graph_kernel(gdb);

                for (uint i = 0; i <= nfolds; ++i)
                {
                    cout << ds + "__" + kernel + to_string(k) + "_" + to_string(i) << endl;

                    GramMatrix gm;
                    if (i == nfolds)
                    {
                        high_resolution_clock::time_point t1 = high_resolution_clock::now();
                        gm = graph_kernel.compute_gram_matrix(i, use_node_labels, use_edge_labels, algorithm, true, true);
                        high_resolution_clock::time_point t2 = high_resolution_clock::now();
                        auto duration = duration_cast<seconds>(t2 - t1).count();
                        cout << duration << endl;
                    }
                    else
                    {
                        gm = graph_kernel.compute_gram_matrix(i, use_node_labels, use_edge_labels, algorithm, true, true);
                    }

                    AuxiliaryMethods::write_libsvm(gm, classes[d], "./svm/GM/EXP/" + ds + "__" + kernel + to_string(k) + "_" + to_string(i) + ".gram");
                }
            }
        }
        else if (k == 3)
        {
            if (kernel.find("WL") != std::string::npos)
            {
                string algorithm;
                if (kernel.compare("WL") == 0)
                {
                    algorithm = "wl";
                }
                else if (kernel.compare("DWL") == 0)
                {
                    algorithm = "malkin";
                }
                else if (kernel.compare("LWL") == 0)
                {
                    algorithm = "local";
                }
                else if (kernel.compare("LWLP") == 0)
                {
                    algorithm = "localp";
                }
                else
                {
                    throw kernel;
                }
                GenerateThree::GenerateThree graph_kernel(gdb);

                for (uint i = 0; i <= nfolds; ++i)
                {
                    cout << ds + "__" + kernel + "_" + to_string(i) << endl;

                    GramMatrix gm;
                    if (i == nfolds)
                    {
                        high_resolution_clock::time_point t1 = high_resolution_clock::now();
                        gm = graph_kernel.compute_gram_matrix(i, use_node_labels, use_edge_labels, algorithm, true, true);
                        high_resolution_clock::time_point t2 = high_resolution_clock::now();
                        auto duration = duration_cast<seconds>(t2 - t1).count();
                        cout << duration << endl;
                    }
                    else
                    {
                        gm = graph_kernel.compute_gram_matrix(i, use_node_labels, use_edge_labels, algorithm, true, true);
                    }

                    AuxiliaryMethods::write_libsvm(gm, classes[d], "./svm/GM/EXP/" + ds + "__" + kernel + to_string(k) + "_" + to_string(i) + ".gram");
                }
            }
        }
    }
    
    return 0;
}
