#include "Graph.h"

namespace GraphLibrary {
Graph::Graph(const bool directed)
    : m_is_directed(directed),
      m_dummy_node(-1),
      m_num_nodes(0),
      m_num_edges(0),
      m_node_labels(),
      m_edge_labels(),
      m_vertex_id(),
      m_local(),
      m_node_to_two_tuple(),
      m_node_to_three_tuple() {}

Graph::Graph(const bool directed, const uint num_nodes, const EdgeList &edgeList, const Labels node_labels)
    : m_is_directed(directed),
      m_dummy_node(-1),
      m_adjacency_lists(),
      m_num_nodes(num_nodes),
      m_num_edges(edgeList.size()),
      m_node_labels(node_labels),
      m_edge_labels(),
      m_vertex_id(),
      m_local(),
      m_node_to_two_tuple(),
      m_node_to_three_tuple() {
    m_adjacency_lists.resize(num_nodes);

    for (auto const &e : edgeList) {
        add_edge(get<0>(e), get<1>(e));
    }
}

size_t Graph::add_node() {
    vector<Node> new_node;
    m_adjacency_lists.push_back(move(new_node));
    m_num_nodes++;

    return m_num_nodes - 1;
}

void Graph::add_edge(const Node &v, const Node &w) {
    if (!m_is_directed) {
        m_adjacency_lists[v].push_back(w);
        m_adjacency_lists[w].push_back(v);
    } else {
        m_adjacency_lists[v].push_back(w);
    }

    if (!m_is_directed) {
        m_num_edges++;
    } else {
        m_num_edges += 2;
    }
}

size_t Graph::add_dummy() {
    if (m_dummy_node == -1) {
        bool has_node_labels = (m_num_nodes == m_node_labels.size());
        bool has_node_attributes = (m_num_nodes == m_node_attributes.size());
        bool has_edge_labels = (m_num_edges == m_edge_labels.size());
        bool has_edge_attributes = (m_num_edges == m_edge_attributes.size());

        size_t dummy_node = add_node();
        if (has_node_labels && m_node_labels.size() != 0) {
            Label dummy_node_label = 0;
            if (*min_element(m_node_labels.begin(), m_node_labels.end()) == dummy_node_label) {
                Labels _m_node_labels;
                for (auto label : m_node_labels) {
                    _m_node_labels.push_back(label + 1);
                }
                m_node_labels.swap(_m_node_labels);
            }
            m_node_labels.push_back(dummy_node_label);
        } else {
            Label dummy_node_label = 0;
            Labels _m_node_labels(m_num_nodes, dummy_node_label + 1);
            m_node_labels.swap(_m_node_labels);
            m_node_labels[dummy_node] = dummy_node_label;
        }

        if (has_node_attributes && m_node_attributes.size() != 0) {
            Attribute dummy_node_attribute;
            for (uint i = 0; i < m_node_attributes.begin()->size(); ++i) {
                dummy_node_attribute.push_back(0);
            }
            m_node_attributes.push_back(dummy_node_attribute);
        }

        for (Node n = 0; n < dummy_node; ++n) {
            add_edge(dummy_node, n);
        }
        if (has_edge_labels && m_edge_labels.size() != 0) {
            Label dummy_edge_label = 0;
            bool flag = false;
            for (auto it = m_edge_labels.begin(); it != m_edge_labels.end(); ++it) {
                if (it->second == dummy_edge_label) {
                    flag = true;
                    break;
                }
            }
            if (flag) {
                EdgeLabels _m_edge_labels;
                for (auto label : m_edge_labels) {
                    _m_edge_labels.insert({{label.first, label.second + 1}});
                }
                m_edge_labels.swap(_m_edge_labels);
            }
            for (Node n = 0; n < dummy_node; ++n) {
                m_edge_labels.insert({{make_tuple(dummy_node, n), dummy_edge_label}});
            }
            if (!m_is_directed) {
                for (Node n = 0; n < dummy_node; ++n) {
                    m_edge_labels.insert({{make_tuple(dummy_node, n), dummy_edge_label}});
                    m_edge_labels.insert({{make_tuple(n, dummy_node), dummy_edge_label}});
                }
            } else {
                for (Node n = 0; n < dummy_node; ++n) {
                    m_edge_labels.insert({{make_tuple(dummy_node, n), dummy_edge_label}});
                }
            }
        } else {
            Label dummy_edge_label = 0;
            for (Node n = 0; n < dummy_node; ++n) {
                for (Node& v : m_adjacency_lists[n]) {
                    m_edge_labels.insert({{make_tuple(n, v), dummy_edge_label + 1}});
                }
            }
            if (!m_is_directed) {
                for (Node n = 0; n < dummy_node; ++n) {
                    m_edge_labels.insert({{make_tuple(dummy_node, n), dummy_edge_label}});
                    m_edge_labels.insert({{make_tuple(n, dummy_node), dummy_edge_label}});
                }
            } else {
                for (Node n = 0; n < dummy_node; ++n) {
                    m_edge_labels.insert({{make_tuple(dummy_node, n), dummy_edge_label}});
                }
            }
        }

        if (has_edge_attributes && m_edge_attributes.size() != 0) {
            Attribute dummy_edge_attribute;
            for (uint i = 0; i < m_edge_attributes.begin()->second.size(); ++i) {
                dummy_edge_attribute.push_back(0);
            }
            if (!m_is_directed) {
                for (Node n = 0; n < dummy_node; ++n) {
                    m_edge_attributes.insert({{make_tuple(dummy_node, n), dummy_edge_attribute}});
                    m_edge_attributes.insert({{make_tuple(n, dummy_node), dummy_edge_attribute}});
                }
            } else {
                for (Node n = 0; n < dummy_node; ++n) {
                    m_edge_attributes.insert({{make_tuple(dummy_node, n), dummy_edge_attribute}});
                }
            }
        }

        m_dummy_node = dummy_node;
        return dummy_node;
    } else {
        return m_dummy_node;
    }
}

size_t Graph::get_degree(const Node &v) const { return m_adjacency_lists[v].size(); }

Nodes Graph::get_neighbours(const Node &v) const { return m_adjacency_lists[v]; }

size_t Graph::get_num_nodes() const { return m_num_nodes; }

size_t Graph::get_num_edges() const { return m_num_edges; }

Node Graph::get_dummy() const { return m_dummy_node; }

uint Graph::has_edge(const Node &v, const Node &w) const {
    // This works for directed as well as undirected graphs.
    if (find(m_adjacency_lists[v].begin(), m_adjacency_lists[v].end(), w) != m_adjacency_lists[v].end()) {
        return 1;
    } else {
        return 0;
    }
}

Labels Graph::get_labels() const { return m_node_labels; }

void Graph::set_labels(Labels &labels) {
    // Copy labels.
    m_node_labels = labels;
}

Attributes Graph::get_attributes() const { return m_node_attributes; }

void Graph::set_attributes(Attributes &attributes) { m_node_attributes = attributes; }

void Graph::set_edge_labels(EdgeLabels &labels) {
    // Copy labels.
    m_edge_labels = labels;
}

EdgeLabels Graph::get_edge_labels() const { return m_edge_labels; }

void Graph::set_edge_attributes(EdgeAttributes &labels) {
    // Copy labels.
    m_edge_attributes = labels;
}

EdgeAttributes Graph::get_edge_attributes() const { return m_edge_attributes; }

EdgeLabels Graph::get_vertex_id() const { return m_vertex_id; }

void Graph::set_vertex_id(EdgeLabels &vertex_id) {
    // Copy labels.
    m_vertex_id = vertex_id;
}

EdgeLabels Graph::get_local() const { return m_local; }

void Graph::set_local(EdgeLabels &local) {
    // Copy labels.
    m_local = local;
}

void Graph::set_node_to_two_tuple(unordered_map<Node, TwoTuple> &n) { m_node_to_two_tuple = n; }

unordered_map<Node, TwoTuple> Graph::get_node_to_two_tuple() const { return m_node_to_two_tuple; }

void Graph::set_node_to_three_tuple(unordered_map<Node, ThreeTuple> &n) { m_node_to_three_tuple = n; }

unordered_map<Node, ThreeTuple> Graph::get_node_to_three_tuple() const { return m_node_to_three_tuple; }

Graph::~Graph() {}
}  // namespace GraphLibrary