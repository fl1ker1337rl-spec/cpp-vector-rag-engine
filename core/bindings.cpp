#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "vector.hpp"

namespace py = pybind11;

PYBIND11_MODULE(vector_engine, m)
{
    m.doc() = "High perfomance C++ Vector Search Engine for RAG";

    py::class_<VectorIndex>(m, "VectorIndex")
        .def(py::init<>())
        .def("add_vector", &VectorIndex::add_vector, "Add a vector with ID")
        .def("search", &VectorIndex::search, "Search top_k similar documents", py::arg("query"), py::arg("top_k"))
        .def("calculate_cosine_similarity", &VectorIndex::calculateCosineSimilarity, "Calculate cosine similarity")
        .def("save_to_file", &VectorIndex::save_to_file, "Save index to binary file")
        .def("load_from_file", &VectorIndex::load_from_file, "Load index from binary file");
}