#include "vector.hpp"
#include <algorithm>
#include <fstream>

float VectorIndex::calculateCosineSimilarity(const std::vector<float> &a, const std::vector<float> &b)
{
    if (a.empty() || a.size() != b.size())
    {
        return 0.0f;
    }

    float dot_product = 0.0f; // Numerator: dot product (sum of A_i * B_i)
    float norm_a = 0.0f;      // Denominator part: sum of squared elements of vector A
    float norm_b = 0.0f;      // Denominator part: sum of squared elements of vector B

    for (size_t i = 0; i < a.size(); ++i)
    {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    if (norm_a == 0.0f || norm_b == 0.0f)
    {
        return 0.0f;
    }

    // Final formula: dot product / product of vector norms
    return dot_product / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

void VectorIndex::add_vector(int id, const std::vector<float> &vec)
{
    ids_.push_back(id);
    vectors_.push_back(vec);
}

std::vector<int> VectorIndex::search(const std::vector<float> &query_vec, int top_k)
{
    std::vector<std::pair<float, int>> scored_docs;
    for (size_t i = 0; i < vectors_.size(); ++i)
    {
        const auto &doc_vec = vectors_[i];
        int doc_id = ids_[i];

        float score = calculateCosineSimilarity(query_vec, doc_vec);

        scored_docs.push_back({score, doc_id});
    }

    // Sort descending so the highest-scoring documents come first
    std::sort(scored_docs.begin(), scored_docs.end(), [](const auto &a, const auto &b)
              { return a.first > b.first; });

    std::vector<int> result_ids;
    // Take at most top_k results or the total number of stored documents
    int limit = std::min(static_cast<int>(scored_docs.size()), top_k);

    for (int i = 0; i < limit; ++i)
    {
        result_ids.push_back(scored_docs[i].second); // Keep only the document ID
    }

    return result_ids;
}

void VectorIndex::save_to_file(const std::string &filepath)
{
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open())
    {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }

    size_t size = ids_.size();
    out.write(reinterpret_cast<const char *>(&size), sizeof(size));

    for (size_t i = 0; i < vectors_.size(); ++i)
    {
        out.write(reinterpret_cast<const char *>(&ids_[i]), sizeof(ids_[i]));

        size_t vec_size = vectors_[i].size();
        out.write(reinterpret_cast<const char *>(&vec_size), sizeof(vec_size));

        // Write raw float bytes from memory (.data())
        out.write(reinterpret_cast<const char *>(vectors_[i].data()), vec_size * sizeof(float));
    }
}

void VectorIndex::load_from_file(const std::string &filepath)
{
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open())
    {
        throw std::runtime_error("Failed to open file for reading: " + filepath);
    }

    // Clear existing in-memory data before loading new records
    ids_.clear();
    vectors_.clear();

    // 1. Read total number of documents
    size_t size = 0;
    in.read(reinterpret_cast<char *>(&size), sizeof(size));

    // Pre-allocate vectors to avoid repeated reallocations
    ids_.resize(size);
    vectors_.resize(size);

    // 2. Read each document entry back into memory
    for (size_t i = 0; i < size; ++i)
    {
        // Read document ID
        in.read(reinterpret_cast<char *>(&ids_[i]), sizeof(ids_[i]));

        // Read vector dimension
        size_t vec_size = 0;
        in.read(reinterpret_cast<char *>(&vec_size), sizeof(vec_size));

        // Resize target vector and read raw float data
        vectors_[i].resize(vec_size);
        in.read(reinterpret_cast<char *>(vectors_[i].data()), vec_size * sizeof(float));
    }
}