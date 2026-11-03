#include "vector.hpp"
#include <algorithm>
#include <fstream>

float VectorIndex::calculateCosineSimilarity(const std::vector<float> &a, const std::vector<float> &b)
{
    
    if (a.empty() || a.size() != b.size())
    {
        return 0.0f;
    }

    float dot_product = 0.0f; // Числитель: скалярное произведение (сумма A_i * B_i)
    float norm_a = 0.0f;      // Часть знаменателя: сумма квадратов элементов вектора A
    float norm_b = 0.0f;      // Часть знаменателя: сумма квадратов элементов вектора B

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

    std::sort(scored_docs.begin(), scored_docs.end(), [](const auto &a, const auto &b)
              { return a.first > b.first; });

    std::vector<int> result_ids;
    // Берем минимум: либо top_k, либо сколько всего документов есть в базе
    int limit = std::min(static_cast<int>(scored_docs.size()), top_k);

    for (int i = 0; i < limit; ++i)
    {
        result_ids.push_back(scored_docs[i].second); // забираем только ID
    }

    return result_ids;
}

void VectorIndex::save_to_file(const std::string &filepath)
{
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open())
    {
        throw std::runtime_error("Не удалось открыть файл для записи" + filepath);
    }

    size_t size = ids_.size();
    out.write(reinterpret_cast<const char *>(&size), sizeof(size));

    for (size_t i = 0; i < vectors_.size(); ++i)
    {
        out.write(reinterpret_cast<const char *>(&ids_[i]), sizeof(ids_[i]));

        size_t vec_size = vectors_[i].size();
        out.write(reinterpret_cast<const char *>(&vec_size), sizeof(vec_size));

        // Сохраняем массив float-чисел вектора «сырыми» байтами из памяти (.data())
        out.write(reinterpret_cast<const char *>(vectors_[i].data()), vec_size * sizeof(float));
    }
}

void VectorIndex::load_from_file(const std::string &filepath)
{
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open())
    {
        throw std::runtime_error("Не удалось открыть файл для чтения: " + filepath);
    }

    ids_.clear();
    vectors_.clear();

    // Читаем общее количество документов
    size_t size = 0;
    in.read(reinterpret_cast<char *>(&size), sizeof(size));

    ids_.resize(size);
    vectors_.resize(size);

    // Читаем каждый документ обратно
    for (size_t i = 0; i < size; ++i)
    {
        // Читаем ID
        in.read(reinterpret_cast<char *>(&ids_[i]), sizeof(ids_[i]));

        // Читаем размер вектора
        size_t vec_size = 0;
        in.read(reinterpret_cast<char *>(&vec_size), sizeof(vec_size));

        // Выделяем память под сам вектор и читаем данные в его память
        vectors_[i].resize(vec_size);
        in.read(reinterpret_cast<char *>(vectors_[i].data()), vec_size * sizeof(float));
    }
}
