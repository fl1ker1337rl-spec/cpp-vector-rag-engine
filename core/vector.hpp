#include <vector>
#include <string>

class VectorIndex
{
public:
    VectorIndex() = default;
    ~VectorIndex() = default;

    static float calculateCosineSimilarity(const std::vector<float> &a, const std::vector<float> &b);
    void add_vector(int id, const std::vector<float> &vec);
    std::vector<int> search(const std::vector<float> &query_vec, int top_k);
    void save_to_file(const std::string &filepath);
    void load_from_file(const std::string &filepath);

private:
    std::vector<int> ids_;
    std::vector<std::vector<float>> vectors_;

    VectorIndex(const VectorIndex &) = delete;
    VectorIndex &operator=(const VectorIndex &) = delete;
    VectorIndex(VectorIndex &&) = delete;
    VectorIndex &operator=(const VectorIndex &&) = delete;
};