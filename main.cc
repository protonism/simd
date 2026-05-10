#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include <omp.h>
#include "hnswlib/hnswlib/hnswlib.h"
#include "flat_scan.h"
// 可以自行添加需要的头文件
#include <cstdint>
#include <utility>
#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <climits>

#include <limits>
#include <cstdlib>

using namespace hnswlib;

inline float inner_product_neon(const float *a, const float *b, size_t dim)
{
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);

    size_t d = 0;

    for (; d + 15 < dim; d += 16)
    {
        float32x4_t a0 = vld1q_f32(a + d);
        float32x4_t b0 = vld1q_f32(b + d);
        sum0 = vmlaq_f32(sum0, a0, b0);

        float32x4_t a1 = vld1q_f32(a + d + 4);
        float32x4_t b1 = vld1q_f32(b + d + 4);
        sum1 = vmlaq_f32(sum1, a1, b1);

        float32x4_t a2 = vld1q_f32(a + d + 8);
        float32x4_t b2 = vld1q_f32(b + d + 8);
        sum2 = vmlaq_f32(sum2, a2, b2);

        float32x4_t a3 = vld1q_f32(a + d + 12);
        float32x4_t b3 = vld1q_f32(b + d + 12);
        sum3 = vmlaq_f32(sum3, a3, b3);
    }

    float32x4_t sum01 = vaddq_f32(sum0, sum1);
    float32x4_t sum23 = vaddq_f32(sum2, sum3);
    float32x4_t sum = vaddq_f32(sum01, sum23);

    float tmp[4];
    vst1q_f32(tmp, sum);

    float result = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    // 处理维度不是 16 的倍数时剩余的元素
    // 本实验 vecdim = 96，理论上不会进入该循环
    for (; d < dim; ++d)
    {
        result += a[d] * b[d];
    }

    return result;
}

// 根据 DEEP100K 数据集维度固定为 96 的特点，进一步将 SIMD 内积函数特化为 96 维
inline float inner_product_neon_96(const float *a, const float *b)
{
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);

    for (int d = 0; d < 96; d += 16)
    {
        float32x4_t a0 = vld1q_f32(a + d);
        float32x4_t b0 = vld1q_f32(b + d);
        sum0 = vmlaq_f32(sum0, a0, b0);

        float32x4_t a1 = vld1q_f32(a + d + 4);
        float32x4_t b1 = vld1q_f32(b + d + 4);
        sum1 = vmlaq_f32(sum1, a1, b1);

        float32x4_t a2 = vld1q_f32(a + d + 8);
        float32x4_t b2 = vld1q_f32(b + d + 8);
        sum2 = vmlaq_f32(sum2, a2, b2);

        float32x4_t a3 = vld1q_f32(a + d + 12);
        float32x4_t b3 = vld1q_f32(b + d + 12);
        sum3 = vmlaq_f32(sum3, a3, b3);
    }

    float32x4_t sum = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));

    float tmp[4];
    vst1q_f32(tmp, sum);

    return tmp[0] + tmp[1] + tmp[2] + tmp[3];
}

// pq-simd
inline float hsum_f32x4(float32x4_t v)
{
    float tmp[4];
    vst1q_f32(tmp, v);
    return tmp[0] + tmp[1] + tmp[2] + tmp[3];
}

inline float l2_neon(const float *a, const float *b, size_t dim)
{
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);

    size_t d = 0;
    for (; d + 7 < dim; d += 8)
    {
        float32x4_t da0 = vsubq_f32(vld1q_f32(a + d), vld1q_f32(b + d));
        float32x4_t da1 = vsubq_f32(vld1q_f32(a + d + 4), vld1q_f32(b + d + 4));

        sum0 = vmlaq_f32(sum0, da0, da0);
        sum1 = vmlaq_f32(sum1, da1, da1);
    }

    float result = hsum_f32x4(vaddq_f32(sum0, sum1));

    for (; d < dim; ++d)
    {
        float diff = a[d] - b[d];
        result += diff * diff;
    }

    return result;
}

struct PQIndex
{
    size_t M;      // 子空间个数
    size_t Ks;     // 每个子空间中心数
    size_t subdim; // 每个子空间维度
    size_t vecdim;
    size_t base_number;

    // codebooks[(m * Ks + c) * subdim + j]
    std::vector<float> codebooks;

    // codes[i * M + m]
    std::vector<uint8_t> codes;
};

inline const float *centroid_ptr(const PQIndex &idx, size_t m, size_t c)
{
    return idx.codebooks.data() + (m * idx.Ks + c) * idx.subdim;
}

inline uint8_t find_nearest_centroid_l2(
    const float *subvec,
    const float *centroids,
    size_t Ks,
    size_t subdim)
{
    float best_dis = std::numeric_limits<float>::max();
    uint8_t best_id = 0;

    for (size_t c = 0; c < Ks; ++c)
    {
        float dis = l2_neon(subvec, centroids + c * subdim, subdim);
        if (dis < best_dis)
        {
            best_dis = dis;
            best_id = static_cast<uint8_t>(c);
        }
    }

    return best_id;
}

void train_one_subspace_kmeans(
    const float *base,
    size_t base_number,
    size_t vecdim,
    size_t m,
    size_t M,
    size_t Ks,
    size_t subdim,
    size_t train_n,
    int iters,
    float *centroids)
{
    size_t stride = std::max<size_t>(1, base_number / train_n);

    // 初始化：从 base 中抽取 Ks 个子向量作为初始中心
    for (size_t c = 0; c < Ks; ++c)
    {
        size_t id = (c * stride * 7 + c * 13) % base_number;
        const float *src = base + id * vecdim + m * subdim;
        std::memcpy(centroids + c * subdim, src, sizeof(float) * subdim);
    }

    std::vector<float> sums(Ks * subdim);
    std::vector<int> counts(Ks);

    for (int it = 0; it < iters; ++it)
    {
        std::fill(sums.begin(), sums.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);

        for (size_t t = 0; t < train_n; ++t)
        {
            size_t id = (t * stride) % base_number;
            const float *subvec = base + id * vecdim + m * subdim;

            uint8_t cid = find_nearest_centroid_l2(subvec, centroids, Ks, subdim);
            counts[cid]++;

            float *sum = sums.data() + cid * subdim;
            for (size_t j = 0; j < subdim; ++j)
            {
                sum[j] += subvec[j];
            }
        }

        for (size_t c = 0; c < Ks; ++c)
        {
            if (counts[c] == 0)
            {
                continue;
            }

            float inv = 1.0f / counts[c];
            for (size_t j = 0; j < subdim; ++j)
            {
                centroids[c * subdim + j] = sums[c * subdim + j] * inv;
            }
        }
    }
}

PQIndex build_pq_index(
    const float *base,
    size_t base_number,
    size_t vecdim,
    size_t M = 8,
    size_t Ks = 256,
    size_t train_n = 12000,
    int iters = 6)
{
    if (vecdim % M != 0)
    {
        std::cerr << "PQ error: vecdim must be divisible by M\n";
        std::exit(1);
    }

    PQIndex idx;
    idx.M = M;
    idx.Ks = Ks;
    idx.vecdim = vecdim;
    idx.base_number = base_number;
    idx.subdim = vecdim / M;

    idx.codebooks.resize(M * Ks * idx.subdim);
    idx.codes.resize(base_number * M);

    train_n = std::min(train_n, base_number);

    std::cerr << "PQ build: M=" << M
              << " Ks=" << Ks
              << " subdim=" << idx.subdim
              << " train_n=" << train_n
              << " iters=" << iters << "\n";

    for (size_t m = 0; m < M; ++m)
    {
        std::cerr << "train subspace " << m << " / " << M << "\n";

        train_one_subspace_kmeans(
            base,
            base_number,
            vecdim,
            m,
            M,
            Ks,
            idx.subdim,
            train_n,
            iters,
            idx.codebooks.data() + m * Ks * idx.subdim);
    }

#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(base_number); ++i)
    {
        for (size_t m = 0; m < M; ++m)
        {
            const float *subvec = base + static_cast<size_t>(i) * vecdim + m * idx.subdim;
            const float *centroids = idx.codebooks.data() + m * Ks * idx.subdim;

            idx.codes[static_cast<size_t>(i) * M + m] =
                find_nearest_centroid_l2(subvec, centroids, Ks, idx.subdim);
        }
    }

    std::cerr << "PQ build done. code bytes = " << idx.codes.size() << "\n";

    return idx;
}

void build_adc_lut_simd(
    const PQIndex &idx,
    const float *query,
    std::vector<float> &lut)
{
    lut.resize(idx.M * idx.Ks);

    for (size_t m = 0; m < idx.M; ++m)
    {
        const float *qsub = query + m * idx.subdim;

        for (size_t c = 0; c < idx.Ks; ++c)
        {
            lut[m * idx.Ks + c] =
                inner_product_neon(qsub, centroid_ptr(idx, m, c), idx.subdim);
        }
    }
}

std::priority_queue<std::pair<float, uint32_t>> pq_search_adc_simd(
    const float *base,
    const PQIndex &idx,
    const float *query,
    size_t k,
    size_t top_p)
{
    //使用 static 让 lut 只分配一次内存
    static std::vector<float> lut;
    lut.reserve(idx.M * idx.Ks); // 避免重复分配
    build_adc_lut_simd(idx, query, lut);

    struct Candidate
    {
        float score;
        uint32_t id;
    };

    //使用 static 让 scores 数组变成持久化的缓存块
    static std::vector<Candidate> scores;
    if (scores.size() < idx.base_number)
    {
        scores.resize(idx.base_number);
    }

    size_t i = 0;
    
    // 跨向量并行Batching每次同时处理 4 条向量，让 CPU 预取和指令流水线满载
    if (idx.M == 8)
    {
        for (; i + 3 < idx.base_number; i += 4)
        {
            const uint8_t *c0 = idx.codes.data() + (i + 0) * idx.M;
            const uint8_t *c1 = idx.codes.data() + (i + 1) * idx.M;
            const uint8_t *c2 = idx.codes.data() + (i + 2) * idx.M;
            const uint8_t *c3 = idx.codes.data() + (i + 3) * idx.M;

            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

            //交错执行内存加载和加法
            s0 += lut[0 * idx.Ks + c0[0]]; s1 += lut[0 * idx.Ks + c1[0]]; s2 += lut[0 * idx.Ks + c2[0]]; s3 += lut[0 * idx.Ks + c3[0]];
            s0 += lut[1 * idx.Ks + c0[1]]; s1 += lut[1 * idx.Ks + c1[1]]; s2 += lut[1 * idx.Ks + c2[1]]; s3 += lut[1 * idx.Ks + c3[1]];
            s0 += lut[2 * idx.Ks + c0[2]]; s1 += lut[2 * idx.Ks + c1[2]]; s2 += lut[2 * idx.Ks + c2[2]]; s3 += lut[2 * idx.Ks + c3[2]];
            s0 += lut[3 * idx.Ks + c0[3]]; s1 += lut[3 * idx.Ks + c1[3]]; s2 += lut[3 * idx.Ks + c2[3]]; s3 += lut[3 * idx.Ks + c3[3]];
            s0 += lut[4 * idx.Ks + c0[4]]; s1 += lut[4 * idx.Ks + c1[4]]; s2 += lut[4 * idx.Ks + c2[4]]; s3 += lut[4 * idx.Ks + c3[4]];
            s0 += lut[5 * idx.Ks + c0[5]]; s1 += lut[5 * idx.Ks + c1[5]]; s2 += lut[5 * idx.Ks + c2[5]]; s3 += lut[5 * idx.Ks + c3[5]];
            s0 += lut[6 * idx.Ks + c0[6]]; s1 += lut[6 * idx.Ks + c1[6]]; s2 += lut[6 * idx.Ks + c2[6]]; s3 += lut[6 * idx.Ks + c3[6]];
            s0 += lut[7 * idx.Ks + c0[7]]; s1 += lut[7 * idx.Ks + c1[7]]; s2 += lut[7 * idx.Ks + c2[7]]; s3 += lut[7 * idx.Ks + c3[7]];

            scores[i + 0] = {s0, static_cast<uint32_t>(i + 0)};
            scores[i + 1] = {s1, static_cast<uint32_t>(i + 1)};
            scores[i + 2] = {s2, static_cast<uint32_t>(i + 2)};
            scores[i + 3] = {s3, static_cast<uint32_t>(i + 3)};
        }
    }

    // 尾部处理（处理不能被 4 整除剩余的向量）
    for (; i < idx.base_number; ++i)
    {
        const uint8_t *code = idx.codes.data() + i * idx.M;
        float score = 0.0f;

        if (idx.M == 8)
        {
            score += lut[0 * idx.Ks + code[0]];
            score += lut[1 * idx.Ks + code[1]];
            score += lut[2 * idx.Ks + code[2]];
            score += lut[3 * idx.Ks + code[3]];
            score += lut[4 * idx.Ks + code[4]];
            score += lut[5 * idx.Ks + code[5]];
            score += lut[6 * idx.Ks + code[6]];
            score += lut[7 * idx.Ks + code[7]];
        }
        else
        {
            for (size_t m = 0; m < idx.M; ++m)
            {
                score += lut[m * idx.Ks + code[m]];
            }
        }
        scores[i] = {score, static_cast<uint32_t>(i)};
    }

    if (top_p < k)
    {
        top_p = k;
    }

    if (top_p > scores.size())
    {
        top_p = scores.size();
    }

    if (top_p < scores.size())
    {
        std::nth_element(
            scores.begin(),
            scores.begin() + top_p,
            scores.end(),
            [](const Candidate &a, const Candidate &b)
            {
                return a.score > b.score;
            });
    }

    std::priority_queue<std::pair<float, uint32_t>> result_heap;

    for (size_t t = 0; t < top_p; ++t)
    {
        uint32_t id = scores[t].id;
        const float *base_vec = base + static_cast<size_t>(id) * idx.vecdim;

        float ip;
        if (idx.vecdim == 96)
        {
            ip = inner_product_neon_96(base_vec, query);
        }
        else
        {
            ip = inner_product_neon(base_vec, query, idx.vecdim);
        }

        float dis = 1.0f - ip;

        if (result_heap.size() < k)
        {
            result_heap.push({dis, id});
        }
        else if (dis < result_heap.top().first)
        {
            result_heap.push({dis, id});
            result_heap.pop();
        }
    }

    return result_heap;
}

template <typename T>
T *LoadData(std::string data_path, size_t &n, size_t &d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char *)&n, 4);
    fin.read((char *)&d, 4);
    T *data = new T[n * d];
    int sz = sizeof(T);
    for (int i = 0; i < n; ++i)
    {
        fin.read(((char *)data + i * d * sz), d * sz);
    }
    fin.close();

    std::cerr << "load data " << data_path << "\n";
    std::cerr << "dimension: " << d << "  number:" << n << "  size_per_element:" << sizeof(T) << "\n";

    return data;
}

struct SearchResult
{
    float recall;
    int64_t latency; // 单位us
};

void build_index(float *base, size_t base_number, size_t vecdim)
{
    const int efConstruction = 150; // 为防止索引构建时间过长，efc建议设置200以下
    const int M = 16;               // M建议设置为16以下

    HierarchicalNSW<float> *appr_alg;
    InnerProductSpace ipspace(vecdim);
    appr_alg = new HierarchicalNSW<float>(&ipspace, base_number, M, efConstruction);

    appr_alg->addPoint(base, 0);
#pragma omp parallel for
    for (int i = 1; i < base_number; ++i)
    {
        appr_alg->addPoint(base + 1ll * vecdim * i, i);
    }

    char path_index[1024] = "files/hnsw.index";
    appr_alg->saveIndex(path_index);
}

int main(int argc, char *argv[])
{
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "/anndata/";
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);

    PQIndex pq = build_pq_index(base, base_number, vecdim, 8, 256, 12000, 6);

    // 只测试前2000条查询
    test_number = 2000;

    const size_t k = 10;

    std::vector<SearchResult> results;
    results.resize(test_number);

    // 如果你需要保存索引，可以在这里添加你需要的函数，你可以将下面的注释删除来查看pbs是否将build.index返回到你的files目录中
    // 要保存的目录必须是files/*
    // 每个人的目录空间有限，不需要的索引请及时删除，避免占空间太大
    // 不建议在正式测试查询时同时构建索引，否则性能波动会较大
    // 下面是一个构建hnsw索引的示例
    // build_index(base, base_number, vecdim);

    // 查询测试代码
    for (int i = 0; i < test_number; ++i)
    {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        int ret = gettimeofday(&val, NULL);

        // 该文件已有代码中你只能修改该函数的调用方式
        // 可以任意修改函数名，函数参数或者改为调用成员函数，但是不能修改函数返回值。
        const size_t top_p = 1000;

        auto res = pq_search_adc_simd(
            base,
            pq,
            test_query + static_cast<size_t>(i) * vecdim,
            k,
            top_p);

        struct timeval newVal;
        ret = gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        std::set<uint32_t> gtset;
        for (int j = 0; j < k; ++j)
        {
            int t = test_gt[j + i * test_gt_d];
            gtset.insert(t);
        }

        size_t acc = 0;
        while (res.size())
        {
            int x = res.top().second;
            if (gtset.find(x) != gtset.end())
            {
                ++acc;
            }
            res.pop();
        }
        float recall = (float)acc / k;

        results[i] = {recall, diff};
    }

    float avg_recall = 0, avg_latency = 0;
    for (int i = 0; i < test_number; ++i)
    {
        avg_recall += results[i].recall;
        avg_latency += results[i].latency;
    }

    // 浮点误差可能导致一些精确算法平均recall不是1
    std::cout << "average recall: " << avg_recall / test_number << "\n";
    std::cout << "average latency (us): " << avg_latency / test_number << "\n";
    return 0;
}
