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


inline float inner_product_neon(const float* a, const float* b, size_t dim)
{
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);

    size_t d = 0;

    for (; d + 15 < dim; d += 16) {
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
    for (; d < dim; ++d) {
        result += a[d] * b[d];
    }

    return result;
}

//根据 DEEP100K 数据集维度固定为 96 的特点，进一步将 SIMD 内积函数特化为 96 维
inline float inner_product_neon_96(const float* a, const float* b)
{
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);

    for (int d = 0; d < 96; d += 16) {
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


//pq-simd
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


//FastScan需要的 SoA 数据结构
struct FastScanIndex
{
    size_t M;      
    size_t Ks;     // 16
    size_t subdim; 
    size_t vecdim;
    size_t base_number;
    size_t padded_base_number; // 为了16路对齐填充

    std::vector<float> codebooks;
    
    // SoA 布局在搜索时一条指令加载 16 条不同向量在同一个子空间
    std::vector<uint8_t> codes_soa;
};


inline const float *centroid_ptr(const FastScanIndex &idx, size_t m, size_t c)
{
    return idx.codebooks.data() + (m * idx.Ks + c) * idx.subdim;
}


inline uint8_t find_nearest_centroid_l2(const float *subvec, const float *centroids, size_t Ks, size_t subdim)
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
    const float *base, size_t base_number, size_t vecdim, size_t m,
    size_t M, size_t Ks, size_t subdim, size_t train_n, int iters, float *centroids)
{
    size_t stride = std::max<size_t>(1, base_number / train_n);
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
                sum[j] += subvec[j];
        }

        for (size_t c = 0; c < Ks; ++c)
        {
            if (counts[c] == 0) continue;
            float inv = 1.0f / counts[c];
            for (size_t j = 0; j < subdim; ++j)
                centroids[c * subdim + j] = sums[c * subdim + j] * inv;
        }
    }
}


FastScanIndex build_fastscan_index(
    const float *base, size_t base_number, size_t vecdim,
    size_t M = 16, size_t Ks = 16, size_t train_n = 12000, int iters = 6)
{
    if (Ks != 16) {
        std::cerr << "FastScan Error: Ks MUST be 16 for NEON register lookup.\n";
        std::exit(1);
    }

    FastScanIndex idx;
    idx.M = M;
    idx.Ks = Ks;
    idx.vecdim = vecdim;
    idx.base_number = base_number;
    idx.subdim = vecdim / M;
    
    // 为了每次处理 16 条向量，进行向上取整对齐
    idx.padded_base_number = (base_number + 15) / 16 * 16; 

    idx.codebooks.resize(M * Ks * idx.subdim);
    idx.codes_soa.resize(M * idx.padded_base_number, 0); // 默认填充 0

    train_n = std::min(train_n, base_number);
    std::cerr << "FastScan build: M=" << M << " Ks=" << Ks << " subdim=" << idx.subdim << "\n";

    for (size_t m = 0; m < M; ++m)
    {
        train_one_subspace_kmeans(
            base, base_number, vecdim, m, M, Ks, idx.subdim, train_n, iters,
            idx.codebooks.data() + m * Ks * idx.subdim);
    }

    // 编码阶段写入 SoA 布局
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(base_number); ++i)
    {
        for (size_t m = 0; m < M; ++m)
        {
            const float *subvec = base + static_cast<size_t>(i) * vecdim + m * idx.subdim;
            const float *centroids = idx.codebooks.data() + m * Ks * idx.subdim;
            
            // 此处的索引计算方式为m * padded_base_number + i
            idx.codes_soa[m * idx.padded_base_number + i] = 
                find_nearest_centroid_l2(subvec, centroids, Ks, idx.subdim);
        }
    }
    return idx;
}


//构建量化版的LUT使其适应 uint8_t 寄存器查表
void build_adc_lut_fastscan(
    const FastScanIndex &idx,
    const float *query,
    std::vector<uint8_t> &lut_uint8)
{
    lut_uint8.resize(idx.M * 16);
    std::vector<float> lut_float(idx.M * 16);
    
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();

    for (size_t m = 0; m < idx.M; ++m)
    {
        const float *qsub = query + m * idx.subdim;
        for (size_t c = 0; c < 16; ++c)
        {
            float ip = inner_product_neon(qsub, centroid_ptr(idx, m, c), idx.subdim);
            lut_float[m * 16 + c] = ip;
            if (ip < min_val) min_val = ip;
            if (ip > max_val) max_val = ip;
        }
    }

    //将浮点LUT量化到 0-255
    float range = max_val - min_val;
    if (range == 0.0f) range = 1.0f;
    float scale = 255.0f / range;

    for (size_t m = 0; m < idx.M; ++m)
    {
        for (size_t c = 0; c < 16; ++c)
        {
            float v = (lut_float[m * 16 + c] - min_val) * scale;
            lut_uint8[m * 16 + c] = static_cast<uint8_t>(std::round(v));
        }
    }
}


//利用 SIMD 寄存器内查表 vqtbl1q_u8进行Batching检索
std::priority_queue<std::pair<float, uint32_t>> fastscan_search_simd(
    const float *base,
    const FastScanIndex &idx,
    const float *query,
    size_t k,
    size_t top_p)
{
    std::vector<uint8_t> lut_u8;
    build_adc_lut_fastscan(idx, query, lut_u8);

    //放弃priority_queue改用扁平数组
    struct Candidate {
        uint16_t score;
        uint32_t id;
    };
    std::vector<Candidate> scores(idx.base_number);

    // 每次处理 16 条 base 向量
    for (size_t i = 0; i < idx.padded_base_number; i += 16)
    {
        uint16x8_t sum_lo = vdupq_n_u16(0);
        uint16x8_t sum_hi = vdupq_n_u16(0);

        for (size_t m = 0; m < idx.M; ++m)
        {
            uint8x16_t lut_reg = vld1q_u8(lut_u8.data() + m * 16);
            uint8x16_t idx_reg = vld1q_u8(idx.codes_soa.data() + m * idx.padded_base_number + i);
            uint8x16_t sub_scores = vqtbl1q_u8(lut_reg, idx_reg);
            
            sum_lo = vaddw_u8(sum_lo, vget_low_u8(sub_scores));
            sum_hi = vaddw_u8(sum_hi, vget_high_u8(sub_scores));
        }

        uint16_t batch_scores[16];
        vst1q_u16(batch_scores, sum_lo);
        vst1q_u16(batch_scores + 8, sum_hi);

        // 写入数组，没有任何分支和堆操作
        for (size_t j = 0; j < 16; ++j)
        {
            size_t actual_id = i + j;
            if (actual_id < idx.base_number) 
            {
                scores[actual_id] = {batch_scores[j], static_cast<uint32_t>(actual_id)};
            }
        }
    }

    // 利用 std::nth_element 一次性切分出 top_p
    size_t actual_top_p = std::min(top_p, scores.size());
    std::nth_element(
        scores.begin(),
        scores.begin() + actual_top_p,
        scores.end(),
        [](const Candidate &a, const Candidate &b) {
            return a.score > b.score; // 降序，分数越大越好
        });

    // 精排阶段：从 scores 的前 top_p 个元素中取 ID
    std::priority_queue<std::pair<float, uint32_t>> result_heap;

    for (size_t t = 0; t < actual_top_p; ++t)
    {
        uint32_t id = scores[t].id;
        const float *base_vec = base + static_cast<size_t>(id) * idx.vecdim;

        float ip = inner_product_neon_96(base_vec, query);
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



template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();

    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";

    return data;
}




struct SearchResult
{
    float recall;
    int64_t latency; // 单位us
};



int main(int argc, char *argv[])
{
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "/anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);


    FastScanIndex pq_fastscan = build_fastscan_index(base, base_number, vecdim, 16, 16, 12000, 6);


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
    for(int i = 0; i < test_number; ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        int ret = gettimeofday(&val, NULL);

        // 该文件已有代码中你只能修改该函数的调用方式
        // 可以任意修改函数名，函数参数或者改为调用成员函数，但是不能修改函数返回值。
        const size_t top_p = 2000;

        auto res = fastscan_search_simd(
            base,
            pq_fastscan,
            test_query + static_cast<size_t>(i) * vecdim,
            k,
            top_p);



        struct timeval newVal;
        ret = gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        std::set<uint32_t> gtset;
        for(int j = 0; j < k; ++j){
            int t = test_gt[j + i*test_gt_d];
            gtset.insert(t);
        }

        size_t acc = 0;
        while (res.size()) {   
            int x = res.top().second;
            if(gtset.find(x) != gtset.end()){
                ++acc;
            }
            res.pop();
        }
        float recall = (float)acc/k;

        results[i] = {recall, diff};
    }

    float avg_recall = 0, avg_latency = 0;
    for(int i = 0; i < test_number; ++i) {
        avg_recall += results[i].recall;
        avg_latency += results[i].latency;
    }

    // 浮点误差可能导致一些精确算法平均recall不是1
    std::cout << "average recall: "<<avg_recall / test_number<<"\n";
    std::cout << "average latency (us): "<<avg_latency / test_number<<"\n";
    return 0;
}
