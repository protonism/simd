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


//sq-simd新增量化函数
inline int8_t quantize_to_int8(float x, float scale)
{
    int v = static_cast<int>(std::round(x * scale));

    if (v > 127) v = 127;
    if (v < -127) v = -127;

    return static_cast<int8_t>(v);
}



float build_sq_base(
    const float* base,
    size_t base_number,
    size_t vecdim,
    std::vector<int8_t>& base_sq
) {
    float max_abs = 0.0f;

    for (size_t i = 0; i < base_number * vecdim; ++i) {
        float v = std::fabs(base[i]);
        if (v > max_abs) {
            max_abs = v;
        }
    }

    if (max_abs == 0.0f) {
        max_abs = 1.0f;
    }

    float scale = 127.0f / max_abs;

    base_sq.resize(base_number * vecdim);

    for (size_t i = 0; i < base_number * vecdim; ++i) {
        base_sq[i] = quantize_to_int8(base[i], scale);
    }

    return scale;
}



std::priority_queue<std::pair<float, uint32_t> > flat_search_simd(
    float* base,
    float* query,
    size_t base_number,
    size_t vecdim,
    size_t k
) {
    std::priority_queue<std::pair<float, uint32_t> > q;

    for (size_t i = 0; i < base_number; ++i) {
        const float* base_vec = base + i * vecdim;

        float ip = inner_product_neon(base_vec, query,vecdim);
        float dis = 1.0f - ip;

        if (q.size() < k) {
            q.push({dis, static_cast<uint32_t>(i)});
        } else {
            if (dis < q.top().first) {
                q.push({dis, static_cast<uint32_t>(i)});
                q.pop();
            }
        }
    }

    return q;
}


//新增query量化函数
void quantize_query(
    const float* query,
    size_t vecdim,
    float scale,
    std::vector<int8_t>& query_sq
) {
    query_sq.resize(vecdim);

    for (size_t d = 0; d < vecdim; ++d) {
        query_sq[d] = quantize_to_int8(query[d], scale);
    }
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


//sq-simd计算内积
inline int32_t inner_product_int8_neon(
    const int8_t* a,
    const int8_t* b,
    size_t dim
) {
    int32x4_t sum0 = vdupq_n_s32(0);
    int32x4_t sum1 = vdupq_n_s32(0);

    size_t d = 0;

    for (; d + 15 < dim; d += 16) {
        int8x16_t va = vld1q_s8(a + d);
        int8x16_t vb = vld1q_s8(b + d);

        int8x8_t va_low = vget_low_s8(va);
        int8x8_t va_high = vget_high_s8(va);
        int8x8_t vb_low = vget_low_s8(vb);
        int8x8_t vb_high = vget_high_s8(vb);

        int16x8_t prod_low = vmull_s8(va_low, vb_low);
        int16x8_t prod_high = vmull_s8(va_high, vb_high);

        sum0 = vaddq_s32(sum0, vmovl_s16(vget_low_s16(prod_low)));
        sum0 = vaddq_s32(sum0, vmovl_s16(vget_high_s16(prod_low)));

        sum1 = vaddq_s32(sum1, vmovl_s16(vget_low_s16(prod_high)));
        sum1 = vaddq_s32(sum1, vmovl_s16(vget_high_s16(prod_high)));
    }

    int32x4_t sum = vaddq_s32(sum0, sum1);

    int32_t tmp[4];
    vst1q_s32(tmp, sum);

    int32_t result = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    for (; d < dim; ++d) {
        result += static_cast<int32_t>(a[d]) * static_cast<int32_t>(b[d]);
    }

    return result;
}



//sq-simd搜索函数
std::priority_queue<std::pair<float, uint32_t> > sq_search_simd(
    float* base,
    const int8_t* base_sq,
    float* query,
    size_t base_number,
    size_t vecdim,
    size_t k,
    size_t top_p,
    float scale
) {
    std::vector<int8_t> query_sq;
    quantize_query(query, vecdim, scale, query_sq);

    // coarse阶段要找approx_ip最大的top_p
    // priority_queue默认是大根堆，不方便直接维护最小score
    // 用greater做小根堆
    using Candidate = std::pair<int32_t, uint32_t>;

    struct MinScoreCmp {
        bool operator()(const Candidate& a, const Candidate& b) const {
            return a.first > b.first;
        }
    };

    std::priority_queue<
        Candidate,
        std::vector<Candidate>,
        MinScoreCmp
    > coarse_heap;

    for (size_t i = 0; i < base_number; ++i) {
        const int8_t* base_vec_sq = base_sq + i * vecdim;

        int32_t score = inner_product_int8_neon(
            base_vec_sq,
            query_sq.data(),
            vecdim
        );

        if (coarse_heap.size() < top_p) {
            coarse_heap.push({score, static_cast<uint32_t>(i)});
        } else {
            if (score > coarse_heap.top().first) {
                coarse_heap.pop();
                coarse_heap.push({score, static_cast<uint32_t>(i)});
            }
        }
    }


    std::vector<uint32_t> candidates;
    candidates.reserve(coarse_heap.size());

    while (!coarse_heap.empty()) {
        candidates.push_back(coarse_heap.top().second);
        coarse_heap.pop();
    }

    // rerank 阶段对候选集使用原始float向量重新计算精确距离
    std::priority_queue<std::pair<float, uint32_t> > result_heap;

    for (uint32_t id : candidates) {
        const float* base_vec = base + static_cast<size_t>(id) * vecdim;

        float ip = inner_product_neon(base_vec, query, vecdim);
        float dis = 1.0f - ip;

        if (result_heap.size() < k) {
            result_heap.push({dis, id});
        } else {
            if (dis < result_heap.top().first) {
                result_heap.push({dis, id});
                result_heap.pop();
            }
        }
    }

    return result_heap;
}



struct SearchResult
{
    float recall;
    int64_t latency; // 单位us
};

void build_index(float* base, size_t base_number, size_t vecdim)
{
    const int efConstruction = 150; // 为防止索引构建时间过长，efc建议设置200以下
    const int M = 16; // M建议设置为16以下

    HierarchicalNSW<float> *appr_alg;
    InnerProductSpace ipspace(vecdim);
    appr_alg = new HierarchicalNSW<float>(&ipspace, base_number, M, efConstruction);

    appr_alg->addPoint(base, 0);
    #pragma omp parallel for
    for(int i = 1; i < base_number; ++i) {
        appr_alg->addPoint(base + 1ll*vecdim*i, i);
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


    std::vector<int8_t> base_sq;
    float sq_scale = build_sq_base(base, base_number, vecdim, base_sq);
    std::cerr << "SQ build done. scale = " << sq_scale << "\n";


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
        const size_t top_p = 1000;
        auto res = sq_search_simd(
            base,
            base_sq.data(),
            test_query + i * vecdim,
            base_number,
            vecdim,
            k,
            top_p,
            sq_scale
        );


        
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
