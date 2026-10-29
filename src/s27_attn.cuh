// The 27B's attention split by heads (included by s27.cu): card c holds query heads [8c, 8c + 8) and the two KV heads
// they read (query head h reads KV head h / 6), K/V for every position (f16, [ctx][2][HD] a layer). No exchange
// between the cards until wo's sum. Rows of the projection: [q | gate] for each of the 8 heads, then the 2 K heads,
// then the 2 V heads (S27_QKV values).
#pragma once

#include <cuda_fp16.h>

namespace fnx {

constexpr int S27_NQ = NQH / NCARD, S27_NKV = 2;
constexpr int S27_QKV = S27_NQ * 2 * HD + 2 * S27_NKV * HD;
constexpr int S27_ASPLIT = 32;   // position splits of a row's attention
static_assert(NQH % NCARD == 0, "query heads split evenly");

__device__ __forceinline__ float s27_sum256(float v, float* sh) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float s = 0.f;
    for (int i = 0; i < HD / 32; ++i) s += sh[i];
    __syncthreads();
    return s;
}

// per (head, row): q heads normed and rotated in place; K heads normed, rotated and stored at the row's position; V
// heads stored. HD threads a block, one per value
__global__ void __launch_bounds__(HD) k_s27_prep(float* qkv, const float* qn, const float* kn, __half* kc, __half* vc,
                                                 const int* pos0p, float eps, Rope rope) {
    __shared__ float sh[HD / 32];
    __shared__ float x[HD];
    const int h = blockIdx.x, t = blockIdx.y, i = threadIdx.x, pos = *pos0p + t;
    float* row = qkv + (size_t) t * S27_QKV;
    float* p;
    const float* w;
    if (h < S27_NQ) p = row + h * 2 * HD, w = qn;
    else if (h < S27_NQ + S27_NKV) p = row + S27_NQ * 2 * HD + (h - S27_NQ) * HD, w = kn;
    else {
        const int vh = h - S27_NQ - S27_NKV;
        vc[((size_t) pos * S27_NKV + vh) * HD + i] = __float2half_rn(row[S27_NQ * 2 * HD + (S27_NKV + vh) * HD + i]);
        return;
    }
    const float v = p[i];
    const float ms = s27_sum256(v * v, sh) / (float) HD;
    x[i] = v * rsqrtf(ms + eps) * w[i];
    __syncthreads();
    if (i < NROT / 2) {
        const float theta = (float) rope_pos(rope, pos, i) * rope_unit_freq(rope, i, NROT);
        float sn, cs;
        sincosf(theta, &sn, &cs);
        sn *= rope.mscale, cs *= rope.mscale;
        const float x0 = x[i], x1 = x[i + NROT / 2];
        x[i] = x0 * cs - x1 * sn;
        x[i + NROT / 2] = x0 * sn + x1 * cs;
    }
    __syncthreads();
    if (h < S27_NQ) p[i] = x[i];
    else kc[((size_t) pos * S27_NKV + (h - S27_NQ)) * HD + i] = __float2half_rn(x[i]);
}

// per (split, row, head): the split's share of the row's positions [0, pos0 + t], four warps over them, each lane 8
// values of the head; the split's max, sum and weighted values into part ([T][NQ][S27_ASPLIT] of HD + 2)
__global__ void __launch_bounds__(128) k_s27_attn(const float* qkv, const __half* kc, const __half* vc, const int* pos0p, int kvh0,
                                                  int q0, float* part) {
    const int s = blockIdx.x, t = blockIdx.y, h = blockIdx.z, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int len = *pos0p + t + 1, chunk = (len + S27_ASPLIT - 1) / S27_ASPLIT;
    const int p0 = s * chunk, p1 = min(len, p0 + chunk);
    const int kvh = (q0 + h) / (NQH / NKVH) - kvh0;
    const float* q = qkv + (size_t) t * S27_QKV + h * 2 * HD + lane * 8;
    float qv[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) qv[j] = q[j] * 0.0625f;   // 1 / sqrt(HD)
    float m = -INFINITY, l = 0.f, acc[8] = {};
    for (int p = p0 + warp; p < p1; p += 4) {
        const size_t at = ((size_t) p * S27_NKV + kvh) * HD + lane * 8;
        const uint4 kr = *reinterpret_cast<const uint4*>(kc + at);
        const __half2* k2 = reinterpret_cast<const __half2*>(&kr);
        float d = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __half22float2(k2[j]);
            d = fmaf(qv[2 * j], f.x, fmaf(qv[2 * j + 1], f.y, d));
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
        const float mn = fmaxf(m, d), c = __expf(m - mn), e = __expf(d - mn);
        const uint4 vr = *reinterpret_cast<const uint4*>(vc + at);
        const __half2* v2 = reinterpret_cast<const __half2*>(&vr);
        l = l * c + e;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __half22float2(v2[j]);
            acc[2 * j] = fmaf(acc[2 * j], c, e * f.x);
            acc[2 * j + 1] = fmaf(acc[2 * j + 1], c, e * f.y);
        }
        m = mn;
    }
    __shared__ float wm[4], wl[4], wa[4][HD];
    if (lane == 0) wm[warp] = m, wl[warp] = l;
#pragma unroll
    for (int j = 0; j < 8; ++j) wa[warp][lane * 8 + j] = acc[j];
    __syncthreads();
    float M = -INFINITY;
    for (int w = 0; w < 4; ++w) M = fmaxf(M, wm[w]);
    float* out = part + (((size_t) t * S27_NQ + h) * S27_ASPLIT + s) * (HD + 2);
    for (int d = threadIdx.x; d < HD; d += blockDim.x) {
        float a = 0.f;
        if (M != -INFINITY)
            for (int w = 0; w < 4; ++w) a += wa[w][d] * __expf(wm[w] - M);
        out[d] = a;
    }
    if (threadIdx.x == 0) {
        float L = 0.f;
        if (M != -INFINITY)
            for (int w = 0; w < 4; ++w) L += wl[w] * __expf(wm[w] - M);
        out[HD] = M;
        out[HD + 1] = L;
    }
}

// per (row, head): the splits merged, gated by sigmoid(gate), into att [T][NQ][HD]
__global__ void __launch_bounds__(HD) k_s27_merge(const float* qkv, const float* part, float* att) {
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const float* ps = part + ((size_t) t * S27_NQ + h) * S27_ASPLIT * (HD + 2);
    float M = -INFINITY;
    for (int s = 0; s < S27_ASPLIT; ++s) M = fmaxf(M, ps[(size_t) s * (HD + 2) + HD]);
    float a = 0.f, L = 0.f;
    for (int s = 0; s < S27_ASPLIT; ++s) {
        const float* p = ps + (size_t) s * (HD + 2);
        if (p[HD] == -INFINITY) continue;
        const float e = __expf(p[HD] - M);
        a = fmaf(p[d], e, a);
        L = fmaf(p[HD + 1], e, L);
    }
    const float g = qkv[(size_t) t * S27_QKV + h * 2 * HD + HD + d];
    att[((size_t) t * S27_NQ + h) * HD + d] = a / L / (1.f + __expf(-g));
}

}  // namespace fnx
