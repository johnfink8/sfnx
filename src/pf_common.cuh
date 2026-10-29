// What both models' prompt paths (prefill.cu, s27_prefill.cu) share: their error checks and the copy-engine ring's
// conversions (pf_ring_plan.hpp's steps: a card's addend packed to fp16, added to what came in, the gathered sum read
// back), each setting the window's error bit 2 past fp16's range.
#pragma once

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "kernels.hpp"

namespace fnx {

inline void ckc(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("prefill: ") + what + ": " + cudaGetErrorString(e));
}
inline void ckb(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string("prefill: cuBLAS ") + what + ": status " + std::to_string((int) s));
}
inline void cud(CUresult e, const char* what) {
    if (e != CUDA_SUCCESS) {
        const char* s = nullptr;
        cuGetErrorString(e, &s);
        throw std::runtime_error(std::string("prefill: ") + what + ": " + (s ? s : "?"));
    }
}

// out = f16(x) over n values (n even)
static __global__ void k_pfr_pack(const float2* __restrict__ x, int n2, __half2* __restrict__ out, unsigned* err) {
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < n2; e += gridDim.x * blockDim.x) {
        const float2 v = x[e];
        if (!(fabsf(v.x) <= RING_MAX && fabsf(v.y) <= RING_MAX)) atomicOr(err, 2u);   // past fp16's range (or NaN)
        out[e] = __floats2half2_rn(v.x, v.y);
    }
}
// out = f16(in + x) over n values (n even)
static __global__ void k_pfr_add(const __half2* __restrict__ in, const float2* __restrict__ x, int n2, __half2* __restrict__ out,
                                 unsigned* err) {
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < n2; e += gridDim.x * blockDim.x) {
        const float2 a = __half22float2(in[e]), b = x[e];
        const float2 v = make_float2(a.x + b.x, a.y + b.y);
        if (!(fabsf(v.x) <= RING_MAX && fabsf(v.y) <= RING_MAX)) atomicOr(err, 2u);
        out[e] = __floats2half2_rn(v.x, v.y);
    }
}
static __global__ void k_pfr_out(const __half2* __restrict__ ag, float2* __restrict__ x, int n2) {
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < n2; e += gridDim.x * blockDim.x) x[e] = __half22float2(ag[e]);
}

}  // namespace fnx
