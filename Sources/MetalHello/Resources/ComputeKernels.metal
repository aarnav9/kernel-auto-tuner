#include <metal_stdlib>
using namespace metal;

struct MatMulParams {
    uint M;
    uint N;
    uint K;
};

struct EpilogueParams {
    uint M;
    uint N;
};

kernel void add_arrays(device const float *inA [[buffer(0)]],
                       device const float *inB [[buffer(1)]],
                       device float *result [[buffer(2)]],
                       constant uint &length [[buffer(3)]],
                       uint index [[thread_position_in_grid]]) {
    // Dispatch may round up, so guard the tail.
    if (index >= length) {
        return;
    }

    result[index] = inA[index] + inB[index];
}

kernel void matmul_naive(device const float *a [[buffer(0)]],
                         device const float *b [[buffer(1)]],
                         device float *c [[buffer(2)]],
                         constant MatMulParams &params [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
    // gid.x maps to output column, gid.y maps to output row.
    const uint col = gid.x;
    const uint row = gid.y;

    // Non-multiple shapes dispatch extra threads; skip anything outside C.
    if (row >= params.M || col >= params.N) {
        return;
    }

    float acc = 0.0f;
    // One thread computes one output cell by walking the entire reduction axis.
    for (uint k = 0; k < params.K; ++k) {
        acc += a[row * params.K + k] * b[k * params.N + col];
    }

    c[row * params.N + col] = acc;
}

kernel void add_bias_relu(device float *matrix [[buffer(0)]],
                          device const float *bias [[buffer(1)]],
                          constant EpilogueParams &params [[buffer(2)]],
                          uint2 gid [[thread_position_in_grid]]) {
    const uint col = gid.x;
    const uint row = gid.y;

    if (row >= params.M || col >= params.N) {
        return;
    }

    const uint index = row * params.N + col;
    // Bias is per output column, which matches common GEMM epilogues in ML workloads.
    const float value = matrix[index] + bias[col];
    matrix[index] = max(value, 0.0f);
}

#define DEFINE_TILED_MATMUL(NAME, TILE_M, TILE_N, TILE_K) \
kernel void NAME(device const float *a [[buffer(0)]], \
                 device const float *b [[buffer(1)]], \
                 device float *c [[buffer(2)]], \
                 constant MatMulParams &params [[buffer(3)]], \
                 uint2 gid [[thread_position_in_grid]], \
                 uint2 tid [[thread_position_in_threadgroup]]) { \
    const uint col = gid.x; \
    const uint row = gid.y; \
    const uint localCol = tid.x; \
    const uint localRow = tid.y; \
    threadgroup float aTile[TILE_M][TILE_K]; \
    threadgroup float bTile[TILE_K][TILE_N]; \
    float acc = 0.0f; \
    for (uint tileStart = 0; tileStart < params.K; tileStart += TILE_K) { \
        const uint aCol = tileStart + localCol; \
        const uint bRow = tileStart + localRow; \
        if (localCol < TILE_K) { \
            aTile[localRow][localCol] = \
                (row < params.M && aCol < params.K) ? a[row * params.K + aCol] : 0.0f; \
        } \
        if (localRow < TILE_K) { \
            bTile[localRow][localCol] = \
                (bRow < params.K && col < params.N) ? b[bRow * params.N + col] : 0.0f; \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        for (uint k = 0; k < TILE_K; ++k) { \
            acc += aTile[localRow][k] * bTile[k][localCol]; \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    if (row < params.M && col < params.N) { \
        c[row * params.N + col] = acc; \
    } \
}

#define DEFINE_FUSED_TILED_MATMUL(NAME, TILE_M, TILE_N, TILE_K) \
kernel void NAME(device const float *a [[buffer(0)]], \
                 device const float *b [[buffer(1)]], \
                 device float *c [[buffer(2)]], \
                 constant MatMulParams &params [[buffer(3)]], \
                 device const float *bias [[buffer(4)]], \
                 uint2 gid [[thread_position_in_grid]], \
                 uint2 tid [[thread_position_in_threadgroup]]) { \
    const uint col = gid.x; \
    const uint row = gid.y; \
    const uint localCol = tid.x; \
    const uint localRow = tid.y; \
    threadgroup float aTile[TILE_M][TILE_K]; \
    threadgroup float bTile[TILE_K][TILE_N]; \
    float acc = 0.0f; \
    for (uint tileStart = 0; tileStart < params.K; tileStart += TILE_K) { \
        const uint aCol = tileStart + localCol; \
        const uint bRow = tileStart + localRow; \
        if (localCol < TILE_K) { \
            aTile[localRow][localCol] = \
                (row < params.M && aCol < params.K) ? a[row * params.K + aCol] : 0.0f; \
        } \
        if (localRow < TILE_K) { \
            bTile[localRow][localCol] = \
                (bRow < params.K && col < params.N) ? b[bRow * params.N + col] : 0.0f; \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        for (uint k = 0; k < TILE_K; ++k) { \
            acc += aTile[localRow][k] * bTile[k][localCol]; \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    if (row < params.M && col < params.N) { \
        /* Fuse the epilogue so we avoid a second pass over C. */ \
        const float biased = acc + bias[col]; \
        c[row * params.N + col] = max(biased, 0.0f); \
    } \
}

// These are fixed kernel configs so we can exhaustively sweep a small space
// before introducing a real search loop.
DEFINE_TILED_MATMUL(matmul_tiled_m8_n8_k8_tx8_ty8, 8, 8, 8)
DEFINE_TILED_MATMUL(matmul_tiled_m8_n16_k8_tx16_ty8, 8, 16, 8)
DEFINE_TILED_MATMUL(matmul_tiled_m16_n8_k8_tx8_ty16, 16, 8, 8)
DEFINE_TILED_MATMUL(matmul_tiled_m16_n16_k8_tx16_ty16, 16, 16, 8)
DEFINE_TILED_MATMUL(matmul_tiled_m16_n16_k16_tx16_ty16, 16, 16, 16)
DEFINE_TILED_MATMUL(matmul_tiled_m8_n32_k8_tx32_ty8, 8, 32, 8)
DEFINE_TILED_MATMUL(matmul_tiled_m32_n8_k8_tx8_ty32, 32, 8, 8)
DEFINE_TILED_MATMUL(matmul_tiled_m16_n32_k8_tx32_ty16, 16, 32, 8)
DEFINE_TILED_MATMUL(matmul_tiled_m32_n16_k8_tx16_ty32, 32, 16, 8)
DEFINE_TILED_MATMUL(matmul_tiled_m32_n32_k8_tx32_ty32, 32, 32, 8)
DEFINE_TILED_MATMUL(matmul_tiled_m8_n64_k8_tx64_ty8, 8, 64, 8)
DEFINE_TILED_MATMUL(matmul_tiled_m64_n8_k8_tx8_ty64, 64, 8, 8)
DEFINE_TILED_MATMUL(matmul_tiled_m16_n64_k8_tx64_ty16, 16, 64, 8)
DEFINE_TILED_MATMUL(matmul_tiled_m64_n16_k8_tx16_ty64, 64, 16, 8)
DEFINE_TILED_MATMUL(matmul_tiled_m16_n16_k4_tx16_ty16, 16, 16, 4)
DEFINE_TILED_MATMUL(matmul_tiled_m32_n8_k4_tx8_ty32, 32, 8, 4)

DEFINE_FUSED_TILED_MATMUL(matmul_bias_relu_tiled_m8_n8_k8_tx8_ty8, 8, 8, 8)
DEFINE_FUSED_TILED_MATMUL(matmul_bias_relu_tiled_m8_n16_k8_tx16_ty8, 8, 16, 8)
DEFINE_FUSED_TILED_MATMUL(matmul_bias_relu_tiled_m16_n8_k8_tx8_ty16, 16, 8, 8)
DEFINE_FUSED_TILED_MATMUL(matmul_bias_relu_tiled_m16_n16_k8_tx16_ty16, 16, 16, 8)
DEFINE_FUSED_TILED_MATMUL(matmul_bias_relu_tiled_m16_n16_k16_tx16_ty16, 16, 16, 16)
DEFINE_FUSED_TILED_MATMUL(matmul_bias_relu_tiled_m8_n32_k8_tx32_ty8, 8, 32, 8)
DEFINE_FUSED_TILED_MATMUL(matmul_bias_relu_tiled_m32_n8_k8_tx8_ty32, 32, 8, 8)
DEFINE_FUSED_TILED_MATMUL(matmul_bias_relu_tiled_m16_n32_k8_tx32_ty16, 16, 32, 8)
DEFINE_FUSED_TILED_MATMUL(matmul_bias_relu_tiled_m32_n16_k8_tx16_ty32, 32, 16, 8)
DEFINE_FUSED_TILED_MATMUL(matmul_bias_relu_tiled_m32_n32_k8_tx32_ty32, 32, 32, 8)
DEFINE_FUSED_TILED_MATMUL(matmul_bias_relu_tiled_m8_n64_k8_tx64_ty8, 8, 64, 8)
DEFINE_FUSED_TILED_MATMUL(matmul_bias_relu_tiled_m64_n8_k8_tx8_ty64, 64, 8, 8)
DEFINE_FUSED_TILED_MATMUL(matmul_bias_relu_tiled_m16_n64_k8_tx64_ty16, 16, 64, 8)
DEFINE_FUSED_TILED_MATMUL(matmul_bias_relu_tiled_m64_n16_k8_tx16_ty64, 64, 16, 8)
DEFINE_FUSED_TILED_MATMUL(matmul_bias_relu_tiled_m16_n16_k4_tx16_ty16, 16, 16, 4)
DEFINE_FUSED_TILED_MATMUL(matmul_bias_relu_tiled_m32_n8_k4_tx8_ty32, 32, 8, 4)
