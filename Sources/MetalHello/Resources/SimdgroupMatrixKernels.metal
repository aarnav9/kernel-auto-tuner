// These kernels target Apple GPU families that expose SIMD-group matrix multiply
// operations. They use four 32-lane SIMD groups to build one 16x16 output tile
// from 8x8 matrix primitives.

#define DEFINE_SIMDGROUP_MATMUL(NAME) \
kernel void NAME(device const float *a [[buffer(0)]], \
                 device const float *b [[buffer(1)]], \
                 device float *c [[buffer(2)]], \
                 constant MatMulParams &params [[buffer(3)]], \
                 uint2 tgp [[threadgroup_position_in_grid]], \
                 uint tid [[thread_index_in_threadgroup]], \
                 uint simdGroup [[simdgroup_index_in_threadgroup]]) { \
    constexpr uint TILE_M = 16; \
    constexpr uint TILE_N = 16; \
    constexpr uint TILE_K = 8; \
    threadgroup float aTile[TILE_M * TILE_K]; \
    threadgroup float bTile[TILE_K * TILE_N]; \
    threadgroup float cTile[TILE_M * TILE_N]; \
    const uint tileRowBase = tgp.y * TILE_M; \
    const uint tileColBase = tgp.x * TILE_N; \
    const uint loadARow = tid / TILE_K; \
    const uint loadACol = tid % TILE_K; \
    const uint loadBRow = tid / TILE_N; \
    const uint loadBCol = tid % TILE_N; \
    const uint simdRow = simdGroup / 2; \
    const uint simdCol = simdGroup % 2; \
    simdgroup_float8x8 matA; \
    simdgroup_float8x8 matB; \
    simdgroup_float8x8 matC(0.0f); \
    for (uint tileStart = 0; tileStart < params.K; tileStart += TILE_K) { \
        const uint aRow = tileRowBase + loadARow; \
        const uint aCol = tileStart + loadACol; \
        aTile[loadARow * TILE_K + loadACol] = \
            (aRow < params.M && aCol < params.K) ? a[aRow * params.K + aCol] : 0.0f; \
        const uint bRow = tileStart + loadBRow; \
        const uint bCol = tileColBase + loadBCol; \
        bTile[loadBRow * TILE_N + loadBCol] = \
            (bRow < params.K && bCol < params.N) ? b[bRow * params.N + bCol] : 0.0f; \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        simdgroup_load(matA, aTile + (simdRow * 8) * TILE_K, TILE_K); \
        simdgroup_load(matB, bTile + simdCol * 8, TILE_N); \
        simdgroup_multiply_accumulate(matC, matA, matB, matC); \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    simdgroup_store(matC, cTile + (simdRow * 8) * TILE_N + simdCol * 8, TILE_N); \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    const uint firstIndex = tid; \
    const uint secondIndex = tid + 128; \
    if (firstIndex < TILE_M * TILE_N) { \
        const uint localRow = firstIndex / TILE_N; \
        const uint localCol = firstIndex % TILE_N; \
        const uint row = tileRowBase + localRow; \
        const uint col = tileColBase + localCol; \
        if (row < params.M && col < params.N) { \
            c[row * params.N + col] = cTile[firstIndex]; \
        } \
    } \
    if (secondIndex < TILE_M * TILE_N) { \
        const uint localRow = secondIndex / TILE_N; \
        const uint localCol = secondIndex % TILE_N; \
        const uint row = tileRowBase + localRow; \
        const uint col = tileColBase + localCol; \
        if (row < params.M && col < params.N) { \
            c[row * params.N + col] = cTile[secondIndex]; \
        } \
    } \
}

#define DEFINE_FUSED_SIMDGROUP_MATMUL(NAME) \
kernel void NAME(device const float *a [[buffer(0)]], \
                 device const float *b [[buffer(1)]], \
                 device float *c [[buffer(2)]], \
                 constant MatMulParams &params [[buffer(3)]], \
                 device const float *bias [[buffer(4)]], \
                 uint2 tgp [[threadgroup_position_in_grid]], \
                 uint tid [[thread_index_in_threadgroup]], \
                 uint simdGroup [[simdgroup_index_in_threadgroup]]) { \
    constexpr uint TILE_M = 16; \
    constexpr uint TILE_N = 16; \
    constexpr uint TILE_K = 8; \
    threadgroup float aTile[TILE_M * TILE_K]; \
    threadgroup float bTile[TILE_K * TILE_N]; \
    threadgroup float cTile[TILE_M * TILE_N]; \
    const uint tileRowBase = tgp.y * TILE_M; \
    const uint tileColBase = tgp.x * TILE_N; \
    const uint loadARow = tid / TILE_K; \
    const uint loadACol = tid % TILE_K; \
    const uint loadBRow = tid / TILE_N; \
    const uint loadBCol = tid % TILE_N; \
    const uint simdRow = simdGroup / 2; \
    const uint simdCol = simdGroup % 2; \
    simdgroup_float8x8 matA; \
    simdgroup_float8x8 matB; \
    simdgroup_float8x8 matC(0.0f); \
    for (uint tileStart = 0; tileStart < params.K; tileStart += TILE_K) { \
        const uint aRow = tileRowBase + loadARow; \
        const uint aCol = tileStart + loadACol; \
        aTile[loadARow * TILE_K + loadACol] = \
            (aRow < params.M && aCol < params.K) ? a[aRow * params.K + aCol] : 0.0f; \
        const uint bRow = tileStart + loadBRow; \
        const uint bCol = tileColBase + loadBCol; \
        bTile[loadBRow * TILE_N + loadBCol] = \
            (bRow < params.K && bCol < params.N) ? b[bRow * params.N + bCol] : 0.0f; \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        simdgroup_load(matA, aTile + (simdRow * 8) * TILE_K, TILE_K); \
        simdgroup_load(matB, bTile + simdCol * 8, TILE_N); \
        simdgroup_multiply_accumulate(matC, matA, matB, matC); \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    simdgroup_store(matC, cTile + (simdRow * 8) * TILE_N + simdCol * 8, TILE_N); \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    const uint firstIndex = tid; \
    const uint secondIndex = tid + 128; \
    if (firstIndex < TILE_M * TILE_N) { \
        const uint localRow = firstIndex / TILE_N; \
        const uint localCol = firstIndex % TILE_N; \
        const uint row = tileRowBase + localRow; \
        const uint col = tileColBase + localCol; \
        if (row < params.M && col < params.N) { \
            const float biased = cTile[firstIndex] + bias[col]; \
            c[row * params.N + col] = max(biased, 0.0f); \
        } \
    } \
    if (secondIndex < TILE_M * TILE_N) { \
        const uint localRow = secondIndex / TILE_N; \
        const uint localCol = secondIndex % TILE_N; \
        const uint row = tileRowBase + localRow; \
        const uint col = tileColBase + localCol; \
        if (row < params.M && col < params.N) { \
            const float biased = cTile[secondIndex] + bias[col]; \
            c[row * params.N + col] = max(biased, 0.0f); \
        } \
    } \
}

DEFINE_SIMDGROUP_MATMUL(matmul_simdgroup_m16_n16_k8_tx32_ty4)
DEFINE_SIMDGROUP_MATMUL(matmul_simdgroup_m16_n16_k8_tx16_ty8)
DEFINE_SIMDGROUP_MATMUL(matmul_simdgroup_m16_n16_k8_tx8_ty16)

DEFINE_FUSED_SIMDGROUP_MATMUL(matmul_bias_relu_simdgroup_m16_n16_k8_tx32_ty4)
DEFINE_FUSED_SIMDGROUP_MATMUL(matmul_bias_relu_simdgroup_m16_n16_k8_tx16_ty8)
DEFINE_FUSED_SIMDGROUP_MATMUL(matmul_bias_relu_simdgroup_m16_n16_k8_tx8_ty16)
