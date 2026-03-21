/**
 * FP8 (e4m3fn) Dequantization + Matrix Multiplication on Metal
 *
 * Metal has no native FP8 type, so we store FP8 as uint8_t and decode
 * in-register using IEEE-754 bit extraction.
 *
 * e4m3fn format: [sign:1][exponent:4][mantissa:3], bias=7, no inf, NaN=0x7F/0xFF
 *
 * Patterns from:
 *   metalQwen3/quantized_matmul_optimized.metal  — SIMD reduction, 4-element unroll
 *   metalQwen3/quantize.metal                    — group-wise quant/dequant
 */

#include <metal_stdlib>
using namespace metal;

// ─── FP8 e4m3fn → float32 decode ───────────────────────────────────────────

inline float fp8_e4m3fn_to_float(uint8_t bits) {
    // NaN values: 0x7F (0_1111_111) and 0xFF (1_1111_111)
    if ((bits & 0x7F) == 0x7F) return NAN;

    uint sign = (bits >> 7) & 1;
    uint exp_bits = (bits >> 3) & 0xF;  // 4-bit exponent
    uint mant_bits = bits & 0x7;         // 3-bit mantissa

    float value;
    if (exp_bits == 0) {
        // Subnormal: value = (-1)^sign * 2^(1-bias) * (0.mantissa)
        // bias=7, so 2^(-6) = 1/64
        value = float(mant_bits) / 8.0f * (1.0f / 64.0f);
    } else {
        // Normal: value = (-1)^sign * 2^(exp-bias) * (1.mantissa)
        float mantissa = 1.0f + float(mant_bits) / 8.0f;
        int exponent = int(exp_bits) - 7;  // bias = 7
        value = mantissa * exp2(float(exponent));
    }

    return sign ? -value : value;
}

// ─── float32 → FP8 e4m3fn encode ───────────────────────────────────────────

inline uint8_t float_to_fp8_e4m3fn(float val) {
    uint sign = 0;
    if (val < 0.0f) {
        sign = 1;
        val = -val;
    }

    // Max representable: 448.0 (1111_110 = exp=14, bias=7 → 2^7 * 1.75 = 224... actually 2^8*(1+6/8)=448)
    // Clamp to max
    if (val >= 448.0f) {
        return (sign << 7) | 0x7E;  // 0_1111_110 = max normal
    }

    // Min subnormal: 2^(-9) = 1/512
    if (val < (1.0f / 512.0f)) {
        return sign << 7;  // Preserve sign for zero
    }

    // Try subnormal first: val = mant/8 * 2^(-6)
    // mant = val * 8 * 64 = val * 512
    if (val < (1.0f / 64.0f)) {
        // Round-half-to-even (banker's rounding) to match PyTorch CPU
        float mant_f = val * 512.0f;
        uint mant = uint(rint(mant_f));
        mant = min(mant, 7u);
        return (sign << 7) | uint8_t(mant);
    }

    // Normal: find exponent
    int exp_val = int(floor(log2(val)));
    // Clamp exponent to [0-7, 14-7] = [-7, 7]
    exp_val = clamp(exp_val, -6, 8);

    float mantissa = val / exp2(float(exp_val));  // 1.xxx
    // Round-half-to-even (banker's rounding) to match PyTorch CPU
    float mant_f = (mantissa - 1.0f) * 8.0f;
    uint mant = uint(rint(mant_f));
    mant = min(mant, 7u);

    uint exp_bits = uint(exp_val + 7);  // add bias
    exp_bits = clamp(exp_bits, 1u, 15u);

    // Avoid NaN encoding (exp=15, mant=7)
    if (exp_bits == 15 && mant == 7) {
        mant = 6;  // clamp to max normal
    }

    return (sign << 7) | uint8_t(exp_bits << 3) | uint8_t(mant);
}

// ─── General MxN Scaled MatMul ──────────────────────────────────────────────
// A: (M,K) uint8 FP8, B: (N,K) uint8 FP8 (transposed), out: (M,N) float32
// scale_mode: 0=per-tensor, 1=per-channel(row)
// Threadgroup: 16x16

kernel void fp8_scaled_matmul_kernel(
    device const uint8_t* A [[buffer(0)]],       // (M, K) row-major FP8
    device const uint8_t* B [[buffer(1)]],       // (N, K) row-major FP8 (B is transposed: output row j uses B[j,:])
    device float* C [[buffer(2)]],               // (M, N) output
    device const float* scale_a [[buffer(3)]],   // per-tensor [1] or per-row [M]
    device const float* scale_b [[buffer(4)]],   // per-tensor [1] or per-row [N]
    constant uint& M [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    constant uint& scale_mode [[buffer(8)]],     // 0=per-tensor, 1=per-channel
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;  // M dimension
    uint col = gid.x;  // N dimension

    if (row >= M || col >= N) return;

    float sum = 0.0f;

    // 4-element unrolling on K for better memory bandwidth
    uint k = 0;
    uint K4 = (K / 4) * 4;
    for (; k < K4; k += 4) {
        uint a_idx = row * K + k;
        uint b_idx = col * K + k;

        float a0 = fp8_e4m3fn_to_float(A[a_idx]);
        float a1 = fp8_e4m3fn_to_float(A[a_idx + 1]);
        float a2 = fp8_e4m3fn_to_float(A[a_idx + 2]);
        float a3 = fp8_e4m3fn_to_float(A[a_idx + 3]);

        float b0 = fp8_e4m3fn_to_float(B[b_idx]);
        float b1 = fp8_e4m3fn_to_float(B[b_idx + 1]);
        float b2 = fp8_e4m3fn_to_float(B[b_idx + 2]);
        float b3 = fp8_e4m3fn_to_float(B[b_idx + 3]);

        sum += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
    }

    // Handle remaining elements
    for (; k < K; k++) {
        sum += fp8_e4m3fn_to_float(A[row * K + k]) * fp8_e4m3fn_to_float(B[col * K + k]);
    }

    // Apply scaling
    float sa = (scale_mode == 0) ? scale_a[0] : scale_a[row];
    float sb = (scale_mode == 0) ? scale_b[0] : scale_b[col];
    C[row * N + col] = sum * sa * sb;
}


// ─── Optimized Vec-Mat for Single Token (M=1) ──────────────────────────────
// x: (K,) uint8 FP8, W: (N,K) uint8 FP8, out: (N,) float32
// Uses SIMD reduction across K dimension
// Threadgroup: 256

kernel void fp8_scaled_vecmat_kernel(
    device const uint8_t* x [[buffer(0)]],       // (K,) input vector FP8
    device const uint8_t* W [[buffer(1)]],       // (N, K) weight matrix FP8 row-major
    device float* output [[buffer(2)]],           // (N,) output
    device const float* scale_x [[buffer(3)]],   // per-tensor scale for x
    device const float* scale_w [[buffer(4)]],   // per-tensor [1] or per-row [N]
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& scale_mode [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = gid / 32;  // Each SIMD group handles one output row
    if (row >= N) return;

    uint row_offset = row * K;

    float sum = 0.0f;

    // Each SIMD lane processes a stride of K
    // 4-element unrolling within each lane
    for (uint k = simd_lane * 4; k < K; k += 32 * 4) {
        if (k + 3 < K) {
            uint x_idx = k;
            uint w_idx = row_offset + k;

            float x0 = fp8_e4m3fn_to_float(x[x_idx]);
            float x1 = fp8_e4m3fn_to_float(x[x_idx + 1]);
            float x2 = fp8_e4m3fn_to_float(x[x_idx + 2]);
            float x3 = fp8_e4m3fn_to_float(x[x_idx + 3]);

            float w0 = fp8_e4m3fn_to_float(W[w_idx]);
            float w1 = fp8_e4m3fn_to_float(W[w_idx + 1]);
            float w2 = fp8_e4m3fn_to_float(W[w_idx + 2]);
            float w3 = fp8_e4m3fn_to_float(W[w_idx + 3]);

            sum += x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3;
        } else {
            // Handle tail
            for (uint kk = k; kk < min(k + 4, K); kk++) {
                sum += fp8_e4m3fn_to_float(x[kk]) * fp8_e4m3fn_to_float(W[row_offset + kk]);
            }
        }
    }

    // SIMD reduction — hardware-accelerated sum across 32 lanes
    sum = simd_sum(sum);

    // First lane writes result
    if (simd_lane == 0) {
        float sx = scale_x[0];
        float sw = (scale_mode == 0) ? scale_w[0] : scale_w[row];
        output[row] = sum * sx * sw;
    }
}


// ─── Standalone FP8 → half dequantize ───────────────────────────────────────

kernel void fp8_to_half_kernel(
    device const uint8_t* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    output[gid] = half(fp8_e4m3fn_to_float(input[gid]));
}


// ─── Standalone float → FP8 quantize ───────────────────────────────────────

kernel void float_to_fp8_kernel(
    device const float* input [[buffer(0)]],
    device uint8_t* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    output[gid] = float_to_fp8_e4m3fn(input[gid]);
}
