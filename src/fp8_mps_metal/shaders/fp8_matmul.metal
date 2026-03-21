#include <metal_stdlib>
using namespace metal;

// FP8 e4m3fn decode via LUT — single constant memory load, no branching
constant float fp8_e4m3fn_lut[256] = {
    0.0f, 0.001953125f, 0.00390625f, 0.005859375f, 0.0078125f, 0.009765625f, 0.01171875f, 0.013671875f,
    0.015625f, 0.017578125f, 0.01953125f, 0.021484375f, 0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
    0.03125f, 0.03515625f, 0.0390625f, 0.04296875f, 0.046875f, 0.05078125f, 0.0546875f, 0.05859375f,
    0.0625f, 0.0703125f, 0.078125f, 0.0859375f, 0.09375f, 0.1015625f, 0.109375f, 0.1171875f,
    0.125f, 0.140625f, 0.15625f, 0.171875f, 0.1875f, 0.203125f, 0.21875f, 0.234375f,
    0.25f, 0.28125f, 0.3125f, 0.34375f, 0.375f, 0.40625f, 0.4375f, 0.46875f,
    0.5f, 0.5625f, 0.625f, 0.6875f, 0.75f, 0.8125f, 0.875f, 0.9375f,
    1.0f, 1.125f, 1.25f, 1.375f, 1.5f, 1.625f, 1.75f, 1.875f,
    2.0f, 2.25f, 2.5f, 2.75f, 3.0f, 3.25f, 3.5f, 3.75f,
    4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f,
    8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
    16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f,
    32.0f, 36.0f, 40.0f, 44.0f, 48.0f, 52.0f, 56.0f, 60.0f,
    64.0f, 72.0f, 80.0f, 88.0f, 96.0f, 104.0f, 112.0f, 120.0f,
    128.0f, 144.0f, 160.0f, 176.0f, 192.0f, 208.0f, 224.0f, 240.0f,
    256.0f, 288.0f, 320.0f, 352.0f, 384.0f, 416.0f, 448.0f, NAN,
    0.0f, -0.001953125f, -0.00390625f, -0.005859375f, -0.0078125f, -0.009765625f, -0.01171875f, -0.013671875f,
    -0.015625f, -0.017578125f, -0.01953125f, -0.021484375f, -0.0234375f, -0.025390625f, -0.02734375f, -0.029296875f,
    -0.03125f, -0.03515625f, -0.0390625f, -0.04296875f, -0.046875f, -0.05078125f, -0.0546875f, -0.05859375f,
    -0.0625f, -0.0703125f, -0.078125f, -0.0859375f, -0.09375f, -0.1015625f, -0.109375f, -0.1171875f,
    -0.125f, -0.140625f, -0.15625f, -0.171875f, -0.1875f, -0.203125f, -0.21875f, -0.234375f,
    -0.25f, -0.28125f, -0.3125f, -0.34375f, -0.375f, -0.40625f, -0.4375f, -0.46875f,
    -0.5f, -0.5625f, -0.625f, -0.6875f, -0.75f, -0.8125f, -0.875f, -0.9375f,
    -1.0f, -1.125f, -1.25f, -1.375f, -1.5f, -1.625f, -1.75f, -1.875f,
    -2.0f, -2.25f, -2.5f, -2.75f, -3.0f, -3.25f, -3.5f, -3.75f,
    -4.0f, -4.5f, -5.0f, -5.5f, -6.0f, -6.5f, -7.0f, -7.5f,
    -8.0f, -9.0f, -10.0f, -11.0f, -12.0f, -13.0f, -14.0f, -15.0f,
    -16.0f, -18.0f, -20.0f, -22.0f, -24.0f, -26.0f, -28.0f, -30.0f,
    -32.0f, -36.0f, -40.0f, -44.0f, -48.0f, -52.0f, -56.0f, -60.0f,
    -64.0f, -72.0f, -80.0f, -88.0f, -96.0f, -104.0f, -112.0f, -120.0f,
    -128.0f, -144.0f, -160.0f, -176.0f, -192.0f, -208.0f, -224.0f, -240.0f,
    -256.0f, -288.0f, -320.0f, -352.0f, -384.0f, -416.0f, -448.0f, NAN,
};

// Half-precision LUT for dequantize kernel
constant half fp8_e4m3fn_lut_half[256] = {
    0.0h, 0.001953125h, 0.00390625h, 0.005859375h, 0.0078125h, 0.009765625h, 0.01171875h, 0.013671875h,
    0.015625h, 0.017578125h, 0.01953125h, 0.021484375h, 0.0234375h, 0.025390625h, 0.02734375h, 0.029296875h,
    0.03125h, 0.03515625h, 0.0390625h, 0.04296875h, 0.046875h, 0.05078125h, 0.0546875h, 0.05859375h,
    0.0625h, 0.0703125h, 0.078125h, 0.0859375h, 0.09375h, 0.1015625h, 0.109375h, 0.1171875h,
    0.125h, 0.140625h, 0.15625h, 0.171875h, 0.1875h, 0.203125h, 0.21875h, 0.234375h,
    0.25h, 0.28125h, 0.3125h, 0.34375h, 0.375h, 0.40625h, 0.4375h, 0.46875h,
    0.5h, 0.5625h, 0.625h, 0.6875h, 0.75h, 0.8125h, 0.875h, 0.9375h,
    1.0h, 1.125h, 1.25h, 1.375h, 1.5h, 1.625h, 1.75h, 1.875h,
    2.0h, 2.25h, 2.5h, 2.75h, 3.0h, 3.25h, 3.5h, 3.75h,
    4.0h, 4.5h, 5.0h, 5.5h, 6.0h, 6.5h, 7.0h, 7.5h,
    8.0h, 9.0h, 10.0h, 11.0h, 12.0h, 13.0h, 14.0h, 15.0h,
    16.0h, 18.0h, 20.0h, 22.0h, 24.0h, 26.0h, 28.0h, 30.0h,
    32.0h, 36.0h, 40.0h, 44.0h, 48.0h, 52.0h, 56.0h, 60.0h,
    64.0h, 72.0h, 80.0h, 88.0h, 96.0h, 104.0h, 112.0h, 120.0h,
    128.0h, 144.0h, 160.0h, 176.0h, 192.0h, 208.0h, 224.0h, 240.0h,
    256.0h, 288.0h, 320.0h, 352.0h, 384.0h, 416.0h, 448.0h, NAN,
    0.0h, -0.001953125h, -0.00390625h, -0.005859375h, -0.0078125h, -0.009765625h, -0.01171875h, -0.013671875h,
    -0.015625h, -0.017578125h, -0.01953125h, -0.021484375h, -0.0234375h, -0.025390625h, -0.02734375h, -0.029296875h,
    -0.03125h, -0.03515625h, -0.0390625h, -0.04296875h, -0.046875h, -0.05078125h, -0.0546875h, -0.05859375h,
    -0.0625h, -0.0703125h, -0.078125h, -0.0859375h, -0.09375h, -0.1015625h, -0.109375h, -0.1171875h,
    -0.125h, -0.140625h, -0.15625h, -0.171875h, -0.1875h, -0.203125h, -0.21875h, -0.234375h,
    -0.25h, -0.28125h, -0.3125h, -0.34375h, -0.375h, -0.40625h, -0.4375h, -0.46875h,
    -0.5h, -0.5625h, -0.625h, -0.6875h, -0.75h, -0.8125h, -0.875h, -0.9375h,
    -1.0h, -1.125h, -1.25h, -1.375h, -1.5h, -1.625h, -1.75h, -1.875h,
    -2.0h, -2.25h, -2.5h, -2.75h, -3.0h, -3.25h, -3.5h, -3.75h,
    -4.0h, -4.5h, -5.0h, -5.5h, -6.0h, -6.5h, -7.0h, -7.5h,
    -8.0h, -9.0h, -10.0h, -11.0h, -12.0h, -13.0h, -14.0h, -15.0h,
    -16.0h, -18.0h, -20.0h, -22.0h, -24.0h, -26.0h, -28.0h, -30.0h,
    -32.0h, -36.0h, -40.0h, -44.0h, -48.0h, -52.0h, -56.0h, -60.0h,
    -64.0h, -72.0h, -80.0h, -88.0h, -96.0h, -104.0h, -112.0h, -120.0h,
    -128.0h, -144.0h, -160.0h, -176.0h, -192.0h, -208.0h, -224.0h, -240.0h,
    -256.0h, -288.0h, -320.0h, -352.0h, -384.0h, -416.0h, -448.0h, NAN,
};

// Float → FP8 encode using integer bit manipulation (no transcendentals)
inline uint8_t float_to_fp8_e4m3fn(float val) {
    uint sign = 0;
    if (val < 0.0f) { sign = 1; val = -val; }

    if (val >= 448.0f) return (sign << 7) | 0x7E;
    if (val < (1.0f / 512.0f)) return sign << 7;

    // Extract exponent and mantissa from IEEE 754 float32 bits
    uint bits = as_type<uint>(val);
    int f32_exp = int((bits >> 23) & 0xFF) - 127;  // unbiased float32 exponent
    uint f32_mant = bits & 0x7FFFFF;                // 23-bit mantissa

    // Subnormal FP8: exponent would be <= -7 (biased 0)
    if (f32_exp < -6) {
        float mant_f = val * 512.0f;
        uint mant = uint(rint(mant_f));
        return (sign << 7) | uint8_t(min(mant, 7u));
    }

    // Normal FP8: shift float32 mantissa down to 3 bits
    // float32 has 23 mantissa bits, we need 3 → shift right by 20
    // Round-half-to-even (banker's rounding) to match PyTorch CPU
    uint truncated = f32_mant & 0xFFFFF;  // bottom 20 bits being discarded
    uint halfway = 1u << 19;
    uint mant = f32_mant >> 20;
    if (truncated > halfway || (truncated == halfway && (mant & 1))) {
        mant++;
    }
    int fp8_exp = f32_exp + 7;                    // FP8 bias

    if (mant > 7) { mant = 0; fp8_exp++; }       // mantissa overflow → bump exponent
    fp8_exp = clamp(fp8_exp, 1, 15);
    if (fp8_exp == 15 && mant == 7) mant = 6;    // avoid NaN encoding

    return (sign << 7) | uint8_t(fp8_exp << 3) | uint8_t(mant);
}

// Tiled matmul with threadgroup shared memory
// A: (M,K) uint8 FP8, B: (N,K) uint8 FP8 (transposed), out: (M,N) float32
constant uint TILE_SIZE = 16;

kernel void fp8_scaled_matmul_kernel(
    device const uint8_t* A [[buffer(0)]],
    device const uint8_t* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    device const float* scale_a [[buffer(3)]],
    device const float* scale_b [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    constant uint& scale_mode [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;

    threadgroup float tile_a[TILE_SIZE][TILE_SIZE];
    threadgroup float tile_b[TILE_SIZE][TILE_SIZE];

    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < num_tiles; t++) {
        uint k_base = t * TILE_SIZE;

        // Load A tile — vectorized 4-byte load where possible
        uint a_k = k_base + lid.x;
        if (row < M && a_k < K) {
            tile_a[lid.y][lid.x] = fp8_e4m3fn_lut[A[row * K + a_k]];
        } else {
            tile_a[lid.y][lid.x] = 0.0f;
        }

        // Load B tile
        uint b_k = k_base + lid.y;
        if (col < N && b_k < K) {
            tile_b[lid.y][lid.x] = fp8_e4m3fn_lut[B[col * K + b_k]];
        } else {
            tile_b[lid.y][lid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate tile product
        for (uint i = 0; i < TILE_SIZE; i++) {
            sum += tile_a[lid.y][i] * tile_b[i][lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float sa = (scale_mode == 0) ? scale_a[0] : scale_a[row];
    float sb = (scale_mode == 0) ? scale_b[0] : scale_b[col];
    C[row * N + col] = sum * sa * sb;
}

// Vecmat (M=1) with SIMD reduction and vectorized loads
kernel void fp8_scaled_vecmat_kernel(
    device const uint8_t* x [[buffer(0)]],
    device const uint8_t* W [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const float* scale_x [[buffer(3)]],
    device const float* scale_w [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& scale_mode [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint row = gid / 32;
    if (row >= N) return;

    uint row_offset = row * K;
    float sum = 0.0f;

    // Vectorized: load 4 bytes at once
    device const uint32_t* x4 = reinterpret_cast<device const uint32_t*>(x);
    device const uint32_t* w4 = reinterpret_cast<device const uint32_t*>(W + row_offset);
    uint K4 = K / 4;
    for (uint i = simd_lane; i < K4; i += 32) {
        uint px = x4[i];
        uint pw = w4[i];

        sum += fp8_e4m3fn_lut[px & 0xFF]         * fp8_e4m3fn_lut[pw & 0xFF]
             + fp8_e4m3fn_lut[(px >> 8) & 0xFF]  * fp8_e4m3fn_lut[(pw >> 8) & 0xFF]
             + fp8_e4m3fn_lut[(px >> 16) & 0xFF] * fp8_e4m3fn_lut[(pw >> 16) & 0xFF]
             + fp8_e4m3fn_lut[(px >> 24) & 0xFF] * fp8_e4m3fn_lut[(pw >> 24) & 0xFF];
    }

    for (uint k = K4 * 4 + simd_lane; k < K; k += 32) {
        sum += fp8_e4m3fn_lut[x[k]] * fp8_e4m3fn_lut[W[row_offset + k]];
    }

    sum = simd_sum(sum);

    if (simd_lane == 0) {
        float sx = scale_x[0];
        float sw = (scale_mode == 0) ? scale_w[0] : scale_w[row];
        output[row] = sum * sx * sw;
    }
}

// FP8 → half dequantize (direct half LUT, no float intermediate)
kernel void fp8_to_half_kernel(
    device const uint8_t* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    output[gid] = fp8_e4m3fn_lut_half[input[gid]];
}

// float → FP8 encode (integer bit manipulation, no transcendentals)
kernel void float_to_fp8_kernel(
    device const float* input [[buffer(0)]],
    device uint8_t* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    output[gid] = float_to_fp8_e4m3fn(input[gid]);
}
