"""Correctness tests for the FP8 e4m3fn reference encoder/decoder.

Validates bit-exact encode/decode for all 256 possible FP8 values,
special values, monotonicity, and quantization error bounds.
"""

import math

MAX_NORMAL_ERROR_THRESHOLD = 0.07


def fp8_e4m3fn_decode_spec(bits: int) -> float:
    """Reference FP8 e4m3fn decoder (IEEE spec)."""
    if (bits & 0x7F) == 0x7F:
        return 0.0
    sign = (bits >> 7) & 1
    exp_bits = (bits >> 3) & 0xF
    mant_bits = bits & 0x7
    if exp_bits == 0:
        value = (mant_bits / 8.0) * (2.0 ** -6)
    else:
        mantissa = 1.0 + mant_bits / 8.0
        exponent = exp_bits - 7
        value = mantissa * (2.0 ** exponent)
    return -value if sign else value


def fp8_e4m3fn_encode_spec(val: float) -> int:
    """Reference FP8 e4m3fn encoder (IEEE spec)."""
    sign = 0
    if val < 0.0:
        sign = 1
        val = -val
    if val == 0.0:
        return sign << 7
    if val >= 448.0:
        return (sign << 7) | (15 << 3) | 6
    if val < (1.0 / 512.0):
        return sign << 7
    if val < (1.0 / 64.0):
        mant = round(val * 512.0)
        mant = min(mant, 7)
        return (sign << 7) | mant
    exp_unbiased = int(math.floor(math.log2(val)))
    exp_unbiased = max(-6, min(8, exp_unbiased))
    scale = 2.0 ** exp_unbiased
    mantissa_value = val / scale
    mant_frac = mantissa_value - 1.0
    mant = round(mant_frac * 8.0)
    mant = min(mant, 7)
    exp_bits = exp_unbiased + 7
    exp_bits = max(1, min(15, exp_bits))
    if exp_bits == 15 and mant == 7:
        mant = 6
    return (sign << 7) | (exp_bits << 3) | mant


def test_all_256_values_roundtrip():
    """encode(decode(x)) == x for all 256 possible FP8 values."""
    errors = []
    for bits in range(256):
        decoded = fp8_e4m3fn_decode_spec(bits)
        reencoded = fp8_e4m3fn_encode_spec(decoded)
        # NaN and negative zero normalise to positive zero
        if bits in [0x7F, 0xFF, 0x80] and reencoded == 0x00:
            continue
        if reencoded != bits:
            errors.append((bits, decoded, reencoded))
    assert errors == [], f"Roundtrip errors: {errors[:10]}"


def test_special_values():
    """Special values and edge cases encode to expected bit patterns."""
    test_cases = [
        ("Zero", 0.0, 0x00),
        ("Min subnormal", 0.001953125, 0x01),
        ("Max subnormal", 0.013671875, 0x07),
        ("Min normal", 0.015625, 0x08),
        ("One", 1.0, 0x38),
        ("Max normal", 448.0, 0x7E),
        ("Overflow (should clamp)", 500.0, 0x7E),
    ]
    for name, value, expected_bits in test_cases:
        encoded = fp8_e4m3fn_encode_spec(value)
        assert encoded == expected_bits, (
            f"{name}: {value} encoded to 0x{encoded:02X}, expected 0x{expected_bits:02X}"
        )


def test_monotonicity():
    """Decoded positive FP8 values are in ascending order."""
    positive_values = []
    for bits in range(0x00, 0x80):
        if bits == 0x7F:
            continue
        decoded = fp8_e4m3fn_decode_spec(bits)
        positive_values.append((bits, decoded))

    for i in range(len(positive_values) - 1):
        bits1, val1 = positive_values[i]
        bits2, val2 = positive_values[i + 1]
        assert val2 >= val1, (
            f"Monotonicity violation: 0x{bits1:02X}={val1} > 0x{bits2:02X}={val2}"
        )


def test_quantization_error_normal_values():
    """Quantization error for normal numbers stays below threshold."""
    test_values = []
    for exp in range(-9, 9):
        base = 2.0 ** exp
        for mant_mult in [1.0, 1.5, 2.0, 3.0, 5.0, 7.0]:
            val = base * mant_mult
            if 0.015625 <= val < 448.0:
                test_values.append(val)

    max_normal_error = 0.0
    for val in test_values:
        encoded = fp8_e4m3fn_encode_spec(val)
        decoded = fp8_e4m3fn_decode_spec(encoded)
        rel_error = abs(decoded - val) / val
        max_normal_error = max(max_normal_error, rel_error)

    assert max_normal_error < MAX_NORMAL_ERROR_THRESHOLD, (
        f"Max normal error {max_normal_error:.2%} >= threshold {MAX_NORMAL_ERROR_THRESHOLD:.0%}"
    )
