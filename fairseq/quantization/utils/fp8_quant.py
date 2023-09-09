#!/usr/bin/env python
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.
import torch
import torch.nn as nn
import numpy as np
from itertools import product
from torch.autograd import Function
from .quant_utils import *

def decode_binary_str(F_str):
    F = sum([2 ** -(i + 1) * int(a) for i, a in enumerate(F_str)]) * 2 ** len(F_str)
    return F

def generate_all_values_fp(
    num_total_bits: int = 8, num_exponent_bits: int = 4, bias: int = 8, use_ieee_standard: bool = True
) -> list:
    num_fraction_bits = num_total_bits - 1 - num_exponent_bits

    all_values = []
    exp_lower = -bias
    for S in [-1.0, 1.0]:
        for E_str_iter in product(*[[0, 1]] * num_exponent_bits): # from (0, ..., 0) to (1, ..., 0) if ieee_standard else (1, ..., 1) 
            if use_ieee_standard:
                if E_str_iter == (1,) * num_exponent_bits:
                    break
            for F_str_iter in product(*[[0, 1]] * num_fraction_bits): # except cases for NaN
                if not use_ieee_standard and E_str_iter == (1,) * num_exponent_bits and F_str_iter == (1,) * num_fraction_bits:
                    break
                E_str = "".join(str(i) for i in E_str_iter)
                F_str = "".join(str(i) for i in F_str_iter)

                # encoded exponent
                E_enc = decode_binary_str(E_str)
                E_eff = E_enc - bias
                if E_eff == exp_lower:
                    is_subnormal = 1
                else:
                    is_subnormal = 0

                F_enc = decode_binary_str(F_str) * 2**-num_fraction_bits
                F_eff = F_enc + 1 - is_subnormal

                fp8_val = S * 2.0 ** (E_enc - bias + is_subnormal) * F_eff
                all_values.append(fp8_val)
    res = np.array(all_values)
    res = np.sort(res)
    return res


def decode_float8(S, E, F, bias=16):
    sign = -2 * int(S) + 1
    exponent = int(E, 2) if E else 0
    # Normal FP8   : exponent > 0 : result = 2^(exponent-bias) * 1.F
    # Subnormal FP8: exponent == 0: result = 2^(-bias+1)       * 0.F
    # Lowest quantization bin: 2^(-bias+1)       * {0.0 ... 1 + (2^mantissa-1)/2^mantissa}
    # All other bins         : 2^(exponent-bias) * {1.0 ... 1 + (2^mantissa-1)/2^mantissa}; exponent > 0
    A = int(exponent != 0)
    fraction = A + sum([2 ** -(i + 1) * int(a) for i, a in enumerate(F)])
    exponent += int(exponent == 0)
    return sign * fraction * 2.0 ** (exponent - bias)


def get_max_value(num_total_bits: int = 8, num_exponent_bits: int = 4, bias: int = 8, use_ieee_standard: bool = True):
    num_fraction_bits = num_total_bits - 1 - num_exponent_bits
    if use_ieee_standard:
        scale = 2 ** (-num_fraction_bits)
        max_frac = 1 - scale
        max_value = 2 ** (2**num_exponent_bits - 2 - bias) * (1 + max_frac)
    else:
        scale = 2 ** (-num_fraction_bits)
        max_frac = 1 - scale * 2
        max_value = 2 ** (2**num_exponent_bits - 1 - bias) * (1 + max_frac)

    return max_value


def quantize_to_fp8_ste_MM(
    x_float: torch.Tensor,
    n_bits: int,
    maxval: torch.Tensor,
    num_mantissa_bits: torch.Tensor,
    sign_bits: int,
    use_ieee_standard: bool = True,
) -> torch.Tensor:
    """
    Simpler FP8 quantizer that exploits the fact that FP quantization is just INT quantization with
    scales that depend on the input.

    This allows to define FP8 quantization using STE rounding functions and thus learn the bias

    """
    M = torch.clamp(round_ste.apply(num_mantissa_bits), 1, n_bits - sign_bits)
    E = n_bits - sign_bits - M

    if maxval.shape[0] != 1 and len(maxval.shape) != len(x_float.shape):
        maxval = maxval.view([-1] + [1] * (len(x_float.shape) - 1))
    
    if use_ieee_standard: 
        bias = 2**E - torch.log2(maxval) + torch.log2(2 - 2 ** (-M)) - 2
    else:
        bias = 2**E - torch.log2(maxval) + torch.log2(2 - 2 * 2 ** (-M)) - 1

    minval = -maxval if sign_bits == 1 else torch.zeros_like(maxval)
    xc = torch.clamp(x_float, minval, maxval)

    """
    2 notes here:
    1: Shifting by bias to ensure data is aligned to the scaled grid in case bias not in Z.
       Recall that implicitly bias := bias' - log2(alpha), where bias' in Z. If we assume 
       alpha in (0.5, 1], then alpha contracts the grid, which is equivalent to translate the
       data 'to the right' relative to the grid, which is what the subtraction of log2(alpha) 
       (which is negative) accomplishes. 
    2: Ideally this procedure doesn't affect gradients wrt the input (we want to use the STE).
       We can achieve this by detaching log2 of the (absolute) input.

    """

    log_scales = torch.clamp((floor_ste.apply(torch.log2(torch.abs(xc)) + bias)), 1.0)

    scales = 2.0 ** (log_scales - M - bias)

    result = round_ste.apply(xc / scales) * scales
    return result


class FP8QuantizerFunc(Function):
    @staticmethod
    def forward(ctx, x_float, n_bits, maxval, mantissa_bits, sign_bits, use_ieee_standard):
        return quantize_to_fp8_ste_MM(x_float, n_bits, maxval, mantissa_bits, sign_bits, use_ieee_standard)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None, None, None, None


class QuantizerBase(nn.Module):
    def __init__(self, n_bits, per_channel=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_bits = n_bits
        self.per_channel = per_channel
        self.state = None
        self.x_min_fp32 = self.x_max_fp32 = None


class FPQuantizer(QuantizerBase):
    """
    8-bit Floating Point Quantizer
    """

    def __init__(
        self,
        *args,
        scale_domain=None,
        mantissa_bits=4,
        maxval=None,
        set_maxval=False,
        learn_maxval=False,
        learn_mantissa_bits=False,
        mse_include_mantissa_bits=True,
        allow_unsigned=False,
        use_ieee_standard=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.mantissa_bits = mantissa_bits

        self.ebits = self.n_bits - self.mantissa_bits - 1
        self.default_bias = 2 ** (self.ebits - 1)

        # assume signed, correct when range setting turns out to be unsigned
        if use_ieee_standard:
            default_maxval = (2 - 2 ** (-self.mantissa_bits)) * 2 ** (
                2**self.ebits - 2 - self.default_bias
            )
        else:
            default_maxval = (2 - 2 ** (-self.mantissa_bits + 1)) * 2 ** (
                2**self.ebits - 1 - self.default_bias
            )

        self.maxval = maxval if maxval is not None else default_maxval

        self.maxval = torch.Tensor([self.maxval])
        self.mantissa_bits = torch.Tensor([float(self.mantissa_bits)])

        self.set_maxval = set_maxval
        self.learning_maxval = learn_maxval
        self.learning_mantissa_bits = learn_mantissa_bits
        self.mse_include_mantissa_bits = mse_include_mantissa_bits

        self.allow_unsigned = allow_unsigned
        self.sign_bits = 1
        
        self.use_ieee_standard = use_ieee_standard

    def forward(self, x_float):
        if self.maxval.device != x_float.device:
            self.maxval = self.maxval.to(x_float.device)
        if self.mantissa_bits.device != x_float.device:
            self.mantissa_bits = self.mantissa_bits.to(x_float.device)

        res = FP8QuantizerFunc.apply(
            x_float, self.n_bits, self.maxval, self.mantissa_bits, self.sign_bits, self.use_ieee_standard
        )

        return res