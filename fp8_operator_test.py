import torch
from fairseq.quantization.utils.fp8_quant import *

# FP8 quantizer test
if __name__ == '__main__':
    x = torch.abs(torch.randn(10, 10))
    quantizer_1 = FPQuantizer(n_bits=8, mantissa_bits=2, sign_bits=1, use_ieee_standard=True) # S1E5M2
    quantizer_2 = FPQuantizer(n_bits=8, mantissa_bits=3, sign_bits=1, use_ieee_standard=True) # S1E4M3
    quantizer_3 = FPQuantizer(n_bits=8, mantissa_bits=3, sign_bits=0, use_ieee_standard=True) # S0E5M3
    
    print(quantizer_1(x))
    print(quantizer_2(x))
    print(quantizer_3(x))
    