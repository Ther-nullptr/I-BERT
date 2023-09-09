import torch
from fairseq.quantization.utils.quant_modules import *

# FP8 quantizer test
if __name__ == '__main__':
    attn = torch.randn(8, 10, 10).cuda()
    v = torch.randn(8, 10, 4).cuda()

    ground_truth = torch.softmax(attn, dim=-1) @ v
    print(ground_truth)
    
    fp8_linear_softmax = FP8LinearSoftmax(output_bit = 8, quant_mode='symmetric', force_dequant='none', head_dim = 4)
    output = fp8_linear_softmax.forward(attn, v)
    print(output)
