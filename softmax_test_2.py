import torch
from fairseq.quantization.utils.quant_modules import IntSoftmax, FP8LinearSoftmax

if __name__ == '__main__':
    heads = 12
    batch = 16
    seq_len = 114
    head_dim = 64
    
    torch.manual_seed(0)
    attn = torch.load('/home/yujin-wa20/projects/DATE-2024-linear-softmax/I-BERT/attn_weights_0.pt')
    v = torch.load('/home/yujin-wa20/projects/DATE-2024-linear-softmax/I-BERT/v_0.pt')
    
    # baseline
    output_1 = torch.softmax(attn, dim=-1) @ v
    
    # fp8 version
    fp8_linear_softmax = FP8LinearSoftmax(output_bit=8, quant_mode='symmetric', force_dequant='none', x_th_ratio=0.05, head_dim = head_dim, num_heads=heads)
    
    output_3 = fp8_linear_softmax(attn, v)
    
    print(output_1)
    print(output_3)
    