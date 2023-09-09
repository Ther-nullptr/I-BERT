import torch

if __name__ == '__main__':
    x = torch.randn(4, 5, 6) + torch.tensor(3.)
    x_top, indices = torch.topk(x, 2, dim=-1)
    mask = torch.zeros_like(x, dtype=torch.bool).scatter_(-1, indices, 1)
    # print(mask)
    dense = torch.masked_fill(x, mask, float('-inf'))
    # print(dense)
    sparse = x * mask
    # print(sparse)
    
    x_th = x_top[:, :, -1]
    print(x_th.shape)
    
    x_th_save = x_th.view(-1, 2, x_th.shape[-1]).transpose(0, 1)
    x_th_save = x_th_save.mean(dim=[-1, -2])
    print(x_th_save.shape)
    
    k = torch.exp(x_th) # with an alpha to adjust 
    b = (1 - x_th) * k
    k = k.unsqueeze(-1)
    b = b.unsqueeze(-1)
    
    print(k.shape, b.shape)
    
    dense_part = torch.exp(dense)
    sparse_part = (k * sparse + b) * mask
    
    softmax_estimate_1 = dense_part + sparse_part
    softmax_estimate_2 = torch.sum(softmax_estimate_1, dim=-1, keepdim=True)
    softmax_estimate = softmax_estimate_1 / softmax_estimate_2
    # print(softmax_estimate)
    
    # test softmax
    softmax_ground_truth = torch.softmax(x, dim=-1)
    # print(softmax_ground_truth)
    