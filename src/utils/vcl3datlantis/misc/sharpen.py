import torch
def sharpen(x : torch.Tensor, dim : int, T : float = 0.01):
    '''
    #https://arxiv.org/pdf/1905.02249.pdf
    chech sharpening algorithm
    '''
    x_p = torch.pow(x, 1/T)
    y = x_p / x_p.sum(dim = dim, keepdim = True)
    return y