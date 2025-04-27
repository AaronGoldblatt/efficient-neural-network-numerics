import torch
import torch.nn.functional as F


def conv2d(x, k, b, method='naive'):
    """
    Convolution of single instance and single input and output channel
    :param x:  (H, W) PyTorch Tensor
    :param k:  (Hk, Wk) PyTorch Tensor
    :param b:  (1,) PyTorch tensor or scalar
    :param method: Which method do we use to implement it. Valid choices include
                   'naive', 'torch', 'pytorch', 'im2col', 'winograd', and 'fft'
    :return:
        Output tensor should have shape (H_out, W_out)
    """
    method = method.lower()
    if method == 'naive':
        return naive(x, k, b)
    elif method in ['torch', 'pytorch']:
        return pytorch(x, k, b)
    elif method == 'im2col':
        return im2col(x, k, b)
    elif method == 'winograd':
        return winograd(x, k, b)
    elif method == 'fft':
        return fft(x, k, b)
    else:
        raise ValueError("Invalid [method] value: %s" % method)


def naive(x, k, b):
    """ Sliding window solution. """
    output_shape_0 = x.shape[0] - k.shape[0] + 1
    output_shape_1 = x.shape[1] - k.shape[1] + 1
    result = torch.zeros(output_shape_0, output_shape_1)
    for row in range(output_shape_0):
        for col in range(output_shape_1):
            window = x[row: row + k.shape[0], col: col + k.shape[1]]
            result[row, col] = torch.sum(torch.multiply(window, k))
    return result + b


def pytorch(x, k, b):
    """ PyTorch solution. """
    return F.conv2d(
        x.unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
        k.unsqueeze(0).unsqueeze(0),  # (1, 1, Hk, Wk)
        b   # (1, )
    ).squeeze(0).squeeze(0)  # (H_out, W_out)


def im2col(x, k, b):
    """ Implementation of im2col method for 2D convolution, which uses matrix multiplication of image patches and kernel to compute the convolution. """
    height_k, width_k = k.shape
    height_out = x.shape[0] - height_k + 1
    width_out = x.shape[1] - width_k + 1
    
    cols = torch.zeros(
        height_k * width_k,
        height_out * width_out
    )
    
    for curr_height in range(height_out):
        for curr_width in range(width_out):
            patch = x[
                curr_height: curr_height + height_k,
                curr_width: curr_width + width_k
            ].reshape(-1)
            cols[
                :,
                 curr_height * width_out + curr_width
            ] = patch
    
    k_flat = k.flatten()
    
    result = k_flat @ cols
    
    result = result.reshape(height_out, width_out)
    
    return result + b


def winograd(x, k, b):
    """ TODO: implement `winograd`"""
    raise NotImplementedError


def fft(x, k, b):
    """ TODO: implement `fft`"""
    raise NotImplementedError