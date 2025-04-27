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
        height_out * width_out,
        dtype=x.dtype,
        device=x.device
    )
    
    for curr_height in range(height_out):
        for curr_width in range(width_out):
            patch = x[
                curr_height : curr_height + height_k,
                curr_width : curr_width + width_k
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
    """ Implementation of Winograd method for 2D convolution, which uses stepwise output tile size 2x2 and kernel size 3x3 matrix multiplication of image patches and kernel to compute the convolution. """
    height_x, width_x = x.shape
    height_k, width_k = k.shape
    height_out = height_x - height_k + 1
    width_out = width_x - width_k + 1
    
    if height_k != 3 or width_k != 3:
        raise ValueError("Kernel must be 3x3 for this Winograd implementation")
    
    bottom_padding = (2 - height_out % 2) % 2
    right_padding = (2 - width_out % 2) % 2
    x_padded = torch.nn.functional.pad(
        x, 
        (
            0,
            right_padding + 2,
            0,
            bottom_padding + 2
        )
    )
    
    input_transform = torch.tensor(
        [
            [1, 0, -1, 0],
            [0, 1, 1, 0],
            [0, -1, 1, 0],
            [0, -1, 0, 1],
        ], 
        dtype=x_padded.dtype,
        device=x_padded.device
    )
    
    kernel_transform = torch.tensor(
        [
            [1, 0, 0],
            [0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0, 0, 1],
        ], 
        dtype=k.dtype,
        device=k.device
    )
    
    output_transform = torch.tensor(
        [
            [1, 1, 1, 0],
            [0, 1, -1, 1],
        ], 
        dtype=x_padded.dtype,
        device=x_padded.device
    )
    
    kernel_transformed = (
        kernel_transform
        @ k
        @ kernel_transform.T
    )
    
    output = torch.zeros(
        height_out,
        width_out,
        dtype=x_padded.dtype,
        device=x_padded.device
    )
    
    for curr_height in range(0, height_out, 2):
        for curr_width in range(0, width_out, 2):
            input_tile = x_padded[
                curr_height : curr_height + 4,
                curr_width : curr_width + 4,
            ]
            
            input_tile_transformed = (
                input_transform
                @ input_tile
                @ input_transform.T
            )
            
            element_wise_multiplication = input_tile_transformed * kernel_transformed
            
            output_tile = (
                output_transform
                @ element_wise_multiplication
                @ output_transform.T
            )
            
            output_h_size = min(2, height_out - curr_height)
            output_w_size = min(2, width_out - curr_width)
            
            output[
                curr_height : curr_height + output_h_size,
                curr_width : curr_width + output_w_size,
            ] = output_tile[
                : output_h_size,
                : output_w_size,
            ]
    
    return output + b


def fft(x, k, b):
    """Implementation of FFT method for 2D convolution, which converts to the frequency domain to compute the convolution as multiplication of two Fourier transforms."""
    height_x, width_x = x.shape
    height_k, width_k = k.shape
    
    height_padded = height_x + height_k - 1
    width_padded = width_x + width_k - 1
    
    x_padded = torch.zeros(
        height_padded,
        width_padded,
        device=x.device,
        dtype=x.dtype
    )
    x_padded[
        : height_x,
        : width_x,
    ] = x
    
    k_padded = torch.zeros(
        height_padded,
        width_padded,
        device=k.device,
        dtype=k.dtype
    )
    k_flipped = k.flip(
        (0, 1)
    )
    k_padded[
        : height_k,
        : width_k,
    ] = k_flipped
    
    x_fft = torch.fft.fft2(x_padded)
    k_fft = torch.fft.fft2(k_padded)
    
    result_fft = x_fft * k_fft
    
    result_full = torch.real(
        torch.fft.ifft2(result_fft)
    )
    
    result = result_full[
        height_k - 1 : height_x,
        width_k - 1 : width_x,
    ]
    
    return result + b