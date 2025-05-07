import torch
import torch.nn as nn
import torch.nn.functional as F

class SVDLinear(nn.Module):
    """
    A custom linear layer that uses SVD decomposition for compression.
    Instead of storing a single weight matrix W, it stores U, S, V matrices
    from the SVD decomposition W = U @ diag(S) @ V.T
    
    This can provide speedup by:
    1. Reducing the number of parameters (compression)
    2. Performing two smaller matrix multiplications instead of one large one
    """
    def __init__(self, in_features, out_features, rank_ratio=1.0, bias=True):
        super(SVDLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize a standard linear layer
        linear = nn.Linear(in_features, out_features, bias=False)
        
        # Apply SVD to weight matrix
        U, S, V = torch.svd(linear.weight.data)
        
        # Calculate rank based on ratio
        full_rank = min(in_features, out_features)
        self.rank = max(1, int(full_rank * rank_ratio))
        
        # Store the SVD components as parameters
        self.U = nn.Parameter(U[:, :self.rank])
        self.S = nn.Parameter(S[:self.rank])
        self.V = nn.Parameter(V[:, :self.rank])
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Calculate compression ratio
        original_params = in_features * out_features + (out_features if bias else 0)
        compressed_params = self.rank * (in_features + out_features + 1) + (out_features if bias else 0)
        self.compression_ratio = original_params / compressed_params
        
    def forward(self, x):
        """
        Forward pass using the decomposed matrices.
        The computation is split into steps:
        1. x @ V → intermediate of shape (batch_size, rank)
        2. intermediate @ (S * U.T) → output of shape (batch_size, out_features)
        
        This is more efficient than reconstructing the full weight matrix
        and then performing a single matrix multiplication.
        """
        # First multiply input by V (batch_size, in_features) @ (in_features, rank)
        x1 = F.linear(x, self.V)  # Result: (batch_size, rank)
        
        # Scale by singular values (multiply each column by the corresponding S value)
        x2 = x1 * self.S  # Result: (batch_size, rank)
        
        # Finally multiply by U.T (batch_size, rank) @ (rank, out_features)
        output = F.linear(x2, self.U.T)  # Result: (batch_size, out_features)
        
        if self.bias is not None:
            output = output + self.bias
            
        return output

class SVDNet(nn.Module):
    """
    Neural network that uses SVD linear layers for MNIST classification.
    """
    def __init__(self, rank_ratio=1.0):
        super(SVDNet, self).__init__()
        self.fc1 = SVDLinear(28 * 28, 512, rank_ratio)
        self.fc2 = SVDLinear(512, 128, rank_ratio)
        self.fc3 = SVDLinear(128, 10, rank_ratio)
        self.rank_ratio = rank_ratio
        
        # Calculate overall compression ratio
        total_params_original = 28*28*512 + 512 + 512*128 + 128 + 128*10 + 10
        total_params_compressed = (
            self.fc1.rank * (28*28 + 512 + 1) + 512 +
            self.fc2.rank * (512 + 128 + 1) + 128 +
            self.fc3.rank * (128 + 10 + 1) + 10
        )
        self.compression_ratio = total_params_original / total_params_compressed

    def forward(self, x):
        bs = x.size(0)
        x = x.view(bs, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output

def convert_model_to_svd(original_model, rank_ratio=1.0):
    """
    Convert a pre-trained standard model to an SVD model with the same weights.
    
    Args:
        original_model: The pre-trained model with standard linear layers
        rank_ratio: The ratio of the full rank to use for SVD compression
        
    Returns:
        A new SVDNet instance with weights initialized from the original model
    """
    # Create a new SVD model
    svd_model = SVDNet(rank_ratio=rank_ratio)
    
    # Apply SVD to each layer
    for layer_name in ['fc1', 'fc2', 'fc3']:
        original_layer = getattr(original_model, layer_name)
        svd_layer = getattr(svd_model, layer_name)
        
        # Apply SVD to the weight matrix
        U, S, V = torch.svd(original_layer.weight.data)
        
        # Set the SVD layer parameters
        svd_layer.U.data = U[:, :svd_layer.rank].clone()
        svd_layer.S.data = S[:svd_layer.rank].clone()
        svd_layer.V.data = V[:, :svd_layer.rank].clone()
        
        # Copy the bias
        if original_layer.bias is not None:
            svd_layer.bias.data = original_layer.bias.data.clone()
    
    return svd_model 