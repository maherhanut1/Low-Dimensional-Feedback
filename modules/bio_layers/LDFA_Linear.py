import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch import autograd


@torch.compiler.allow_in_graph
class LinearGrad(autograd.Function):
    """
    Autograd Function that Does a backward pass using the B matrix of the layer
    """
    @staticmethod
    # Same as reference linear function, but with additional weight tensor for backward
    def forward(context, input, weight, P, Q, bias=None):
        output = F.linear(input, weight, bias)
        context.save_for_backward(input, weight, P, Q, bias)
        return output

    @staticmethod
    def backward(context, grad_output):
        input, weight, P, Q, bias = context.saved_tensors
        # Cast to match grad_output dtype (needed for AMP/FP16 compatibility)
        input = input.to(grad_output.dtype)
        weight = weight.to(grad_output.dtype)
        P = P.to(grad_output.dtype)
        Q = Q.to(grad_output.dtype)
        grad_input = grad_weight = grad_Q = grad_P = grad_bias = grad_input_intermediate = None
        # Gradient input
        
        if context.needs_input_grad[0]:
            grad_input_intermediate = grad_output @ (P)
            grad_input = grad_input_intermediate @ (Q)
        
        if context.needs_input_grad[1]:
            # grad_output = grad_output.reshape(-1, grad_output.shape[-1])
            # input = input.view(-1, input.shape[-1])
            # grad_weight = grad_output.t() @ (input)
            grad_weight = torch.einsum('...o,...i->oi', grad_output, input)
        
        if context.needs_input_grad[2]:
            # grad_P = grad_output.t() @ (input @ Q.t())
            grad_P = grad_weight @ Q.t() #TODO ADD IF STATEMENT TO USE THE MORE EFFICIENT ONE DEPENDING ON DIMENSIONS
              
        if grad_input_intermediate is not None and context.needs_input_grad[3]:
            # grad_Q = grad_input_intermediate.view(-1, grad_input_intermediate.shape[-1]).t() @ (input)
            grad_Q = P.t() @ grad_weight
        
        # Gradient bias
        if bias is not None and context.needs_input_grad[4]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_P, grad_Q, grad_bias



class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True, layer_config: dict = None, update_P = True, update_Q = True, requires_gt = False) -> None:
        super(Linear, self).__init__(in_features, out_features, bias)

        if "options" not in self.layer_config:
            self.layer_config["options"] = {
                "gradient_clip": True,
                "init": "kaiming",
                "svd_niter": 2,
                "clip_value": 10.0
            }
            
        self.options = self.layer_config["options"]
        self.init = self.options["init"]
        self.rank = rank
        self.svd_niter = self.layer_config.get("svd_niter", 2) 
        self.Q = nn.Parameter(torch.Tensor(self.rank, in_features), requires_grad=update_Q)
        self.P = nn.Parameter(torch.Tensor(out_features, self.rank), requires_grad=update_P)
        
        if self.bias is None:
            self.register_parameter("bias", None)
            
        self.init_parameters()


        if self.options['gradient_clip']:
            clip_value = self.options.get('clip_value', 10.0)
            for param in self.parameters():
                if param.requires_grad:
                    param.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
    
    
    def init_svd_approx(self, niter: int = 2):
        """
        Initialize P and Q using a **randomized** SVD to approximate the weight matrix W.
        This is much more efficient than a full SVD for large matrices.
        """

        U, S, V = torch.svd_lowrank(self.weight.data, q=self.rank, niter=2)
        
        # Note: torch.svd_lowrank returns V, not V.T as torch.linalg.svd does.
        # V has shape (in_features, rank), so we need its transpose.
        Vt = V.t() # Vt shape: (rank, in_features)
        
        # Initialize P and Q such that P @ Q ≈ W
        # P = U * sqrt(S), Q = sqrt(S) * Vt
        sqrt_S = torch.sqrt(S)
        self.P.data = (U * sqrt_S.unsqueeze(0)).contiguous()        # (out_features, rank)
        self.Q.data = (sqrt_S.unsqueeze(1) * Vt).contiguous()       # (rank, in_features)

    def init_parameters(self) -> None:
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
        # Xavier initialization
        if self.init == "xavier":
            nn.init.xavier_uniform_(self.weight)
            
            # Scaling factor is the standard deviation of xavier init.
            self.scaling_factor = math.sqrt(2.0 / float(fan_in + fan_out))
            if self.bias is not None:
                nn.init.constant_(self.bias, 0)

        # Pytorch Default (Kaiming)
        elif self.init == 'kaiming':
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            
            if self.bias is not None:
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
        

        #Initialize P and Q #TODO: Add other initialization options
        self.init_svd_approx(niter=self.svd_niter)


    def forward(self, x: Tensor, gt=None) -> Tensor:
        return LinearGrad.apply(x, self.weight, self.P, self.Q, self.bias)


    @staticmethod
    def gradient_clip(module, grad_input, grad_output):
        grad_input = list(grad_input)
        for i in range(len(grad_input)):
            if grad_input[i] is not None:
                grad_input[i] = torch.clamp(grad_input[i], -10, 10)
        return tuple(grad_input)