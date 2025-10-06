import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch import autograd

class Conv2dGrad(autograd.Function):
    """
    Autograd Function that performs a backward pass for the input gradient
    by approximating the weight kernel as a sequence of two convolutions (P and Q).
    """
    @staticmethod
    def forward(context, input, weight, P, Q, bias, stride, padding, dilation, groups):
        # Standard forward convolution
        output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
        
        # Save tensors and parameters for backward pass
        context.save_for_backward(input, weight, P, Q, bias)
        context.stride = stride
        context.padding = padding
        context.dilation = dilation
        context.groups = groups
        context.rank = P.shape[1] # out_channels, rank, 1, 1

        return output

    @staticmethod
    def backward(context, grad_output):
        input, weight, P, Q, bias = context.saved_tensors
        grad_input = grad_weight = grad_P = grad_Q = grad_bias = intermediate_grad = None

        # To avoid in-place modification errors
        grad_output = grad_output.contiguous()

        if context.needs_input_grad[1]:
             grad_weight = torch.nn.grad.conv2d_weight(
                input=input,
                weight_size=weight.shape,
                grad_output=grad_output,
                stride=context.stride,
                padding=context.padding,
                dilation=context.dilation,
                groups=context.groups
            )

        ## 2. Calculate the approximate gradient for the input (x)
        if context.needs_input_grad[0]:

            b, c, h, w = grad_output.shape
            intermediate_grad_size = torch.Size((b, context.rank, h, w))
            
            intermediate_grad = torch.nn.grad.conv2d_input(
                input_size=intermediate_grad_size,
                weight=P,
                grad_output=grad_output,
                stride=1,
                padding=0
            )
            
            grad_input = torch.nn.grad.conv2d_input(
                input_size=input.shape,
                weight=Q, # shape (rank, in_channels, k, k)
                grad_output=intermediate_grad,
                stride=context.stride,
                padding=context.padding,
                dilation=context.dilation,
                groups=context.groups
            )

        if context.needs_input_grad[2]:
            # p_matrix = P.squeeze()
            # q_matrix = Q.view(Q.size(0), -1)

            # # Perform matrix multiplication and reshape back to the original weight's shape
            # w_approx = (p_matrix @ q_matrix).view_as(weight)

            # # Define the error E
            # E = (w_approx - weight)
            # # E = grad_weight
            # grad_P = F.conv2d(E, Q)

            p_matrix = P.squeeze()
            q_matrix = Q.view(Q.size(0), -1)
            w_flat = weight.view(weight.size(0), -1)

            # --- 1. Calculate the error signal E in the flattened space ---
            E_flat = (p_matrix @ q_matrix - w_flat.detach())

            grad_p_flat = E_flat @ q_matrix.t()
            grad_P = grad_p_flat.view_as(P)

        ## 4. Calculate the gradient for Q
        # This is calculated as the gradient of a standard convolutional layer
        # where the input is `input` and the output gradient is `intermediate_grad`.
        if intermediate_grad is not None and context.needs_input_grad[3]:
            # grad_Q = torch.nn.grad.conv2d_weight(
            #     input=input,
            #     weight_size=Q.shape,
            #     grad_output=intermediate_grad,
            #     stride=context.stride,
            #     padding=context.padding,
            #     dilation=context.dilation,
            #     groups=context.groups
            # )
            # grad_Q = F.conv2d(
            #     input=E.permute(1, 0, 2, 3),    # Shape: [in_ch, out_ch, k, k]
            #     weight=P.permute(1, 0, 2, 3)    # Shape: [rank, out_ch, 1, 1]
            # ).permute(1, 0, 2, 3) 

            grad_q_flat = p_matrix.t() @ E_flat
            grad_Q = grad_q_flat.view_as(Q)
            
        ## 5. Calculate the gradient for the bias
        if bias is not None and context.needs_input_grad[4]:
            grad_bias = grad_output.sum(dim=[0, 2, 3])

        return grad_input, grad_weight, grad_P, grad_Q, grad_bias, None, None, None, None


class Conv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        rank: int,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        layer_config: dict = None,
        update_P: bool = True,
        update_Q: bool = True
    ):
        # Convert kernel_size to a tuple
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.layer_config = layer_config or {}
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )

        if "options" not in self.layer_config:
            self.layer_config["options"] = {
                "gradient_clip": True,
                "init": "kaiming",
                "svd_niter": 2,
                "clip_value": 0.1
            }
        self.options = self.layer_config["options"]
        self.init_method = self.options["init"]
        
        # Rank for the factorization
        self.rank = min(rank, in_channels, out_channels)
        self.svd_niter = self.layer_config.get("svd_niter", 2)
        
        # P is the 1x1 convolution kernel, projecting from rank -> out_channels
        self.P = nn.Parameter(torch.Tensor(self.out_channels, self.rank, 1, 1), requires_grad=update_P)
        
        # Q is the k x k convolution kernel, projecting from in_channels -> rank
        self.Q = nn.Parameter(torch.Tensor(self.rank, self.in_channels // self.groups, *self.kernel_size), requires_grad=update_Q)
        
        if self.bias is None:
            self.register_parameter("bias", None)
            
        self.init_parameters()

        if self.options['gradient_clip']:
            clip_value = self.options.get('clip_value', 0.1)
            for param in [self.P, self.Q]:
                if param.requires_grad:
                    param.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value) if grad is not None else None)

    def init_pq_svd(self, niter: int = 10):
        """
        Initializes P and Q by finding the low-rank approximation of the flattened
        convolution kernel `W` using randomized SVD. This ensures P * Q ~ W.
        """
        if self.rank == 0:
            self.P.data.zero_()
            self.Q.data.zero_()
            return
            
        # 1. Flatten the weight tensor: (out_ch, in_ch, k, k) -> (out_ch, in_ch * k * k)
        W_flat = self.weight.data.view(self.out_channels, -1)
        
        # 2. Perform low-rank SVD on the flattened weight
        U, S, V = torch.svd_lowrank(W_flat, q=self.rank, niter=niter)
        
        # V is returned, not V.T
        Vt = V.t() # Shape: (rank, in_ch * k * k)
        
        # 3. Create P and Q from SVD components
        sqrt_S = torch.sqrt(S)
        P_flat = U * sqrt_S.unsqueeze(0)        # (out_ch, rank)
        Q_flat = sqrt_S.unsqueeze(1) * Vt      # (rank, in_ch * k * k)
        
        # 4. Reshape P and Q to their proper kernel shapes
        self.P.data = P_flat.view(self.out_channels, self.rank, 1, 1)
        self.Q.data = Q_flat.view(self.rank, self.in_channels // self.groups, *self.kernel_size)


    def init_parameters(self) -> None:
        # Initialize the main weight `W` using standard methods
        if self.init_method == "xavier":
            nn.init.xavier_uniform_(self.weight)
        elif self.init_method == 'kaiming':
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in > 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

        # Initialize P and Q to be a low-rank approximation of the initialized weight `W`
        self.init_pq_svd(niter=self.svd_niter)

    def forward(self, x: Tensor) -> Tensor:
        return Conv2dGrad.apply(
            x, self.weight, self.P, self.Q, self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )


# test

if __name__ == "__main__":

    layer_1 = Conv2d(16, 32, 3, rank=8, padding=1)
    layer_2 = Conv2d(32, 32, 3, padding=1, rank=8)
    layer3 = Conv2d(32, 32, 3, padding=1, rank=8)

    x = torch.randn(2, 16, 64, 64)
    y1 = layer_1(x) 
    y2 = layer_2(y1)
    y3 = layer3(y2)
    l = ((y3 - torch.randn(y2.shape))**2).sum()
    l.backward()
    print("Output difference:", torch.norm(y1 - y2).item())