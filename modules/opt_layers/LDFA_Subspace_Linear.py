"""
LDFA linear layer with an optimizer-free, subspace-tracked feedback matrix.

The rank-r feedback path is stored as two NON-trainable factors

    Q : (r, in)   orthonormal rows  -> spans an r-dim subspace of the input space
    P : (out, r)  P = W Q^T         -> exact least-squares optimum for the current Q

so that  P Q = W Q^T Q  is the forward weight projected onto span(Q^T), and the
backward signal

    grad_input = grad_output @ P @ Q = (grad_output @ W) Q^T Q

is the true backprop error projected onto that subspace.  W and the bias get
their ordinary gradients (given the projected error coming from above).

Tracking W (call `refresh_feedback` after every optimizer step):

    every step            P <- W Q^T                       1 GEMM (out x in x r)
    every `rotate_every`  Z <- (W Q^T)^T W = Q W^T W       power-iteration step
                          Q <- orthonormalize(Z)           thin QR on (in x r)
                          P <- W Q^T

This is warm-started subspace iteration on W^T W.  Its fixed point is the top-r
right singular subspace of W, i.e. P Q converges to the rank-r truncated SVD of
W.  There is no learning rate, no schedule, no weight decay and no lag in P:
P is always exact for the current W and Q.  Compared with learning P, Q by
gradient descent on ||PQ - W||^2 this needs 1 GEMM per step instead of 3, no
second optimizer / state, no autograd nodes for P, Q and no DDP all-reduce of
P, Q gradients.
"""
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, autograd


@torch.compiler.allow_in_graph
class LinearGrad(autograd.Function):
    """F.linear forward; backward sends the error through the low-rank factors P, Q."""

    @staticmethod
    def forward(ctx, input, weight, bias, P, Q):
        output = F.linear(input, weight, bias)
        ctx.save_for_backward(input, bias, P, Q)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, bias, P, Q = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            # Projected feedback: two skinny GEMMs through rank r (never touches W).
            grad_input = (grad_output @ P.to(grad_output.dtype)) @ Q.to(grad_output.dtype)

        if ctx.needs_input_grad[1]:
            grad_weight = torch.einsum('...o,...i->oi', grad_output, input.to(grad_output.dtype))

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)

        # P and Q are not trained -> no gradient.
        return grad_input, grad_weight, grad_bias, None, None


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True,
                 layer_config: dict = None) -> None:
        self.layer_config = layer_config or {}
        super().__init__(in_features, out_features, bias)

        if "options" not in self.layer_config:
            self.layer_config["options"] = {
                "gradient_clip": True,
                "init": "kaiming",
                "svd_niter": 10,
                "clip_value": 10.0,
            }
        self.options = self.layer_config["options"]
        self.init = self.options["init"]
        self.svd_niter = self.options.get("svd_niter", 10)
        self.rank = min(rank, in_features, out_features)

        # Feedback factors are non-trainable *parameters*, not buffers:
        #  - requires_grad=False  -> DDP does not all-reduce them, autograd ignores them
        #  - parameters (not buffers) -> DDP does not re-broadcast them every forward
        #  - they are still part of state_dict and synced once from rank 0 at DDP init.
        # They are only ever updated in place (copy_) so compile/DDP keep valid references.
        self.P = nn.Parameter(torch.empty(out_features, self.rank), requires_grad=False)
        self.Q = nn.Parameter(torch.empty(self.rank, in_features), requires_grad=False)

        if self.bias is None:
            self.register_parameter("bias", None)

        self.init_parameters()

        if self.options['gradient_clip']:
            clip_value = self.options.get('clip_value', 10.0)
            for param in self.parameters():
                if param.requires_grad:  # weight / bias only
                    param.register_hook(
                        lambda grad: torch.clamp(grad, -clip_value, clip_value) if grad is not None else None)

    # ------------------------------------------------------------------ init
    def init_parameters(self) -> None:
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
        if self.init == "xavier":
            nn.init.xavier_uniform_(self.weight)
            self.scaling_factor = math.sqrt(2.0 / float(fan_in + fan_out))
            if self.bias is not None:
                nn.init.constant_(self.bias, 0)
        elif self.init == 'kaiming':
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
        self.init_svd_approx()

    @torch.no_grad()
    def init_svd_approx(self, niter: Optional[int] = None) -> None:
        """Warm start: Q <- top-r right singular vectors of W (randomized SVD), P <- W Q^T."""
        W = self.weight.detach().to(self.P.dtype)
        _, _, V = torch.svd_lowrank(W, q=self.rank, niter=niter if niter is not None else self.svd_niter)
        V, _ = torch.linalg.qr(V)          # make sure columns are exactly orthonormal
        self.Q.copy_(V.t())
        self.P.copy_(W @ V)

    # -------------------------------------------------------------- tracking
    @torch.no_grad()
    def refresh_feedback(self, rotate: bool = False) -> None:
        """Re-sync the feedback factors to the current W.  Call after optimizer.step().

        rotate=False : P <- W Q^T                      (1 GEMM)
        rotate=True  : one subspace-iteration step on Q, then P <- W Q^T   (3 GEMMs + thin QR)
        """
        W = self.weight.detach()
        if W.dtype != self.P.dtype:
            W = W.to(self.P.dtype)
        if rotate:
            K = W @ self.Q.t()                 # (out, r)
            Z = K.t() @ W                      # (r, in) = Q W^T W  (power step on W^T W)
            V, _ = torch.linalg.qr(Z.t())      # (in, r), orthonormal columns
            self.Q.copy_(V.t())
        self.P.copy_(W @ self.Q.t())           # exact LS-optimal P for the current Q

    @torch.no_grad()
    def feedback_rel_error(self) -> Tensor:
        """||W - P Q||_F / ||W||_F  (diagnostic only; 1 GEMM)."""
        W = self.weight.detach().to(self.P.dtype)
        return torch.linalg.norm(W - self.P @ self.Q) / torch.linalg.norm(W)

    # --------------------------------------------------------------- forward
    def forward(self, x: Tensor, gt=None) -> Tensor:
        return LinearGrad.apply(x, self.weight, self.bias, self.P, self.Q)

    def extra_repr(self) -> str:
        return super().extra_repr() + f', feedback_rank={self.rank}'


# ---------------------------------------------------------------------------
# Helpers for the training loop
# ---------------------------------------------------------------------------
def subspace_layers(model: nn.Module) -> List[Linear]:
    """All LDFA subspace layers inside `model` (looks through compile / DDP wrappers)."""
    return [m for m in model.modules() if isinstance(m, Linear)]


class FeedbackRefresher:
    """Keeps every subspace layer's (P, Q) synced to W.

    Register it as a Trainer `model_modify_fns` entry with `model_modify_iters=1`
    (it is called as fn(trainer) after each optimizer step), or call `.step()`
    yourself after `optimizer.step()`.

    rotate_every : run the subspace-iteration step on Q every this many steps.
                   P is re-synced every step regardless.
    log_every    : if > 0, log the mean relative feedback error to the trainer's
                   TensorBoard writer every this many steps (rank 0 only).
    """

    def __init__(self, model: nn.Module, rotate_every: int = 10, log_every: int = 0):
        self.layers = subspace_layers(model)
        self.rotate_every = max(1, int(rotate_every))
        self.log_every = int(log_every)
        self.step_count = 0

    @torch.no_grad()
    def step(self) -> None:
        self.step_count += 1
        rotate = (self.step_count % self.rotate_every == 0)
        for layer in self.layers:
            layer.refresh_feedback(rotate)

    @torch.no_grad()
    def mean_rel_error(self) -> float:
        if not self.layers:
            return 0.0
        return torch.stack([l.feedback_rel_error() for l in self.layers]).mean().item()

    def __call__(self, trainer=None) -> None:
        self.step()
        if trainer is not None and self.log_every > 0 and self.step_count % self.log_every == 0 \
                and getattr(trainer, 'rank', 0) == 0:
            err = self.mean_rel_error()
            writer = getattr(trainer, 'writer', None)
            if writer is not None:
                writer.add_scalar('ldfa/feedback_rel_error', err, self.step_count)
            print(f'[LDFA] step {self.step_count}: mean ||W-PQ||/||W|| = {err:.4f}')
