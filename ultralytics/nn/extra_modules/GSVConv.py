import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.cuda.amp import autocast

from ..modules.conv import Conv

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except Exception:
    selective_scan_fn = None

__all__ = ['GSVAdaptiveConv', 'GSVSequenceBlock', 'GSVConv', 'GSVConvLarge']


class GSVAdaptiveConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        self.get_weight = nn.Sequential(
            nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=1, groups=in_channel, bias=False),
        )
        self.generate_feature = nn.Sequential(
            nn.Conv2d(
                in_channel,
                in_channel * (kernel_size ** 2),
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=stride,
                groups=in_channel,
                bias=False,
            ),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU(),
        )
        self.conv = Conv(in_channel, out_channel, k=kernel_size, s=kernel_size, p=0)

    def forward(self, x):
        b, c = x.shape[0:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h, w)
        weighted_data = feature * weighted
        conv_data = rearrange(
            weighted_data,
            'b c (n1 n2) h w -> b c (h n1) (w n2)',
            n1=self.kernel_size,
            n2=self.kernel_size,
        )
        return self.conv(conv_data)


def index_reverse(index):
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1]).to(index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r


def semantic_neighbor(x, index):
    dim = index.dim()
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)

    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)

    shuffled_x = torch.gather(x, dim=dim - 1, index=index)
    return shuffled_x


class Selective_Scan(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            device=None,
            dtype=None,
            **kwargs,
    ):
        if selective_scan_fn is None:
            raise ImportError("selective_scan_fn is unavailable. Install mamba-ssm dependencies to use GSVConv.")
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)
        self.selective_scan = selective_scan_fn

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, prompt):
        B, L, C = x.shape
        K = 1
        xs = x.permute(0, 2, 1).view(B, 1, C, L).contiguous()

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) + prompt
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x: torch.Tensor, prompt, **kwargs):
        b, l, c = prompt.shape
        prompt = prompt.permute(0, 2, 1).contiguous().view(b, 1, c, l)
        y = self.forward_core(x, prompt)
        y = y.permute(0, 2, 1).contiguous()
        return y


class GSVSequenceModel(nn.Module):
    def __init__(self, dim, d_state, input_resolution, num_tokens=64, inner_rank=128, mlp_ratio=2.):
        super().__init__()
        if selective_scan_fn is None:
            raise ImportError("selective_scan_fn is unavailable. Install mamba-ssm dependencies to use GSVConv.")
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank

        self.expand = mlp_ratio
        hidden = int(self.dim * self.expand)
        self.d_state = d_state
        self.selectiveScan = Selective_Scan(d_model=hidden, d_state=self.d_state, expand=1)
        self.out_norm = nn.LayerNorm(hidden)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden, dim, bias=True)

        self.in_proj = nn.Sequential(
            nn.Conv2d(self.dim, hidden, 1, 1, 0),
        )

        self.CPE = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden),
        )

        self.embeddingB = nn.Embedding(self.num_tokens, self.inner_rank)
        self.embeddingB.weight.data.uniform_(-1 / self.num_tokens, 1 / self.num_tokens)

        self.route = nn.Sequential(
            nn.Linear(self.dim, self.dim // 3),
            nn.GELU(),
            nn.Linear(self.dim // 3, self.num_tokens),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, x_size, token):
        B, n, C = x.shape
        H, W = x_size
        input_dtype = x.dtype
        if self.route[0].weight.dtype != torch.float32:
            self.route = self.route.float()
            self.in_proj = self.in_proj.float()
            self.CPE = self.CPE.float()
            self.selectiveScan = self.selectiveScan.float()
            self.out_norm = self.out_norm.float()
            self.out_proj = self.out_proj.float()
        compute_dtype = torch.float32

        with autocast(enabled=False):
            x = x.to(dtype=compute_dtype)
            full_embedding = self.embeddingB.weight.to(dtype=compute_dtype) @ token.weight.to(dtype=compute_dtype)

            pred_route = self.route(x)
            cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)

            prompt = torch.matmul(cls_policy, full_embedding).view(B, n, self.d_state)

            detached_index = torch.argmax(cls_policy.detach(), dim=-1, keepdim=False).view(B, n)
            x_sort_values, x_sort_indices = torch.sort(detached_index, dim=-1, stable=False)
            x_sort_indices_reverse = index_reverse(x_sort_indices)

            x_feat = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
            x_feat = self.in_proj(x_feat)
            x_feat = x_feat * torch.sigmoid(self.CPE(x_feat))
            cc = x_feat.shape[1]
            x_feat = x_feat.view(B, cc, -1).contiguous().permute(0, 2, 1)

            semantic_x = semantic_neighbor(x_feat, x_sort_indices)
            y = self.selectiveScan(semantic_x, prompt)
            y = self.out_norm(y)
            y = self.out_proj(y)
            x_out = semantic_neighbor(y, x_sort_indices_reverse)

        return x_out.to(input_dtype)


class GSVSequenceBlock(nn.Module):
    """Sequence modeling block for NCHW tensors."""

    def __init__(self, dim, d_state, num_tokens=64, inner_rank=128, mlp_ratio=2.):
        super().__init__()
        self.sequence_model = GSVSequenceModel(
            dim=dim,
            d_state=d_state,
            input_resolution=None,
            num_tokens=num_tokens,
            inner_rank=inner_rank,
            mlp_ratio=mlp_ratio,
        )
        self.embeddingA = nn.Embedding(inner_rank, d_state)
        self.embeddingA.weight.data.uniform_(-1 / inner_rank, 1 / inner_rank)

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.flatten(2).transpose(1, 2).contiguous()
        y = self.sequence_model(x_flat, (h, w), self.embeddingA)
        y = y.transpose(1, 2).contiguous().view(b, c, h, w)
        return y


class GSVConv(nn.Module):
    """
    Hybrid feature extractor that applies a receptive-field adaptive convolution
    followed by sequence modeling, returning a residual refinement.
    """

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, d_state=16, num_tokens=64, inner_rank=128,
                 mlp_ratio=2.0):
        super().__init__()
        if selective_scan_fn is None:
            raise ImportError("selective_scan_fn is unavailable. Install mamba-ssm dependencies to use GSVConv.")
        self.adaptive_conv = GSVAdaptiveConv(in_channel, out_channel, kernel_size, stride=stride)
        self.norm = nn.BatchNorm2d(out_channel)
        self.act = nn.SiLU()
        self.sequence_block = GSVSequenceBlock(
            dim=out_channel,
            d_state=d_state,
            num_tokens=num_tokens,
            inner_rank=inner_rank,
            mlp_ratio=mlp_ratio,
        )

    def forward(self, x):
        x = self.adaptive_conv(x)
        x = self.act(self.norm(x))
        y = self.sequence_block(x)
        return y + x


class GSVConvLarge(nn.Module):
    """
    Higher-capacity GSVConv variant with deeper/larger sequence modeling to increase FLOPs/accuracy.
    """

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        d_state=32,
        num_tokens=128,
        inner_rank=256,
        mlp_ratio=3.0,
        depth=2,
    ):
        super().__init__()
        if selective_scan_fn is None:
            raise ImportError("selective_scan_fn is unavailable. Install mamba-ssm dependencies to use GSVConvLarge.")
        self.adaptive_conv = GSVAdaptiveConv(in_channel, out_channel, kernel_size, stride=stride)
        self.norm = nn.BatchNorm2d(out_channel)
        self.act = nn.SiLU()
        self.sequence_blocks = nn.ModuleList(
            [
                GSVSequenceBlock(
                    dim=out_channel,
                    d_state=d_state,
                    num_tokens=num_tokens,
                    inner_rank=inner_rank,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        x = self.adaptive_conv(x)
        x = self.act(self.norm(x))
        for block in self.sequence_blocks:
            x = block(x) + x  # residual refinement per block
        return x
