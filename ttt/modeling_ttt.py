"""PyTorch TTT model definition for speed benchmarking"""
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from ttt.configuration_ttt import TTTConfig
from ttt.generation import GenerationMixin, TTTCache

from mamba_ssm.ops.triton.layernorm import RMSNorm, rms_norm_fn
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update

from ttt.triton_kernel.fused_gate_outln import _fuse_gate_ln_kernel
import ttt.triton_kernel.ttt_linear_decode as triton_ttt_linear_decode
import ttt.triton_kernel.ttt_mlp_decode as triton_ttt_mlp_decode

try:
    import tk_ttt_linear_prefill
except:
    tk_ttt_linear_prefill = None

try:
    import tk_ttt_mlp_prefill
except:
    tk_ttt_mlp_prefill = None


def tree_map(fn, inputs):
    if isinstance(inputs, dict):
        out = {}
        for k, val in inputs.items():
            out[k] = fn(val)
    elif isinstance(inputs, list):
        out = []
        for val in inputs:
            out.append(fn(val))
    else:
        out = fn(inputs)
    return out

def tanh(x):
    return 2 * F.sigmoid(2 * x) - 1

def gelu(x):
    return 0.5 * x * (1 + tanh(0.79788456 * (x + 0.044715 * x * x * x)))

# Modified from https://github.com/NVIDIA/Megatron-LM/blob/e33c8f78a35765d5aa37475a144da60e8a2349d1/megatron/core/fusions/fused_bias_gelu.py#L26
def gelu_bwd(x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff

def ln_fwd(x, gamma, beta, eps=1e-6):
    """
    Args:
        x: [B*nh,N,f]
        gamma: [1,nh,1,f]
        beta: [1,nh,1,f]
        eps:

    Returns:
        z: [B*nh,N,f]

    """
    B_nh, N, HF = x.shape
    nh = gamma.shape[1]
    x = x.reshape(-1, nh, N ,HF)

    # Mean and variance computation
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalization
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std

    # Scale and shift
    y = gamma * x_hat + beta
    y = y.reshape(-1, N, HF)
    return y

def ln_fused_l2_bwd(x, l2_target, gamma, beta, eps=1e-6):
    """Batch backward for LayerNorm fused with L2 loss
    Args:
        x: [B*nh,N,f]
        l2_target: [B*nh,N,f]
        gamma: [1,nh,1,f]
        beta: [1,nh,1,f]
        eps:

    Returns:
        grad_l_x: [B*nh,N,f]
    """
    B_nh, N, HF = x.shape
    nh = gamma.shape[1]

    # Mean and variance computation
    mu = x.mean(dim=-1, keepdim=True)  # [B*nh,N=1,1]
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalization
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std  # [B*nh,N=1,f]

    # Scale and shift
    y = gamma * x_hat.reshape(-1, nh, N, HF) + beta  #[1,nh,N=1,f] * [B,nh,N=1,f] + [1,nh,N=1,f]

    grad_output = y - l2_target.reshape(-1, nh, N, HF)  # [B,nh,N=1,f]
    grad_x_hat = (grad_output * gamma).reshape(B_nh, N, HF)  # [B,nh,N=1,f]
    z = (
        (1.0 / HF)
        * (
            HF * grad_x_hat  # [B*nh,1,f]
            - grad_x_hat.sum(dim=-1, keepdim=True)  # [B*nh,1,1]
            - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)  # [B*nh,N=1,f] * [B*nh,N=1,1]
        )
        / std
    )
    return z

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Args:
        q, k: [B,nh,N,f]
        cos, sin: [B,N,f]

    Returns:
        q_embed, k_embed: [B,nh,N,f]
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Modified from https://github.com/huggingface/transformers/blob/1082361a1978d30db5c3932d1ee08914d74d9697/src/transformers/models/llama/modeling_llama.py#L94
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=16, base=10000, device=None, scaling_factor=1.0):
        """
        TTTRotary is equivalent to LlamaLayerLlamaRotaryEmbedding in implementation, except TTT sets max_position_embedding to mini_batch_size.
        """

        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class SwiGluMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.up_gate_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        gate, up = torch.split(self.up_gate_proj(x), split_size_or_sections=self.intermediate_size, dim=-1)
        down_proj = self.down_proj(self.act_fn(gate) * up)
        return down_proj


class Conv(nn.Module):
    
    def __init__(self, layer_idx, config):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.conv_kernel = config.conv_kernel
        self.conv = nn.Conv1d(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            kernel_size=self.conv_kernel,
            groups=self.hidden_size,
            padding=self.conv_kernel - 1,
        )

    def forward(self, x, is_prefill=False, cache_params=None):
        B, N, _ = x.shape

        conv_weights = self.conv.weight.view(self.conv.weight.size(0), self.conv.weight.size(2))

        if is_prefill:
            x = x.transpose(-1, -2).contiguous()
            conv_output = causal_conv1d_fn(
                x, conv_weights, self.conv.bias, activation=None
            ).transpose(-1, -2).contiguous()
            if cache_params is not None:
                conv_cache = F.pad(x, (self.conv_kernel - N, 0))
                cache_params.params_dict["pre_ttt_conv_cache"][self.layer_idx].copy_(conv_cache)

        else:
            assert cache_params is not None
            x = x[:, 0, :]
            conv_output = causal_conv1d_update(
                x,
                cache_params.params_dict['pre_ttt_conv_cache'][self.layer_idx],
                conv_weights,
                self.conv.bias,
                None,
            )
            conv_output = conv_output.unsqueeze(1)

        return conv_output


class TTTBase(nn.Module):

    def __init__(self, config: TTTConfig, layer_idx: Optional[int] = None):
        super().__init__()
        assert layer_idx is not None, "`layer_idx` should not be None"
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.mini_batch_size = config.mini_batch_size
        self.conv_kernel = config.conv_kernel
        
        # `token_idx` is a denominator corresp. to the position of a token in its mini-batch
        # e.g., a later token accumulates grads of earlier ones, thus needs a larger normalizer
        # `token_idx` is initialized to the reciprocal of the position of each token in a mini-batch,
        # and we add a learnable bias to it to make it more flexible
        # [1,1,bs,1]
        token_idx = 1. / torch.arange(1, self.mini_batch_size + 1).reshape(1, 1, -1, 1)
        self.register_buffer('token_idx', token_idx, persistent=False)
        self.learnable_token_idx_bias = nn.Parameter(torch.zeros((1, 1, self.mini_batch_size, 1)))

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.mini_batch_size,
            base=self.config.rope_theta,
        )

        self.qkv_learnable_ttt_lr_proj = nn.Linear(self.hidden_size, 3 * self.hidden_size + self.num_heads, bias=False)
        self.learnable_ttt_lr_bias = nn.Parameter(torch.zeros(1, 1, self.num_heads))
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.conv_q = nn.Conv1d(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            kernel_size=self.conv_kernel,
            groups=self.hidden_size,
            padding=self.conv_kernel - 1,
        )
        self.conv_k = nn.Conv1d(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            kernel_size=self.conv_kernel,
            groups=self.hidden_size,
            padding=self.conv_kernel - 1,
        )

        ttt_norm_weight, ttt_norm_bias = nn.LayerNorm(self.head_dim).weight.data, nn.LayerNorm(self.head_dim).bias.data
        # [1,nh,1,f]
        self.ttt_norm_weight = nn.Parameter(torch.tile(ttt_norm_weight.reshape(1, 1, 1, -1), (1, self.num_heads, 1, 1)))
        self.ttt_norm_bias = nn.Parameter(torch.tile(ttt_norm_bias.reshape(1, 1, 1, -1), (1, self.num_heads, 1, 1)))

        self.post_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)

        if self.config.use_compile:
            self.apply_rotary_pos_emb = torch.compile(apply_rotary_pos_emb)
            self.get_QKV_ttt_lr = torch.compile(self._get_QKV_ttt_lr)
            self.residual_add_ttt_norm = torch.compile(self._residual_add_ttt_norm)
            self.prefill_gate_and_norm = torch.compile(self._prefill_gate_and_norm)
        else:
            self.apply_rotary_pos_emb = apply_rotary_pos_emb
            self.get_QKV_ttt_lr = self._get_QKV_ttt_lr
            self.residual_add_ttt_norm = self._residual_add_ttt_norm
            self.prefill_gate_and_norm = self._prefill_gate_and_norm

    def _get_QKV_ttt_lr(self, hidden_states):
        B, N, D = hidden_states.shape

        XQKV_ttt_lr = self.qkv_learnable_ttt_lr_proj(hidden_states)  # [B,N, 3*F + nh]

        XQKV, ttt_lr = torch.split(XQKV_ttt_lr, split_size_or_sections=[3 * D, self.num_heads], dim=-1)

        # ttt_lr = (fixed ttt base lr) * (learnable lr multiplier)
        # since the loss used in the TTT process is MSE, which averages over feature dimension of each head,
        # we can absorb the division of `head_dim` into ttt_lr for convenience
        ttt_lr = self.config.ttt_base_lr * F.sigmoid(
            (ttt_lr + self.learnable_ttt_lr_bias).permute(0, 2, 1).reshape(-1, N, 1)
        ) / self.head_dim  # ([B,N,nh] + [1,1,nh]) -> [B,nh,N] -> [B*nh,N,1]

        XQK, XGate, XV = torch.split(XQKV, split_size_or_sections=self.hidden_size, dim=-1)  # [B,N,D]
        XV = XV.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3).reshape(-1, N, self.head_dim)

        return XQK, XV, XGate, ttt_lr
    
    def conv_qk_fused(
        self,
        XQK,
        cache_params: Optional[TTTCache] = None,
        is_prefill: bool = False,
    ):
        '''
        Args:
            XQK: (1) prefill: [B,N,F]; (2) decode: [B,N=1,F];
            cache_params: [B,KS,F]

        Returns:
            XQ: [B,N,F]
            XK: [B,N,F]
            in-place update cache_params
        '''
        B, N, D = XQK.shape
        conv_q_weights = self.conv_q.weight.view(self.conv_q.weight.size(0), self.conv_q.weight.size(2))
        conv_k_weights = self.conv_k.weight.view(self.conv_k.weight.size(0), self.conv_k.weight.size(2))
        if is_prefill:
            XQK = XQK.transpose(-1, -2).contiguous()
            XQ = causal_conv1d_fn(
                XQK, conv_q_weights, self.conv_q.bias, activation=None
            ).transpose(-1,-2).contiguous()
            XK = causal_conv1d_fn(
                XQK, conv_k_weights, self.conv_k.bias, activation=None
            ).transpose(-1,-2).contiguous()
            if cache_params is not None:
                conv_cache = F.pad(XQK, (self.conv_kernel - N, 0))
                cache_params.params_dict["conv_cache"][self.layer_idx].copy_(conv_cache)
        else:
            assert cache_params is not None
            XQK = XQK[:,0,:]
            XQ = causal_conv1d_update(
                XQK,
                cache_params.params_dict['conv_cache'][self.layer_idx].clone(),  # clone(): avoid being updated twice
                conv_q_weights,
                self.conv_q.bias,
                None,
            )
            XK = causal_conv1d_update(
                XQK, # [B,F]
                cache_params.params_dict['conv_cache'][self.layer_idx],
                conv_q_weights,
                self.conv_q.bias,
                None,
            )
            XQ = XQ.unsqueeze(1)
            XK = XK.unsqueeze(1)

        return XQ, XK

    def get_inner_loop_inputs(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        cache_params: Optional[TTTCache] = None,
        is_prefill: bool = False,
    ):
        B, N, D = hidden_states.shape
        if is_prefill:
            inner_mini_batch_step_offset = 0
            token_idx = self.token_idx + self.learnable_token_idx_bias
        else:
            inner_mini_batch_step_offset = cache_params.seqlen_offset % self.mini_batch_size
            token_idx = self.token_idx[:, :, inner_mini_batch_step_offset, :] + \
                        self.learnable_token_idx_bias[:, :, inner_mini_batch_step_offset, :]
        # token idx should be greast than 0
        token_idx = torch.clamp_min(token_idx, 0.0)

        XQK, XV, XGate, ttt_lr = self.get_QKV_ttt_lr(hidden_states)

        XQ, XK = self.conv_qk_fused(XQK, cache_params, is_prefill)
        XQ = XQ.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        XK = XK.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Apply rotary on XQ, XK
        cos, sin = self.rotary_emb(XV, position_ids % self.mini_batch_size)
        XQ, XK = self.apply_rotary_pos_emb(XQ, XK, cos, sin)
        XQ = XQ.reshape(-1, N, self.head_dim)
        XK = XK.reshape(-1, N, self.head_dim)

        XV = XV.contiguous()
        XK = XK.contiguous()
        XQ = XQ.contiguous()
        XGate = XGate.contiguous()
        ttt_lr = ttt_lr.contiguous()

        return XQ, XK, XV, token_idx, ttt_lr, XGate


    def TTT_process(
        self,
        inputs,
        is_prefill,
        is_last_in_mini_batch,
        cache_params=None
    ):
        """
        Inputs:
            XV, XK, XQ: [B*nh, N, f]
            token_idx: [1,]
            ttt_lr: [B*nh, N, 1]
        Outputs:
            [B,N,F]
        """
        raise NotImplementedError
    
    def _residual_add_ttt_norm(self, XQ, XQW_batch):
        XQW_batch = XQ + ln_fwd(XQW_batch, self.ttt_norm_weight, self.ttt_norm_bias)
        return XQW_batch

    def _prefill_gate_and_norm(self, B, N, XGate, ttt_process_output):
        ttt_process_output = ttt_process_output.reshape(
            B, self.num_heads, N, self.head_dim
        ).permute(0, 2, 1, 3).reshape(B, N, -1)
        output_hidden_states = gelu(XGate) * self.post_norm(ttt_process_output)
        return output_hidden_states.contiguous()

    def decode_gate_and_norm_fused(self, B, N, XGate, ttt_process_output):
        ttt_process_output = ttt_process_output.reshape(
            B, self.num_heads, N, self.head_dim
        ).permute(0, 2, 1, 3).reshape(B, N, -1).contiguous()
        output_hidden_states = torch.empty_like(ttt_process_output)
        grid = (B, 1, 1)
        _fuse_gate_ln_kernel[grid](XGate, ttt_process_output, output_hidden_states,
                                   self.post_norm.weight.data, self.post_norm.bias.data,
                                   ttt_process_output.stride(0),
                                   self.hidden_size)
        return output_hidden_states
    
    def project_inner_loop_outputs(self, XQW_batch):
        """
        Inputs
            XQW_batch: [B,N,F]
        Outputs
            z_batch: [B,N,F]
        """
        z_batch = self.o_proj(XQW_batch)
        return z_batch

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[TTTCache] = None,
        is_prefill: Optional[bool] = None,
        is_last_in_mini_batch: Optional[bool] = None,
    ):
        B, N = hidden_states.shape[:2]
        # Simplification in benchmark: prefill length is a multiple of mini-batch size
        assert (is_prefill and N % self.mini_batch_size == 0) or ((not is_prefill) and N == 1)

        XQ, XK, XV, token_idx, ttt_lr, XGate = self.get_inner_loop_inputs(
            hidden_states,
            position_ids=position_ids,
            cache_params=cache_params,
            is_prefill=is_prefill
        )  # [B*nh,N,f]
        B_mul_NH = XV.shape[0]
        B = B_mul_NH // self.num_heads
        inputs = {'XQ': XQ, 'XK': XK, 'XV': XV, 'token_idx': token_idx, 'ttt_lr': ttt_lr}

        ttt_process_output = self.TTT_process(
            inputs,
            cache_params=cache_params,
            is_prefill=is_prefill,
            is_last_in_mini_batch=is_last_in_mini_batch,
        )

        if is_prefill:
            output_hidden_states = self.prefill_gate_and_norm(B, N, XGate, ttt_process_output)
        else:
            output_hidden_states = self.decode_gate_and_norm_fused(B, N, XGate, ttt_process_output)

        output_hidden_states = self.project_inner_loop_outputs(output_hidden_states)  # [B,N,F]

        return output_hidden_states


##########################################

def ttt_linear_prefill_mini_batch(states, inputs, i, ttt_norm_weight, ttt_norm_bias):
    """
    Args:
        states: W1_init: [B*nh,f,f], b1_init: [B*nh,1,f]
        inputs: XQ/XK/XV: [B*nh,bs,f], eta: [B*nh,bs,bs], eta_last: [B*nh,1,bs]
        ttt_norm_weight: [1,nh,1,f]
        ttt_norm_bias: [1,nh,1,f]

    Returns:
        Z1_bar: [B*nh,bs,f]
    """
    W1_init = states['W1_init']
    b1_init = states['b1_init']
    XV_mini_batch, XK_mini_batch, XQ_mini_batch, \
    eta_mini_batch, eta_mini_batch_last = inputs['XV'][i], inputs['XK'][i], inputs['XQ'][i], \
                                          inputs['eta'][i], inputs['eta_last'][i]  # [B*nh,mini_batch_size,mini_batch_size], [B*nh,1,mini_batch_size]

    Z1 = XK_mini_batch @ W1_init + b1_init
    l2_target = XV_mini_batch - XK_mini_batch
    dl_dZ1 = ln_fused_l2_bwd(Z1, l2_target, ttt_norm_weight, ttt_norm_bias)

    b1_bar = b1_init - torch.tril(eta_mini_batch) @ dl_dZ1

    Attn1 = torch.tril(XQ_mini_batch @ XK_mini_batch.transpose(-1, -2))
    Z1_bar = XQ_mini_batch @ W1_init - (eta_mini_batch * Attn1) @ dl_dZ1 + b1_bar

    W1_init.sub_((eta_mini_batch_last * XK_mini_batch.transpose(-1, -2)) @ dl_dZ1)
    b1_init.copy_(b1_bar[:,-1:])

    return Z1_bar

def ttt_linear_decode_last_token_in_mini_batch(states, inputs, ttt_norm_weight, ttt_norm_bias):
    """
    Args:
        states: W1_init/W1_grad: [B*nh,f,f], b1_init/b1_grad: [B*nh,1,f]
        inputs: XQ/XK/XV: [B*nh,1,f], token_idx: [1,1,1], ttt_lr: [B*nh,1,1]
        ttt_norm_weight: [1,nh,1,f]
        ttt_norm_bias: [1,nh,1,f]

    Returns:
        Z1_bar: [B*nh,1,f]
    """
    W1 = states['W1_init']
    b1 = states['b1_init']
    W1_grad = states['W1_grad']
    b1_grad = states['b1_grad']

    XV, XK, XQ, \
    token_idx, ttt_lr = inputs['XV'], inputs['XK'], inputs['XQ'], \
                        inputs['token_idx'], inputs['ttt_lr']
    B_mul_NH, K, HF = XV.shape
    NH = ttt_norm_weight.shape[1]

    Z1 = XK @ W1 + b1
    l2_target = XV - XK

    mu = Z1.mean(dim=-1, keepdim=True)
    var = Z1.var(dim=-1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + 1e-6)
    Z1_hat = (Z1 - mu) / std  # [B*nh,K,f]

    # Scale and shift
    LN_out = ttt_norm_weight * Z1_hat.reshape(-1, NH, K, HF) + ttt_norm_bias

    dl_dLN_out = LN_out - l2_target.reshape(-1, NH, K, HF)

    dl_dZ1_hat = (dl_dLN_out * ttt_norm_weight).reshape(B_mul_NH, K, HF)

    dl_dZ1_term_1 = HF * dl_dZ1_hat
    dl_dZ1_term_2 = dl_dZ1_hat.sum(dim=-1, keepdim=True)
    dl_dZ1_term_3 = Z1_hat * (dl_dZ1_hat * Z1_hat).sum(dim=-1, keepdim=True)
    dl_dZ1_sum = dl_dZ1_term_1 - dl_dZ1_term_2 - dl_dZ1_term_3
    dl_dZ1 = dl_dZ1_sum / (std * HF)

    ilr_mul_dl_dZ1 = ttt_lr * dl_dZ1

    W1_grad.add_(XK.transpose(-1, -2) @ ilr_mul_dl_dZ1)
    b1_grad.add_(ilr_mul_dl_dZ1)

    W1.sub_(token_idx * W1_grad)
    b1.sub_(token_idx * b1_grad)
    Z1_bar = XQ @ W1 + b1

    W1_grad.zero_()
    b1_grad.zero_()

    return Z1_bar

def ttt_linear_decode_token(states, inputs, ttt_norm_weight, ttt_norm_bias):
    """
    Args:
        states: W1_init/W1_grad: [B*nh,f,f], b1_init/b1_grad: [B*nh,1,f]
        inputs: X: [B*nh,1,f], token_idx: [1,1,1], ttt_lr: [B*nh,1,1]
        ttt_norm_weight: [1,nh,1,f]
        ttt_norm_bias: [1,nh,1,f]

    Returns:
        Z1_bar: [B*nh,1,f]
    """
    W1 = states['W1_init']
    b1 = states['b1_init']
    W1_grad = states['W1_grad']
    b1_grad = states['b1_grad']

    XV, XK, XQ, \
    token_idx, ttt_lr = inputs['XV'], inputs['XK'], inputs['XQ'], \
                        inputs['token_idx'], inputs['ttt_lr']
    B_mul_NH, K, HF = XV.shape
    NH = ttt_norm_weight.shape[1]

    Z1 = XK @ W1 + b1
    l2_target = XV - XK

    mu = Z1.mean(dim=-1, keepdim=True)
    var = Z1.var(dim=-1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + 1e-6)
    Z1_hat = (Z1 - mu) / std  # [B*nh,K,f]

    # Scale and shift
    LN_out = ttt_norm_weight * Z1_hat.reshape(-1, NH, K, HF) + ttt_norm_bias

    dl_dLN_out = LN_out - l2_target.reshape(-1, NH, K, HF)

    dl_dZ1_hat = (dl_dLN_out * ttt_norm_weight).reshape(B_mul_NH, K, HF)

    dl_dZ1_term_1 = HF * dl_dZ1_hat
    dl_dZ1_term_2 = dl_dZ1_hat.sum(dim=-1, keepdim=True)
    dl_dZ1_term_3 = Z1_hat * (dl_dZ1_hat * Z1_hat).sum(dim=-1, keepdim=True)
    dl_dZ1_sum = dl_dZ1_term_1 - dl_dZ1_term_2 - dl_dZ1_term_3
    dl_dZ1 = dl_dZ1_sum / (std * HF)

    ilr_mul_dl_dZ1 = ttt_lr * dl_dZ1

    W1_grad.add_(XK.transpose(-1, -2) @ ilr_mul_dl_dZ1)
    b1_grad.add_(ilr_mul_dl_dZ1)

    W1_bar = W1 - (token_idx * W1_grad)
    b1_bar = b1 - (token_idx * b1_grad)
    Z1_bar = XQ @ W1_bar + b1_bar

    return Z1_bar


class TTTLinear(TTTBase):

    def __init__(self, config: TTTConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

        if self.config.use_compile:
            self.prefill_mini_batch = torch.compile(ttt_linear_prefill_mini_batch)
            self.decode_last_token_in_mini_batch = torch.compile(ttt_linear_decode_last_token_in_mini_batch)
            self.decode_token = torch.compile(ttt_linear_decode_token)
        else:
            self.prefill_mini_batch = ttt_linear_prefill_mini_batch
            self.decode_last_token_in_mini_batch = ttt_linear_decode_last_token_in_mini_batch
            self.decode_token = ttt_linear_decode_token

    def TTT_process(
        self,
        inputs,
        is_prefill=False,
        is_last_in_mini_batch=False,
        cache_params=None
    ):
        states = {
            "W1_init": cache_params.params_dict["W1_init"][self.layer_idx],
            "b1_init": cache_params.params_dict["b1_init"][self.layer_idx],
            "W1_grad": cache_params.params_dict["W1_grad"][self.layer_idx],
            "b1_grad": cache_params.params_dict["b1_grad"][self.layer_idx],
        }
        XQ_residual = inputs['XQ']

        if is_prefill:
            B_mul_NH, N, HF = inputs['XV'].shape
            n_mini_batch = N // self.mini_batch_size
            token_idx = inputs.pop('token_idx')
            # [B*nh,N,f] -> [B*nh,n_mini_batch,mini_batch_size,f] -> [n_mini_batch,B*nh,mini_batch_size,f]
            inputs = tree_map(lambda x: x.reshape(B_mul_NH, n_mini_batch, self.mini_batch_size, -1).transpose(1,0).contiguous(),
                              inputs)
            ttt_lr = inputs.pop('ttt_lr').transpose(-1,-2)
            # [1,1,mini_batch_size,1] * [n_mini_batch,B*nh,1,mini_batch_size] -> [n_mini_batch, B*nh, mini_batch_size, mini_batch_size]
            inputs['eta'] = token_idx * ttt_lr
            # pre-sclice: [n_mini_batch, B*nh, 1, mini_batch_size]
            inputs['eta_last'] = inputs['eta'][...,-1:,:]

            def prefill_sequence(states, inputs):
                output_tensor = torch.empty_like(inputs['XV'])
                for i in range(n_mini_batch):
                    Z1_bar = self.prefill_mini_batch(states, inputs, i, self.ttt_norm_weight, self.ttt_norm_bias)
                    output_tensor[i] = Z1_bar
                return output_tensor  # [n_mini_batch, B*nh, K, f]

            XQW_batch = prefill_sequence(
                states,
                inputs,
            )
            XQW_batch = XQW_batch.transpose(1,0).reshape(B_mul_NH, -1, HF).contiguous()

        else:
            B_mul_NH, N, HF = inputs['XV'].shape

            if is_last_in_mini_batch:
                XQW_batch = self.decode_last_token_in_mini_batch(
                    states,
                    inputs,
                    self.ttt_norm_weight,
                    self.ttt_norm_bias,
                )
            else:
                XQW_batch = self.decode_token(
                    states,
                    inputs,
                    self.ttt_norm_weight,
                    self.ttt_norm_bias,
                )

        ttt_process_output = self.residual_add_ttt_norm(XQ_residual, XQW_batch)

        return ttt_process_output

##########################################

def ttt_mlp_prefill_mini_batch(states, inputs, i, ttt_norm_weight, ttt_norm_bias):
    """
    Args:
        states: W1: [B*nh,f,4f], b1: [B*nh,1,4f], W2: [B*nh,4f,f], b2: [B*nh,1,f]
        inputs: XQ/XK/XV: [B*nh,bs,f], eta: [B*nh,bs,bs], eta_last: [B*nh,1,bs]
        ttt_norm_weight: [1,nh,1,f]
        ttt_norm_bias: [1,nh,1,f]

    Returns:
        Z2_bar: [B*nh,bs,f]
    """
    W1_init = states['W1_init']
    b1_init = states['b1_init']
    W2_init = states['W2_init']
    b2_init = states['b2_init']
    XV_mini_batch, XK_mini_batch, XQ_mini_batch, \
    eta_mini_batch, eta_mini_batch_last = inputs['XV'][i], inputs['XK'][i], inputs['XQ'][i], \
                                          inputs['eta'][i], inputs['eta_last'][i]

    Z1 = XK_mini_batch @ W1_init + b1_init
    X2 = gelu(Z1)
    Z2 = X2 @ W2_init + b2_init

    l2_target = XV_mini_batch - XK_mini_batch
    dl_dZ2 = ln_fused_l2_bwd(Z2, l2_target, ttt_norm_weight, ttt_norm_bias)
    dl_dZ1 = dl_dZ2 @ W2_init.transpose(-1, -2) * gelu_bwd(Z1)

    b1_bar = b1_init - torch.tril(eta_mini_batch) @ dl_dZ1
    Attn1 = torch.tril(XQ_mini_batch @ XK_mini_batch.transpose(-1, -2))
    Z1_bar = XQ_mini_batch @ W1_init - (eta_mini_batch * Attn1) @ dl_dZ1 + b1_bar
    X2_bar = gelu(Z1_bar)

    b2_bar = b2_init - torch.tril(eta_mini_batch) @ dl_dZ2
    Attn2 = torch.tril(X2_bar @ X2.transpose(-1, -2))
    Z2_bar = X2_bar @ W2_init - (eta_mini_batch * Attn2) @ dl_dZ2 + b2_bar

    W1_init.sub_((eta_mini_batch_last * XK_mini_batch.transpose(-1, -2)) @ dl_dZ1)
    b1_init.copy_(b1_bar[:,-1:])
    W2_init.sub_((eta_mini_batch_last * X2.transpose(-1, -2)) @ dl_dZ2)
    b2_init.copy_(b2_bar[:, -1:])

    return Z2_bar

def ttt_mlp_decode_last_token_in_mini_batch(states, inputs, ttt_norm_weight, ttt_norm_bias):
    """
    Args:
        states:
            W1_init/W1_grad: [B*nh,f,4f], b1_init/b1_grad: [B*nh,1,4f]
            W2_init/W2_grad: [B*nh,4f,f], b2_init/b2_grad: [B*nh,1,f]
        inputs: XQ/XK/XV: [B*nh,1,f], token_idx: [1,1,1], ttt_lr: [B*nh,1,1]
        ttt_norm_weight: [1,nh,1,f]
        ttt_norm_bias: [1,nh,1,f]

    Returns:
        Z2_bar: [B*nh,1,f]
    """
    W1_init = states['W1_init']
    W1_grad = states['W1_grad']
    b1_init = states['b1_init']
    b1_grad = states['b1_grad']
    W2_init = states['W2_init']
    W2_grad = states['W2_grad']
    b2_init = states['b2_init']
    b2_grad = states['b2_grad']

    XV_mini_batch, XK_mini_batch, \
    XQ_mini_batch, token_idx, ttt_lr = inputs['XV'], inputs['XK'], inputs['XQ'], \
                                       inputs['token_idx'], inputs['ttt_lr']  # [B*nh,N=1,f], [1,1,1], [B*nh,N=1,1]

    Z1 = XK_mini_batch @ W1_init + b1_init
    X2 = gelu(Z1)
    Z2 = X2 @ W2_init + b2_init

    l2_target = XV_mini_batch - XK_mini_batch
    dl_dZ2 = ln_fused_l2_bwd(Z2, l2_target, ttt_norm_weight, ttt_norm_bias)
    ilr_mul_dl_dZ2 = ttt_lr * dl_dZ2
    dl_dZ1 = dl_dZ2 @ W2_init.transpose(-1, -2) * gelu_bwd(Z1)
    ilr_mul_dl_dZ1 = ttt_lr * dl_dZ1

    W1_grad.add_(XK_mini_batch.transpose(-1, -2) @ ilr_mul_dl_dZ1)
    b1_grad.add_(ilr_mul_dl_dZ1)
    W1_init.sub_(token_idx * W1_grad)
    b1_init.sub_(token_idx * b1_grad)
    Z1_bar = XQ_mini_batch @ W1_init + b1_init
    X2_bar = gelu(Z1_bar)

    W2_grad.add_(X2.transpose(-1, -2) @ ilr_mul_dl_dZ2)
    b2_grad.add_(ilr_mul_dl_dZ2)
    W2_init.sub_(token_idx * W2_grad)
    b2_init.sub_(token_idx * b2_grad)
    Z2_bar = X2_bar @ W2_init + b2_init

    W1_grad.zero_()
    b1_grad.zero_()
    W2_grad.zero_()
    b2_grad.zero_()

    return Z2_bar

def ttt_mlp_decode_token(states, inputs, ttt_norm_weight, ttt_norm_bias):
    """
    Args:
        states:
            W1_init/W1_grad: [B*nh,f,4f], b1_init/b1_grad: [B*nh,1,4f]
            W2_init/W2_grad: [B*nh,4f,f], b2_init/b2_grad: [B*nh,1,f]
        inputs: XQ/XK/XV: [B*nh,1,f], token_idx: [1,1,1], ttt_lr: [B*nh,1,1]
        ttt_norm_weight: [1,nh,1,f]
        ttt_norm_bias: [1,nh,1,f]

    Returns:
        Z2_bar: [B*nh,1,f]
    """
    W1_init = states['W1_init']
    W1_grad = states['W1_grad']
    b1_init = states['b1_init']
    b1_grad = states['b1_grad']
    W2_init = states['W2_init']
    W2_grad = states['W2_grad']
    b2_init = states['b2_init']
    b2_grad = states['b2_grad']

    XV_mini_batch, XK_mini_batch, \
    XQ_mini_batch, token_idx, ttt_lr = inputs['XV'], inputs['XK'], inputs['XQ'], \
                                       inputs['token_idx'], inputs['ttt_lr']

    Z1 = XK_mini_batch @ W1_init + b1_init
    X2 = gelu(Z1)
    Z2 = X2 @ W2_init + b2_init

    l2_target = XV_mini_batch - XK_mini_batch
    dl_dZ2 = ln_fused_l2_bwd(Z2, l2_target, ttt_norm_weight, ttt_norm_bias)
    ilr_mul_dl_dZ2 = ttt_lr * dl_dZ2
    dl_dZ1 = dl_dZ2 @ W2_init.transpose(-1, -2) * gelu_bwd(Z1)
    ilr_mul_dl_dZ1 = ttt_lr * dl_dZ1

    W1_grad.add_(XK_mini_batch.transpose(-1, -2) @ ilr_mul_dl_dZ1)
    b1_grad.add_(ilr_mul_dl_dZ1)
    W1_last = W1_init - token_idx * W1_grad
    b1_last = b1_init - token_idx * b1_grad
    Z1_bar = XQ_mini_batch @ W1_last + b1_last
    X2_bar = gelu(Z1_bar)

    W2_grad.add_(X2.transpose(-1, -2) @ ilr_mul_dl_dZ2)
    b2_grad.add_(ilr_mul_dl_dZ2)
    W2_last = W2_init - token_idx * W2_grad
    b2_last = b2_init - token_idx * b2_grad
    Z2_bar = X2_bar @ W2_last + b2_last

    return Z2_bar


class TTTMLP(TTTBase):

    def __init__(self, config: TTTConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, 4 * self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, 4 * self.head_dim))
        self.W2 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, 4 * self.head_dim, self.head_dim)))
        self.b2 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

        if self.config.use_compile:
            self.prefill_mini_batch = torch.compile(ttt_mlp_prefill_mini_batch)
            self.decode_last_token_in_mini_batch = torch.compile(ttt_mlp_decode_last_token_in_mini_batch)
            self.decode_token = torch.compile(ttt_mlp_decode_token)
        else:
            self.prefill_mini_batch = ttt_mlp_prefill_mini_batch
            self.decode_last_token_in_mini_batch = ttt_mlp_decode_last_token_in_mini_batch
            self.decode_token = ttt_mlp_decode_token

    def TTT_process(
        self,
        inputs,
        is_prefill=False,
        is_last_in_mini_batch=False,
        cache_params=None
    ):
        states = {
            "W1_init": cache_params.params_dict["W1_init"][self.layer_idx],
            "W1_grad": cache_params.params_dict["W1_grad"][self.layer_idx],
            "b1_init": cache_params.params_dict["b1_init"][self.layer_idx],
            "b1_grad": cache_params.params_dict["b1_grad"][self.layer_idx],
            "W2_init": cache_params.params_dict["W2_init"][self.layer_idx],
            "W2_grad": cache_params.params_dict["W2_grad"][self.layer_idx],
            "b2_init": cache_params.params_dict["b2_init"][self.layer_idx],
            "b2_grad": cache_params.params_dict["b2_grad"][self.layer_idx],
        }
        XQ_residual = inputs['XQ']

        if is_prefill:
            B_mul_NH, N, HF = inputs['XV'].shape  # [B*nh,N,f]
            n_mini_batch = N // self.mini_batch_size
            token_idx = inputs.pop('token_idx')
            # [B*nh,N,f] -> [B*nh,n_mini_batch,mini_batch_size,f] -> [n_mini_batch,B*nh,mini_batch_size,f]
            inputs = tree_map(lambda x: x.reshape(B_mul_NH, n_mini_batch, self.mini_batch_size, -1).transpose(1, 0).contiguous(),
                              inputs)
            ttt_lr = inputs.pop('ttt_lr').transpose(-1, -2)
            # [1,1,mini_batch_size,1] * [n_mini_batch,B*nh,1,mini_batch_size] -> [n_mini_batch,B*nh,mini_batch_size,mini_batch_size]
            inputs['eta'] = token_idx * ttt_lr
            # pre-sclice: [n_mini_batch,B*nh,1,mini_batch_size]
            inputs['eta_last'] = inputs['eta'][..., -1:, :]

            def prefill_sequence(states, inputs):
                output_tensor = torch.empty_like(inputs['XV'])
                for i in range(n_mini_batch):
                    Z2_bar = self.prefill_mini_batch(states, inputs, i, self.ttt_norm_weight, self.ttt_norm_bias)
                    output_tensor[i] = Z2_bar
                return output_tensor

            XQW_batch = prefill_sequence(
                states,
                inputs,
            )
            XQW_batch = XQW_batch.transpose(1, 0).reshape(B_mul_NH, -1, HF).contiguous()

        else:
            B_mul_NH, N, HF = inputs['XV'].shape

            if is_last_in_mini_batch:
                XQW_batch = self.decode_last_token_in_mini_batch(
                    states,
                    inputs,
                    self.ttt_norm_weight,
                    self.ttt_norm_bias,
                )
            else:
                XQW_batch = self.decode_token(
                    states,
                    inputs,
                    self.ttt_norm_weight,
                    self.ttt_norm_bias,
                )

        ttt_process_output = self.residual_add_ttt_norm(XQ_residual, XQW_batch)

        return ttt_process_output

##########################################

class TTTLinearFast(TTTBase):

    def __init__(self, config: TTTConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

    def TTT_process(
        self,
        inputs,
        is_prefill=False,
        is_last_in_mini_batch=False,
        cache_params=None
    ):

        B_mul_NH, N, HF = inputs['XV'].shape
        B = B_mul_NH // self.num_heads
        NH = self.num_heads

        if is_prefill:
            W1_init = cache_params.params_dict["W1_init"][self.layer_idx].reshape(B, NH, HF, HF)
            b1_init = cache_params.params_dict["b1_init"][self.layer_idx].reshape(B, NH, 1, HF)

            mini_batch_size = self.mini_batch_size
            n_mini_batch = N // mini_batch_size
            token_idx = inputs.pop('token_idx')
            # [B*nh,N,f/1] -> [B,nh,n_mini_batch,mini_batch_size,f/1]
            inputs = tree_map(lambda x: x.reshape(B, NH, n_mini_batch, mini_batch_size, -1).contiguous(), inputs)
            # [B,nh,n_mini_batch,1,mini_batch_size]
            ttt_lr = inputs.pop('ttt_lr').transpose(-1, -2)

            # [1,1,1,mini_batch_size,1] * [B,nh,n_mini_batch,1,mini_batch_size] -> [B,nh,n_mini_batch,mini_batch_size,mini_batch_size]
            inputs['eta'] = token_idx[None,:] * ttt_lr

            XV, XK, XQ, eta = inputs['XV'], inputs['XK'], inputs['XQ'], inputs['eta']
            input_device = XV.device
            input_dtype = XV.dtype
            ttt_process_output = torch.empty_like(XV)

            ttt_norm_weight = self.ttt_norm_weight.data.squeeze(0).expand(-1, mini_batch_size, -1).contiguous()
            ttt_norm_bias = self.ttt_norm_bias.data.squeeze(0).expand(-1, mini_batch_size, -1).contiguous()
            b1_init = b1_init.expand(-1, -1, mini_batch_size, -1).contiguous()
            make_last_b_matrix = torch.zeros(mini_batch_size, mini_batch_size, dtype=input_dtype, device=input_device)
            make_last_eta_1_matrix = torch.zeros(mini_batch_size, HF, dtype=input_dtype, device=input_device)
            make_last_b_matrix[:,-1] = 1.
            make_last_eta_1_matrix[-1,:] = 1.

            tk_ttt_linear_prefill.ttt_linear_prefill_fp16(
                W1_init, b1_init, ttt_norm_weight, ttt_norm_bias,
                make_last_b_matrix, make_last_eta_1_matrix,
                XV, XK, XQ, eta, ttt_process_output
            )
            b1_init = b1_init[:,:,-1:,:].reshape(B_mul_NH, 1, -1)
            cache_params.params_dict["b1_init"][self.layer_idx].copy_(b1_init)

            ttt_process_output = ttt_process_output.reshape(B_mul_NH, N, HF)

        else:
            W1 = cache_params.params_dict["W1_init"][self.layer_idx].reshape(B, NH, HF, HF)
            b1 = cache_params.params_dict["b1_init"][self.layer_idx].reshape(B, NH, 1, HF)
            W1_grad = cache_params.params_dict["W1_grad"][self.layer_idx].reshape(B, NH, HF, HF)
            b1_grad = cache_params.params_dict["b1_grad"][self.layer_idx].reshape(B, NH, 1, HF)

            token_idx = inputs.pop('token_idx')
            # [B*nh,N=1,f], [B*nh,N=1,1] -> [BS,nh,N=1,f/1]
            inputs = tree_map(lambda x: x.reshape(B, NH, N, -1), inputs)
            XV, XK, XQ, ttt_lr = inputs['XV'], inputs['XK'], inputs['XQ'], inputs['ttt_lr']

            ttt_process_output = torch.empty_like(XV)
            grid = (B, NH, 1)
            mini_batch_size = 1

            if is_last_in_mini_batch:
                triton_ttt_linear_decode._decode_last_token_in_mini_batch_ker[grid](
                    W1, W1_grad, b1, b1_grad,
                    XV, XK, XQ,
                    self.ttt_norm_weight.data, self.ttt_norm_bias.data,
                    ttt_lr, token_idx, ttt_process_output,
                    W1.stride(0), W1.stride(1), W1.stride(2),
                    b1.stride(0), b1.stride(1), b1.stride(2),
                    XV.stride(0), XV.stride(1), XV.stride(2),
                    self.ttt_norm_weight.data.stride(1), self.ttt_norm_weight.data.stride(2),
                    ttt_lr.stride(0), ttt_lr.stride(1),
                    mini_batch_size, HF
                )
                W1_grad.zero_()
                b1_grad.zero_()
            else:
                triton_ttt_linear_decode._decode_token_ker[grid](
                    W1, W1_grad, b1, b1_grad,
                    XV, XK, XQ,
                    self.ttt_norm_weight.data, self.ttt_norm_bias.data,
                    ttt_lr, token_idx, ttt_process_output,
                    W1.stride(0), W1.stride(1), W1.stride(2),
                    b1.stride(0), b1.stride(1), b1.stride(2),
                    XV.stride(0), XV.stride(1), XV.stride(2),
                    self.ttt_norm_weight.data.stride(1), self.ttt_norm_weight.data.stride(2),
                    ttt_lr.stride(0), ttt_lr.stride(1),
                    mini_batch_size, HF
                )

            ttt_process_output = ttt_process_output.reshape(B_mul_NH, N, HF)

        return ttt_process_output

##########################################

class TTTMLPFast(TTTBase):

    def __init__(self, config: TTTConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, 4 * self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, 4 * self.head_dim))
        self.W2 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, 4 * self.head_dim, self.head_dim)))
        self.b2 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

    def TTT_process(
        self,
        inputs,
        is_prefill=False,
        is_last_in_mini_batch=False,
        cache_params=None
    ):

        B_mul_NH, N, HF = inputs['XV'].shape
        HF_prime = 4 * HF
        B = B_mul_NH // self.num_heads
        NH = self.num_heads

        if is_prefill:
            HF_prime = 4 * HF
            W1_init = cache_params.params_dict["W1_init"][self.layer_idx].reshape(B, NH, HF, HF_prime)
            b1_init = cache_params.params_dict["b1_init"][self.layer_idx].reshape(B, NH, 1, HF_prime)
            W2_init = cache_params.params_dict["W2_init"][self.layer_idx].reshape(B, NH, HF_prime, HF)
            b2_init = cache_params.params_dict["b2_init"][self.layer_idx].reshape(B, NH, 1, HF)

            mini_batch_size = self.mini_batch_size
            n_mini_batch = N // mini_batch_size
            token_idx = inputs.pop('token_idx')
            # [B*nh,N,f/1] -> [B,nh,nc,cs,f/1]
            inputs = tree_map(lambda x: x.reshape(B, NH, n_mini_batch, mini_batch_size, -1).contiguous(), inputs)
            # [B,nh,n_mini_batch,1,mini_batch_size]
            ttt_lr = inputs.pop('ttt_lr').transpose(-1, -2)
            # [1,1,1,mini_batch_size,1] * [B,nh,n_mini_batch,1,mini_batch_size] -> [B,nh,n_mini_batch,mini_batch_size,mini_batch_size]
            inputs['eta'] = token_idx[None, :] * ttt_lr

            XV, XK, XQ, eta = inputs['XV'], inputs['XK'], inputs['XQ'], inputs['eta']
            input_device = XV.device
            input_dtype = XV.dtype
            ttt_process_output = torch.empty_like(XV)

            ttt_norm_weight = self.ttt_norm_weight.data.squeeze(0).expand(-1, mini_batch_size, -1).contiguous()
            ttt_norm_bias = self.ttt_norm_bias.data.squeeze(0).expand(-1, mini_batch_size, -1).contiguous()
            b1_init = b1_init.expand(-1, -1, mini_batch_size, -1).contiguous()
            b2_init = b2_init.expand(-1, -1, mini_batch_size, -1).contiguous()
            make_last_b_matrix = torch.zeros(mini_batch_size, mini_batch_size, dtype=input_dtype, device=input_device)
            make_last_eta_1_matrix = torch.zeros(mini_batch_size, HF, dtype=input_dtype, device=input_device)
            make_last_eta_2_matrix = torch.zeros(mini_batch_size, HF_prime, dtype=input_dtype, device=input_device)
            make_last_b_matrix[:, -1] = 1.
            make_last_eta_1_matrix[-1, :] = 1.
            make_last_eta_2_matrix[-1, :] = 1.

            tk_ttt_mlp_prefill.ttt_mlp_prefill_fp16(
                W1_init, W2_init, b1_init, b2_init,
                ttt_norm_weight, ttt_norm_bias,
                make_last_b_matrix,
                make_last_eta_1_matrix, make_last_eta_2_matrix,
                XV, XK, XQ, eta,
                ttt_process_output
            )

            b1_init = b1_init[:, :, -1:, :].reshape(B_mul_NH, 1, -1)
            b2_init = b2_init[:, :, -1:, :].reshape(B_mul_NH, 1, -1)
            cache_params.params_dict["b1_init"][self.layer_idx].copy_(b1_init)
            cache_params.params_dict["b2_init"][self.layer_idx].copy_(b2_init)
            ttt_process_output = ttt_process_output.reshape(B_mul_NH, N, HF)

        else:
            W1 = cache_params.params_dict["W1_init"][self.layer_idx].reshape(B, NH, HF, HF_prime)
            b1 = cache_params.params_dict["b1_init"][self.layer_idx].reshape(B, NH, 1, HF_prime)
            W1_grad = cache_params.params_dict["W1_grad"][self.layer_idx].reshape(B, NH, HF, HF_prime)
            b1_grad = cache_params.params_dict["b1_grad"][self.layer_idx].reshape(B, NH, 1, HF_prime)

            W2 = cache_params.params_dict["W2_init"][self.layer_idx].reshape(B, NH, HF_prime, HF)
            b2 = cache_params.params_dict["b2_init"][self.layer_idx].reshape(B, NH, 1, HF)
            W2_grad = cache_params.params_dict["W2_grad"][self.layer_idx].reshape(B, NH, HF_prime, HF)
            b2_grad = cache_params.params_dict["b2_grad"][self.layer_idx].reshape(B, NH, 1, HF)

            token_idx = inputs.pop('token_idx')

            # [B*nh,N=1,f], [B*nh,N=1,1] -> [BS,nh,N=1,f/1]
            inputs = tree_map(lambda x: x.reshape(B, NH, N, -1), inputs)
            XV, XK, XQ, ttt_lr = inputs['XV'], inputs['XK'], inputs['XQ'], inputs['ttt_lr']

            ttt_process_output = torch.empty_like(XV)
            grid = (B, NH, 1)
            mini_batch_size = 1

            if is_last_in_mini_batch:
                triton_ttt_mlp_decode._decode_last_token_in_mini_batch_ker[grid](
                    W1, W1_grad, b1, b1_grad,
                    W2, W2_grad, b2, b2_grad,
                    XV, XK, XQ,
                    self.ttt_norm_weight.data, self.ttt_norm_bias.data,
                    ttt_lr, token_idx, ttt_process_output,
                    W1.stride(0), W1.stride(1), W1.stride(2),
                    b1.stride(0), b1.stride(1), b1.stride(2),
                    W2.stride(0), W2.stride(1), W2.stride(2),
                    b2.stride(0), b2.stride(1), b2.stride(2),
                    XV.stride(0), XV.stride(1), XV.stride(2),
                    self.ttt_norm_weight.data.stride(1), self.ttt_norm_weight.data.stride(2),
                    ttt_lr.stride(0), ttt_lr.stride(1),
                    mini_batch_size, HF, HF_prime
                )
                W1_grad.zero_()
                b1_grad.zero_()
                W2_grad.zero_()
                b2_grad.zero_()
            else:
                triton_ttt_mlp_decode._decode_token_ker[grid](
                    W1, W1_grad, b1, b1_grad,
                    W2, W2_grad, b2, b2_grad,
                    XV, XK, XQ,
                    self.ttt_norm_weight.data, self.ttt_norm_bias.data,
                    ttt_lr, token_idx, ttt_process_output,
                    W1.stride(0), W1.stride(1), W1.stride(2),
                    b1.stride(0), b1.stride(1), b1.stride(2),
                    W2.stride(0), W2.stride(1), W2.stride(2),
                    b2.stride(0), b2.stride(1), b2.stride(2),
                    XV.stride(0), XV.stride(1), XV.stride(2),
                    self.ttt_norm_weight.data.stride(1), self.ttt_norm_weight.data.stride(2),
                    ttt_lr.stride(0), ttt_lr.stride(1),
                    mini_batch_size, HF, HF_prime
                )

            ttt_process_output = ttt_process_output.reshape(B_mul_NH, N, HF)

        return ttt_process_output

##########################################


class Block(nn.Module):

    def __init__(self, config: TTTConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        self.fused_add_norm = config.fused_add_norm
        self.residual_in_fp32 = config.residual_in_fp32

        self.conv_before_ttt = config.conv_before_ttt
        if self.conv_before_ttt:
            self.conv_norm = RMSNorm(hidden_size=self.hidden_size, eps=config.rms_norm_eps)
            self.conv = Conv(layer_idx, config)

        self.seq_norm = RMSNorm(hidden_size=self.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(hidden_size=self.hidden_size, eps=config.rms_norm_eps)

        if config.seq_modeling_block == 'ttt-linear':
            self.seq_modeling_block = TTTLinear(config=config, layer_idx=layer_idx)
        elif config.seq_modeling_block == 'ttt-linear-fast':
            self.seq_modeling_block = TTTLinearFast(config=config, layer_idx=layer_idx)
        elif config.seq_modeling_block == 'ttt-mlp':
            self.seq_modeling_block = TTTMLP(config=config, layer_idx=layer_idx)
        elif config.seq_modeling_block == 'ttt-mlp-fast':
            self.seq_modeling_block = TTTMLPFast(config=config, layer_idx=layer_idx)
        else:
            raise NotImplementedError(f"Sequence Modeling Block {config.seq_modeling_block} Not Implemented!")

        self.mlp = SwiGluMLP(config)

        if config.use_compile:
            self.mlp_forward = torch.compile(self._mlp_forward)
        else:
            self.mlp_forward = self._mlp_forward

    def _mlp_forward(self, hidden_states: torch.Tensor):
        hidden_states = self.mlp(hidden_states)
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[TTTCache] = None,
        is_prefill: Optional[bool] = None,
        is_last_in_mini_batch: Optional[bool] = None,
    ):
        if self.conv_before_ttt:
            if not self.fused_add_norm:
                residual = (hidden_states + residual) if residual is not None else hidden_states
                hidden_states = self.conv_norm(residual.to(dtype=self.conv_norm.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                fused_add_norm_fn = rms_norm_fn
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.conv_norm.weight,
                    self.conv_norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.conv_norm.eps,
                )
            # Conv
            hidden_states = self.conv(
                hidden_states,
                is_prefill=is_prefill,
                cache_params=cache_params
            )

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.seq_norm(residual.to(dtype=self.seq_norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.seq_norm.weight,
                self.seq_norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.seq_norm.eps,
            )
        # TTT
        hidden_states = self.seq_modeling_block(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_params=cache_params,
            is_prefill=is_prefill,
            is_last_in_mini_batch=is_last_in_mini_batch,
        )

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.ffn_norm(residual.to(dtype=self.ffn_norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.ffn_norm.weight,
                self.ffn_norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.ffn_norm.eps,
            )
        # FFN
        hidden_states = self.mlp_forward(hidden_states)

        return hidden_states, residual


class TTTPreTrainedModel(nn.Module, GenerationMixin):
    
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


@dataclass
class TTTOutput:
    last_hidden_state: Optional[torch.FloatTensor] = None
    cache_params: Optional[TTTCache] = None


@dataclass
class TTTCausalLMOutput:
    logits: Optional[torch.FloatTensor] = None
    cache_params: Optional[TTTCache] = None


class TTTModel(TTTPreTrainedModel):

    def __init__(self, config: TTTConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.mini_batch_size = config.mini_batch_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Block(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.fused_add_norm = config.fused_add_norm
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[TTTCache] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        is_prefill: Optional[bool] = None,
        is_last_in_mini_batch: Optional[bool] = None,
    ) -> Union[Tuple, TTTOutput]:

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        seqlen_offset = 0
        if cache_params is not None:
            seqlen_offset = cache_params.seqlen_offset
        position_ids = torch.arange(
            seqlen_offset, seqlen_offset+ inputs_embeds.shape[1], dtype=torch.long, device=inputs_embeds.device
        ).unsqueeze(0)

        hidden_states = inputs_embeds
        residual = None

        for decoder_layer in self.layers:
            hidden_states, residual = decoder_layer(
                hidden_states,
                residual=residual,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_params=cache_params,
                is_prefill=is_prefill,
                is_last_in_mini_batch=is_last_in_mini_batch,
            )

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                eps=self.norm.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return TTTOutput(
            last_hidden_state=hidden_states,
            cache_params=cache_params,
        )


class TTTForCausalLM(TTTPreTrainedModel):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = TTTModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.tie_weights()

        self.apply(self._init_weights)

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[TTTCache] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        is_prefill: Optional[bool] = None,
        is_last_in_mini_batch: Optional[bool] = None,
        *,
        num_last_tokens=0,
    ) -> Union[Tuple, TTTCausalLMOutput]:

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            cache_params=cache_params,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
            is_prefill=is_prefill,
            is_last_in_mini_batch=is_last_in_mini_batch,
        )

        if num_last_tokens > 0:
            hidden_states = outputs.last_hidden_state[:, -num_last_tokens:]
        else:
            hidden_states = outputs.last_hidden_state

        logits = self.lm_head(hidden_states)

        return TTTCausalLMOutput(
            logits=logits,
            cache_params=outputs.cache_params,
        )
