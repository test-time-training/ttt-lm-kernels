import triton
import triton.language as tl

from ttt.triton_kernel.activations import gelu_tl, diff_gelu_tl


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=1, num_warps=8),
        triton.Config({}, num_stages=1, num_warps=4),
        triton.Config({}, num_stages=1, num_warps=2),
        triton.Config({}, num_stages=2, num_warps=8),
        triton.Config({}, num_stages=2, num_warps=4),
        triton.Config({}, num_stages=2, num_warps=2),
        triton.Config({}, num_stages=3, num_warps=8),
        triton.Config({}, num_stages=3, num_warps=4),
        triton.Config({}, num_stages=3, num_warps=2),
        triton.Config({}, num_stages=4, num_warps=8),
        triton.Config({}, num_stages=4, num_warps=4),
        triton.Config({}, num_stages=4, num_warps=2),
    ],
    key=['HF', 'HF_prime'],
    restore_value=['__W1', '__b1', '__W1_grad', '__b1_grad',
                   '__W2', '__b2', '__W2_grad', '__b2_grad']
)
@triton.jit
def _decode_token_ker(
    __W1, __W1_grad, __b1, __b1_grad,
    __W2, __W2_grad, __b2, __b2_grad,
    __XV, __XK, __XQ,
    __ln_weight, __ln_bias,
    __ilr_gated, __token_idx, __Out,
    stride_w1_batch, stride_w1_head, stride_w1_fin,
    stride_b1_batch, stride_b1_head, stride_b1_f,
    stride_w2_batch, stride_w2_head, stride_w2_fin,
    stride_b2_batch, stride_b2_head, stride_b2_f,
    stride_x_batch, stride_x_head, stride_x_n,
    stride_ln_head, stride_ln_f,
    stride_ilr_batch, stride_ilr_head,
    CS: tl.constexpr, HF: tl.constexpr, HF_prime: tl.constexpr
):
    batch = tl.program_id(0)
    head = tl.program_id(1)

    rc = tl.arange(0, CS)
    rf = tl.arange(0, HF)
    rf_prime = tl.arange(0, HF_prime)

    W_dtype = __W1.type.element_ty
    O_dtype = __Out.type.element_ty

    x_block_offset = batch * stride_x_batch + head * stride_x_head
    w1_block_offset = batch * stride_w1_batch + head * stride_w1_head
    b1_block_offset = batch * stride_b1_batch + head * stride_b1_head
    w2_block_offset = batch * stride_w2_batch + head * stride_w2_head
    b2_block_offset = batch * stride_b2_batch + head * stride_b2_head
    ln_block_offset = head * stride_ln_head
    ilr_block_offset = batch * stride_ilr_batch + head * stride_ilr_head

    x_inner_offset = rc[:, None] * stride_x_n + rf[None, :]
    w1_inner_offset = rf[:, None] * stride_w1_fin + rf_prime[None, :]
    b1_inner_offset = rc[:, None] * stride_b1_f + rf_prime[None, :]
    w2_inner_offset = rf_prime[:, None] * stride_w2_fin + rf[None, :]
    b2_inner_offset = rc[:, None] * stride_b2_f + rf[None, :]
    ln_inner_offset = rc[:, None] * stride_ln_f + rf[None, :]

    x_offset = x_block_offset + x_inner_offset
    w1_offset = w1_block_offset + w1_inner_offset
    b1_offset = b1_block_offset + b1_inner_offset
    w2_offset = w2_block_offset + w2_inner_offset
    b2_offset = b2_block_offset + b2_inner_offset
    ln_offset = ln_block_offset + ln_inner_offset
    ilr_offset = ilr_block_offset

    _XV = __XV + x_offset
    _XK = __XK + x_offset
    _XQ = __XQ + x_offset
    _Out = __Out + x_offset
    _W1 = __W1 + w1_offset
    _W1_grad = __W1_grad + w1_offset
    _b1 = __b1 + b1_offset
    _b1_grad = __b1_grad + b1_offset
    _W2 = __W2 + w2_offset
    _W2_grad = __W2_grad + w2_offset
    _b2 = __b2 + b2_offset
    _b2_grad = __b2_grad + b2_offset
    _ln_weight = __ln_weight + ln_offset
    _ln_bias = __ln_bias + ln_offset
    _ilr_gated = __ilr_gated + ilr_offset
    _token_idx = __token_idx

    XV = tl.load(_XV)
    XK = tl.load(_XK)
    XQ = tl.load(_XQ)
    token_idx = tl.load(_token_idx)
    ilr_gated = tl.load(_ilr_gated)
    W1 = tl.load(_W1)
    W1_grad = tl.load(_W1_grad)
    b1 = tl.load(_b1)
    b1_grad = tl.load(_b1_grad)
    W2 = tl.load(_W2)
    W2_grad = tl.load(_W2_grad)
    b2 = tl.load(_b2)
    b2_grad = tl.load(_b2_grad)
    ln_weight = tl.load(_ln_weight)
    ln_bias = tl.load(_ln_bias)

    Z1 = tl.sum(tl.trans(XK) * W1, axis=0)[None, :] + b1  # [1,f] @ [f,4f] + [1,4f]
    X2 = gelu_tl(Z1).to(O_dtype)
    Z2 = tl.sum(tl.trans(X2) * W2, axis=0)[None, :] + b2  # [1,4f] @ [4f,f] + [1,f]

    l2_target = XV - XK

    mu = (tl.sum(Z2, 1) / HF).to(O_dtype)
    var = (tl.sum((Z2 - mu) * (Z2 - mu), 1) / HF).to(O_dtype)
    std = tl.sqrt(var + 1e-6).to(O_dtype)
    Z2_hat = ((Z2 - mu) / std).to(O_dtype)  # [1,f]
    LN_out = ln_weight * Z2_hat + ln_bias  # [1,f] * [K=1,f] + [1,f]
    dl_dLN_out = LN_out - l2_target  # [1,f]
    dl_dZ2_hat = dl_dLN_out * ln_weight  # [1,f]

    dl_dZ2_term_1 = HF * dl_dZ2_hat
    dl_dZ2_term_2 = tl.sum(dl_dZ2_hat, 1)
    dl_dZ2_term_3 = Z2_hat * tl.sum(dl_dZ2_hat * Z2_hat, 1)
    dl_dZ2_sum = dl_dZ2_term_1 - dl_dZ2_term_2 - dl_dZ2_term_3
    dl_dZ2 = (dl_dZ2_sum / (std * HF)).to(O_dtype)

    dl_dZ1 = tl.sum(tl.trans(dl_dZ2) * tl.trans(W2), axis=0)[None, :] * diff_gelu_tl(Z1).to(O_dtype)  # [1,f] @ [4f,f].t

    ilr_mul_dl_dZ2 = ilr_gated * dl_dZ2  # [K=1,1] * [K=1,f]
    ilr_mul_dl_dZ1 = ilr_gated * dl_dZ1  # [K=1,1] * [K=1,f]

    ##
    W1_grad += tl.trans(XK) * ilr_mul_dl_dZ1
    b1_grad += ilr_mul_dl_dZ1
    tl.store(_W1_grad, W1_grad.to(W_dtype))
    tl.store(_b1_grad, b1_grad.to(W_dtype))
    W1_bar = W1 - token_idx * W1_grad
    b1_bar = b1 - token_idx * b1_grad
    Z1_bar = tl.sum(tl.trans(XQ) * W1_bar, axis=0)[None, :] + b1_bar

    X2_bar = gelu_tl(Z1_bar).to(O_dtype)

    ##
    W2_grad += tl.trans(X2) * ilr_mul_dl_dZ2
    b2_grad += ilr_mul_dl_dZ2
    tl.store(_W2_grad, W2_grad.to(W_dtype))
    tl.store(_b2_grad, b2_grad.to(W_dtype))
    W2_bar = W2 - token_idx * W2_grad
    b2_bar = b2 - token_idx * b2_grad
    Z2_bar = tl.sum(tl.trans(X2_bar) * W2_bar, axis=0)[None, :] + b2_bar

    ## residual + postln
    mu_bar = (tl.sum(Z2_bar, 1) / HF).to(O_dtype)
    var_bar = (tl.sum((Z2_bar - mu_bar) * (Z2_bar - mu_bar), 1) / HF).to(O_dtype)
    std_bar = tl.sqrt(var_bar + 1e-6).to(O_dtype)
    Z2_bar_hat = ((Z2_bar - mu_bar) / std_bar).to(O_dtype)  # [1,f]
    LN_out_bar = ln_weight * Z2_bar_hat + ln_bias  # [1,f] * [K=1,f] + [1,f]
    Z2_bar = XQ + LN_out_bar

    tl.store(_Out, Z2_bar.to(O_dtype))

@triton.autotune(
    configs=[
        triton.Config({}, num_stages=1, num_warps=8),
        triton.Config({}, num_stages=1, num_warps=4),
        triton.Config({}, num_stages=1, num_warps=2),
        triton.Config({}, num_stages=2, num_warps=8),
        triton.Config({}, num_stages=2, num_warps=4),
        triton.Config({}, num_stages=2, num_warps=2),
        triton.Config({}, num_stages=3, num_warps=8),
        triton.Config({}, num_stages=3, num_warps=4),
        triton.Config({}, num_stages=3, num_warps=2),
        triton.Config({}, num_stages=4, num_warps=8),
        triton.Config({}, num_stages=4, num_warps=4),
        triton.Config({}, num_stages=4, num_warps=2),
    ],
    key=['HF', 'HF_prime'],
    restore_value=['__W1', '__b1', '__W1_grad', '__b1_grad',
                   '__W2', '__b2', '__W2_grad', '__b2_grad']
)
@triton.jit
def _decode_last_token_in_mini_batch_ker(
    __W1, __W1_grad, __b1, __b1_grad,
    __W2, __W2_grad, __b2, __b2_grad,
    __XV, __XK, __XQ,
    __ln_weight, __ln_bias,
    __ilr_gated, __token_idx, __Out,
    stride_w1_batch, stride_w1_head, stride_w1_fin,
    stride_b1_batch, stride_b1_head, stride_b1_f,
    stride_w2_batch, stride_w2_head, stride_w2_fin,
    stride_b2_batch, stride_b2_head, stride_b2_f,
    stride_x_batch, stride_x_head, stride_x_n,
    stride_ln_head, stride_ln_f,
    stride_ilr_batch, stride_ilr_head,
    CS: tl.constexpr, HF: tl.constexpr, HF_prime: tl.constexpr
):
    batch = tl.program_id(0)
    head = tl.program_id(1)

    rc = tl.arange(0, CS)
    rf = tl.arange(0, HF)
    rf_prime = tl.arange(0, HF_prime)

    W_dtype = __W1.type.element_ty
    O_dtype = __Out.type.element_ty

    x_block_offset = batch * stride_x_batch + head * stride_x_head
    w1_block_offset = batch * stride_w1_batch + head * stride_w1_head
    b1_block_offset = batch * stride_b1_batch + head * stride_b1_head
    w2_block_offset = batch * stride_w2_batch + head * stride_w2_head
    b2_block_offset = batch * stride_b2_batch + head * stride_b2_head
    ln_block_offset = head * stride_ln_head
    ilr_block_offset = batch * stride_ilr_batch + head * stride_ilr_head

    x_inner_offset = rc[:, None] * stride_x_n + rf[None, :]
    w1_inner_offset = rf[:, None] * stride_w1_fin + rf_prime[None, :]
    b1_inner_offset = rc[:, None] * stride_b1_f + rf_prime[None, :]
    w2_inner_offset = rf_prime[:, None] * stride_w2_fin + rf[None, :]
    b2_inner_offset = rc[:, None] * stride_b2_f + rf[None, :]
    ln_inner_offset = rc[:, None] * stride_ln_f + rf[None, :]

    x_offset = x_block_offset + x_inner_offset
    w1_offset = w1_block_offset + w1_inner_offset
    b1_offset = b1_block_offset + b1_inner_offset
    w2_offset = w2_block_offset + w2_inner_offset
    b2_offset = b2_block_offset + b2_inner_offset
    ln_offset = ln_block_offset + ln_inner_offset
    ilr_offset = ilr_block_offset

    _XV = __XV + x_offset
    _XK = __XK + x_offset
    _XQ = __XQ + x_offset
    _Out = __Out + x_offset
    _W1 = __W1 + w1_offset
    _W1_grad = __W1_grad + w1_offset
    _b1 = __b1 + b1_offset
    _b1_grad = __b1_grad + b1_offset
    _W2 = __W2 + w2_offset
    _W2_grad = __W2_grad + w2_offset
    _b2 = __b2 + b2_offset
    _b2_grad = __b2_grad + b2_offset
    _ln_weight = __ln_weight + ln_offset
    _ln_bias = __ln_bias + ln_offset
    _ilr_gated = __ilr_gated + ilr_offset
    _token_idx = __token_idx

    XV = tl.load(_XV)
    XK = tl.load(_XK)
    XQ = tl.load(_XQ)
    token_idx = tl.load(_token_idx)
    ilr_gated = tl.load(_ilr_gated)
    W1 = tl.load(_W1)
    W1_grad = tl.load(_W1_grad)
    b1 = tl.load(_b1)
    b1_grad = tl.load(_b1_grad)
    W2 = tl.load(_W2)
    W2_grad = tl.load(_W2_grad)
    b2 = tl.load(_b2)
    b2_grad = tl.load(_b2_grad)
    ln_weight = tl.load(_ln_weight)
    ln_bias = tl.load(_ln_bias)

    Z1 = tl.sum(tl.trans(XK) * W1, axis=0)[None, :] + b1  # [1,f] @ [f,4f] + [1,4f]
    X2 = gelu_tl(Z1).to(O_dtype)
    Z2 = tl.sum(tl.trans(X2) * W2, axis=0)[None, :] + b2  # [1,4f] @ [4f,f] + [1,f]

    l2_target = XV - XK

    mu = (tl.sum(Z2, 1) / HF).to(O_dtype)
    var = (tl.sum((Z2 - mu) * (Z2 - mu), 1) / HF).to(O_dtype)
    std = tl.sqrt(var + 1e-6).to(O_dtype)
    Z2_hat = ((Z2 - mu) / std).to(O_dtype)  # [1,f]
    LN_out = ln_weight * Z2_hat + ln_bias  # [1,f] * [K=1,f] + [1,f]
    dl_dLN_out = LN_out - l2_target  # [1,f]
    dl_dZ2_hat = dl_dLN_out * ln_weight  # [1,f]

    dl_dZ2_term_1 = HF * dl_dZ2_hat
    dl_dZ2_term_2 = tl.sum(dl_dZ2_hat, 1)
    dl_dZ2_term_3 = Z2_hat * tl.sum(dl_dZ2_hat * Z2_hat, 1)
    dl_dZ2_sum = dl_dZ2_term_1 - dl_dZ2_term_2 - dl_dZ2_term_3
    dl_dZ2 = (dl_dZ2_sum / (std * HF)).to(O_dtype)

    dl_dZ1 = tl.sum(tl.trans(dl_dZ2) * tl.trans(W2), axis=0)[None, :] * diff_gelu_tl(Z1).to(O_dtype)  # [1,f] @ [4f,f].t

    ilr_mul_dl_dZ2 = ilr_gated * dl_dZ2  # [K=1,1] * [K=1,f]
    ilr_mul_dl_dZ1 = ilr_gated * dl_dZ1  # [K=1,1] * [K=1,f]

    ##
    W1_grad += tl.trans(XK) * ilr_mul_dl_dZ1
    b1_grad += ilr_mul_dl_dZ1
    W1_bar = W1 - token_idx * W1_grad
    b1_bar = b1 - token_idx * b1_grad
    tl.store(_W1, W1_bar.to(W_dtype))
    tl.store(_b1, b1_bar.to(W_dtype))
    Z1_bar = tl.sum(tl.trans(XQ) * W1_bar, axis=0)[None, :] + b1_bar

    X2_bar = gelu_tl(Z1_bar).to(O_dtype)

    ##
    W2_grad += tl.trans(X2) * ilr_mul_dl_dZ2
    b2_grad += ilr_mul_dl_dZ2
    W2_bar = W2 - token_idx * W2_grad
    b2_bar = b2 - token_idx * b2_grad
    tl.store(_W2, W2_bar.to(W_dtype))
    tl.store(_b2, b2_bar.to(W_dtype))
    Z2_bar = tl.sum(tl.trans(X2_bar) * W2_bar, axis=0)[None, :] + b2_bar

    ## residual + postln
    mu_bar = (tl.sum(Z2_bar, 1) / HF).to(O_dtype)
    var_bar = (tl.sum((Z2_bar - mu_bar) * (Z2_bar - mu_bar), 1) / HF).to(O_dtype)
    std_bar = tl.sqrt(var_bar + 1e-6).to(O_dtype)
    Z2_bar_hat = ((Z2_bar - mu_bar) / std_bar).to(O_dtype)  # [1,f]
    LN_out_bar = ln_weight * Z2_bar_hat + ln_bias  # [1,f] * [K=1,f] + [1,f]
    Z2_bar = XQ + LN_out_bar

    tl.store(_Out, Z2_bar.to(O_dtype))
