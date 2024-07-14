import triton
import triton.language as tl

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
    key=['HF'],
    restore_value=['__W1', '__b1', '__W1_grad', '__b1_grad']
)
@triton.jit
def _decode_token_ker(
    __W1, __W1_grad, __b1, __b1_grad,
    __XV, __XK, __XQ,
    __ln_weight, __ln_bias,
    __ilr_gated, __token_idx, __Out,
    stride_w_batch, stride_w_head, stride_w_fin,
    stride_b_batch, stride_b_head, stride_b_f,
    stride_x_batch, stride_x_head, stride_x_n,
    stride_ln_head, stride_ln_f,
    stride_ilr_batch, stride_ilr_head,
    CS: tl.constexpr, HF: tl.constexpr
):
    batch = tl.program_id(0)
    head = tl.program_id(1)

    rc = tl.arange(0, CS)
    rf = tl.arange(0, HF)

    W_dtype = __W1.type.element_ty
    O_dtype = __Out.type.element_ty

    x_block_offset = batch * stride_x_batch + head * stride_x_head
    w_block_offset = batch * stride_w_batch + head * stride_w_head
    b_block_offset = batch * stride_b_batch + head * stride_b_head
    ln_block_offset = head * stride_ln_head
    ilr_block_offset = batch * stride_ilr_batch + head * stride_ilr_head

    x_inner_offset = rc[:, None] * stride_x_n + rf[None, :]
    w_inner_offset = rf[:, None] * stride_w_fin + rf[None, :]
    b_inner_offset = rc[:, None] * stride_b_f + rf[None, :]
    ln_inner_offset = rc[:, None] * stride_ln_f + rf[None, :]

    x_offset = x_block_offset + x_inner_offset
    w_offset = w_block_offset + w_inner_offset
    b_offset = b_block_offset + b_inner_offset
    ln_offset = ln_block_offset + ln_inner_offset
    ilr_offset = ilr_block_offset

    _XV = __XV + x_offset
    _XK = __XK + x_offset
    _XQ = __XQ + x_offset
    _Out = __Out + x_offset
    _W1 = __W1 + w_offset
    _W1_grad = __W1_grad + w_offset
    _b1 = __b1 + b_offset
    _b1_grad = __b1_grad + b_offset
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
    ln_weight = tl.load(_ln_weight)
    ln_bias = tl.load(_ln_bias)

    Z1 = tl.sum(tl.trans(XK) * W1, axis=0)[None, :] + b1  # [1,f] @ [f,f] + [1,f]: fp16
    l2_target = XV - XK

    mu = (tl.sum(Z1, 1) / HF).to(O_dtype)  # fp16 -> fp32 after division, need cast back
    var = (tl.sum((Z1 - mu) * (Z1 - mu), 1) / HF).to(O_dtype)
    std = tl.sqrt(var + 1e-6).to(O_dtype)  # fp16 -> fp32 after adding 1e-6, sqrt requires input fp32/fp64
    Z1_hat = ((Z1 - mu) / std).to(O_dtype)  # [1,f]: fp16 div fp16 -> fp32

    # Scale and shift
    LN_out = ln_weight * Z1_hat + ln_bias  # [1,f] * [K=1,f] + [1,f]

    dl_dLN_out = LN_out - l2_target  # [1,f]

    dl_dZ1_hat = dl_dLN_out * ln_weight  # [1,f]

    dl_dZ1_term_1 = HF * dl_dZ1_hat  # int * fp16 -> fp16
    dl_dZ1_term_2 = tl.sum(dl_dZ1_hat, 1)
    dl_dZ1_term_3 = Z1_hat * tl.sum(dl_dZ1_hat * Z1_hat, 1)
    dl_dZ1_sum = dl_dZ1_term_1 - dl_dZ1_term_2 - dl_dZ1_term_3
    dl_dZ1 = (dl_dZ1_sum / (std * HF)).to(O_dtype)

    ilr_mul_dl_dZ1 = ilr_gated * dl_dZ1  # [K=1,1] * [K=1,f]

    ##
    W1_grad += tl.trans(XK) * ilr_mul_dl_dZ1
    b1_grad += ilr_mul_dl_dZ1

    tl.store(_W1_grad, W1_grad.to(W_dtype))
    tl.store(_b1_grad, b1_grad.to(W_dtype))

    W1_bar = W1 - token_idx * W1_grad
    b1_bar = b1 - token_idx * b1_grad

    Z1_bar = tl.sum(tl.trans(XQ) * W1_bar, axis=0)[None, :] + b1_bar

    ## residual + postln
    mu_bar = (tl.sum(Z1_bar, 1) / HF).to(O_dtype)
    var_bar = (tl.sum((Z1_bar - mu_bar) * (Z1_bar - mu_bar), 1) / HF).to(O_dtype)
    std_bar = tl.sqrt(var_bar + 1e-6).to(O_dtype)
    Z1_bar_hat = ((Z1_bar - mu_bar) / std_bar).to(O_dtype)  # [1,f]
    LN_out_bar = ln_weight * Z1_bar_hat + ln_bias  # [1,f] * [K=1,f] + [1,f]
    Z1_bar = XQ + LN_out_bar

    tl.store(_Out, Z1_bar.to(O_dtype))

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
    key=['HF'],
    restore_value=['__W1', '__b1', '__W1_grad', '__b1_grad']
)
@triton.jit
def _decode_last_token_in_mini_batch_ker(
    __W1, __W1_grad, __b1, __b1_grad,
    __XV, __XK, __XQ,
    __ln_weight, __ln_bias,
    __ilr_gated, __token_idx, __Out,
    stride_w_batch, stride_w_head, stride_w_fin,
    stride_b_batch, stride_b_head, stride_b_f,
    stride_x_batch, stride_x_head, stride_x_n,
    stride_ln_head, stride_ln_f,
    stride_ilr_batch, stride_ilr_head,
    CS: tl.constexpr, HF: tl.constexpr
):
    batch = tl.program_id(0)
    head = tl.program_id(1)

    rc = tl.arange(0, CS)
    rf = tl.arange(0, HF)

    W_dtype = __W1.type.element_ty
    O_dtype = __Out.type.element_ty

    x_block_offset = batch * stride_x_batch + head * stride_x_head
    w_block_offset = batch * stride_w_batch + head * stride_w_head
    b_block_offset = batch * stride_b_batch + head * stride_b_head
    ln_block_offset = head * stride_ln_head
    ilr_block_offset = batch * stride_ilr_batch + head * stride_ilr_head

    x_inner_offset = rc[:, None] * stride_x_n + rf[None, :]
    w_inner_offset = rf[:, None] * stride_w_fin + rf[None, :]
    b_inner_offset = rc[:, None] * stride_b_f + rf[None, :]
    ln_inner_offset = rc[:, None] * stride_ln_f + rf[None, :]

    x_offset = x_block_offset + x_inner_offset
    w_offset = w_block_offset + w_inner_offset
    b_offset = b_block_offset + b_inner_offset
    ln_offset = ln_block_offset + ln_inner_offset
    ilr_offset = ilr_block_offset

    _XV = __XV + x_offset
    _XK = __XK + x_offset
    _XQ = __XQ + x_offset
    _Out = __Out + x_offset
    _W1 = __W1 + w_offset
    _W1_grad = __W1_grad + w_offset
    _b1 = __b1 + b_offset
    _b1_grad = __b1_grad + b_offset
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
    ln_weight = tl.load(_ln_weight)
    ln_bias = tl.load(_ln_bias)

    Z1 = tl.sum(tl.trans(XK) * W1, axis=0)[None, :] + b1  # [1,f] @ [f,f] + [1,f]: fp16
    l2_target = XV - XK

    mu = (tl.sum(Z1, 1) / HF).to(O_dtype)  # fp16 -> fp32 after division, need cast back
    var = (tl.sum((Z1 - mu) * (Z1 - mu), 1) / HF).to(O_dtype)
    std = tl.sqrt(var + 1e-6).to(O_dtype)  # fp16 -> fp32 after adding 1e-6, sqrt requires input fp32/fp64
    Z1_hat = ((Z1 - mu) / std).to(O_dtype)  # [1,f]: fp16 div fp16 -> fp32

    # Scale and shift
    LN_out = ln_weight * Z1_hat + ln_bias  # [1,f] * [K=1,f] + [1,f]

    dl_dLN_out = LN_out - l2_target  # [1,f]

    dl_dZ1_hat = dl_dLN_out * ln_weight  # [1,f]

    dl_dZ1_term_1 = HF * dl_dZ1_hat  # int * fp16 -> fp16
    dl_dZ1_term_2 = tl.sum(dl_dZ1_hat, 1)
    dl_dZ1_term_3 = Z1_hat * tl.sum(dl_dZ1_hat * Z1_hat, 1)
    dl_dZ1_sum = dl_dZ1_term_1 - dl_dZ1_term_2 - dl_dZ1_term_3
    dl_dZ1 = (dl_dZ1_sum / (std * HF)).to(O_dtype)

    ilr_mul_dl_dZ1 = ilr_gated * dl_dZ1  # [K=1,1] * [K=1,f]

    ##
    W1_grad += tl.trans(XK) * ilr_mul_dl_dZ1
    b1_grad += ilr_mul_dl_dZ1

    W1_bar = W1 - token_idx * W1_grad
    b1_bar = b1 - token_idx * b1_grad

    Z1_bar = tl.sum(tl.trans(XQ) * W1_bar, axis=0)[None, :] + b1_bar

    tl.store(_W1, W1_bar.to(W_dtype))
    tl.store(_b1, b1_bar.to(W_dtype))

    ## residual + postln
    mu_bar = (tl.sum(Z1_bar, 1) / HF).to(O_dtype)
    var_bar = (tl.sum((Z1_bar - mu_bar) * (Z1_bar - mu_bar), 1) / HF).to(O_dtype)
    std_bar = tl.sqrt(var_bar + 1e-6).to(O_dtype)
    Z1_bar_hat = ((Z1_bar - mu_bar) / std_bar).to(O_dtype)  # [1,f]
    LN_out_bar = ln_weight * Z1_bar_hat + ln_bias  # [1,f] * [K=1,f] + [1,f]
    Z1_bar = XQ + LN_out_bar

    tl.store(_Out, Z1_bar.to(O_dtype))
