import triton
import triton.language as tl

from ttt.triton_kernel.activations import gelu_tl

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
    key=['F'],
)
@triton.jit
def _fuse_gate_ln_kernel(__XGate, __X, __Out,
                         __ln_weight, __ln_bias,
                         stride_x_batch,
                         F: tl.constexpr):
    batch = tl.program_id(0)

    rf = tl.arange(0, F)

    O_dtype = __Out.type.element_ty

    x_block_offset = batch * stride_x_batch

    x_inner_offset = rf[None, :]
    ln_inner_offset = rf[None, :]

    x_offset = x_block_offset + x_inner_offset
    ln_offset = ln_inner_offset

    _XGate = __XGate + x_offset
    _X = __X + x_offset
    _Out = __Out + x_offset
    _ln_weight = __ln_weight + ln_offset
    _ln_bias = __ln_bias + ln_offset

    XGate = tl.load(_XGate)
    X = tl.load(_X)
    ln_weight = tl.load(_ln_weight)
    ln_bias = tl.load(_ln_bias)

    ## LN(X)
    mu = (tl.sum(X, 1) / F).to(O_dtype)
    var = (tl.sum((X - mu) * (X - mu), 1) / F).to(O_dtype)
    std = tl.sqrt(var + 1e-6).to(O_dtype)
    X_hat = ((X - mu) / std).to(O_dtype)  # [1,f]
    LN_X = ln_weight * X_hat + ln_bias  # [1,f] * [K=1,f] + [1,f]

    ## gelu(XGate)
    XGate_activated = gelu_tl(XGate).to(O_dtype)

    output = XGate_activated * LN_X

    tl.store(_Out, output.to(O_dtype))
