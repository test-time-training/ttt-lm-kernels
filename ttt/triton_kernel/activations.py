import triton
import triton.language as tl

@triton.jit
def tanh_tl(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1

# from xformers impl.
@triton.jit
def gelu_tl(x):
    return 0.5 * x * (1 + tanh_tl(0.79788456 * (x + 0.044715 * x * x * x)))

@triton.jit
def diff_gelu_tl(x):
    tanh_out = tanh_tl(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff
