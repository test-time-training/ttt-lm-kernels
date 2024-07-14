#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

extern void  ttt_linear_prefill_fp16(
        torch::Tensor W1, torch::Tensor b1,
        torch::Tensor ln_weight, torch::Tensor ln_bias,
        torch::Tensor make_last_b_matrix,
        torch::Tensor make_last_eta_1_matrix,
        torch::Tensor XV, torch::Tensor XK, torch::Tensor XQ, torch::Tensor Eta,
        torch::Tensor Out,
        cudaStream_t stream
);

extern void  ttt_linear_prefill_fp16_ref(
        torch::Tensor W1, torch::Tensor b1,
        torch::Tensor ln_weight, torch::Tensor ln_bias,
        torch::Tensor make_last_b_matrix,
        torch::Tensor make_last_eta_1_matrix,
        torch::Tensor XV, torch::Tensor XK, torch::Tensor XQ, torch::Tensor Eta,
        torch::Tensor Out
)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    ttt_linear_prefill_fp16(
            W1, b1, ln_weight, ln_bias,
            make_last_b_matrix,
            make_last_eta_1_matrix,
            XV, XK, XQ, Eta, Out, stream
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "TTT-Linear prefill kernel with ThunderKittens";
    m.def("ttt_linear_prefill_fp16", &ttt_linear_prefill_fp16_ref);
}
