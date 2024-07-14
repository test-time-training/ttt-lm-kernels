#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

extern void  ttt_mlp_prefill_fp16(
        torch::Tensor W1,
        torch::Tensor W2,
        torch::Tensor b1,
        torch::Tensor b2,
        torch::Tensor ln_weight,
        torch::Tensor ln_bias,
        torch::Tensor make_last_b_matrix,
        torch::Tensor make_last_eta_1_matrix,
        torch::Tensor make_last_eta_2_matrix,
        torch::Tensor XV,
        torch::Tensor XK,
        torch::Tensor XQ,
        torch::Tensor Eta,
        torch::Tensor Output,
        cudaStream_t stream
);


extern void  ttt_mlp_prefill_fp16_ref(
        torch::Tensor W1,
        torch::Tensor W2,
        torch::Tensor b1,
        torch::Tensor b2,
        torch::Tensor ln_weight,
        torch::Tensor ln_bias,
        torch::Tensor make_last_b_matrix,
        torch::Tensor make_last_eta_1_matrix,
        torch::Tensor make_last_eta_2_matrix,
        torch::Tensor XV,
        torch::Tensor XK,
        torch::Tensor XQ,
        torch::Tensor Eta,
        torch::Tensor Output
) {
    auto stream = at::cuda::getCurrentCUDAStream();
    ttt_mlp_prefill_fp16(
            W1,
            W2,
            b1,
            b2,
            ln_weight,
            ln_bias,
            make_last_b_matrix,
            make_last_eta_1_matrix,
            make_last_eta_2_matrix,
            XV,
            XK,
            XQ,
            Eta,
            Output,
            stream
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "TTT-MLP prefill kernel with ThunderKittens";
    m.def("ttt_mlp_prefill_fp16", &ttt_mlp_prefill_fp16_ref);
}
