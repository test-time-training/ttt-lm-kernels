#include <iostream>
#include <string>
#include <math.h>
#include <assert.h>
#include <string>
#include <cuda_runtime_api.h>
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

# include "../../src/kittens.cuh"
# include "../LN/tk_ln.cuh"

// **** ASYn_mini_batch In_mini_batchLUDE *****
#include <cuda/pipeline>
#include <cooperative_groups.h>

#define X_STRIDE 1024     // 16 * 64
#define W_STRIDE 4096     // 64 * 64
#define b_STRIDE 64       // 64
#define Eta_STRIDE 256    // 16 * 16
#define SMEM_BLOCK (3 * X_STRIDE + 2 * Eta_STRIDE) * 2  // bytes: XV/XK/XQ/Eta

using namespace kittens;


template <typename H, typename T>
__global__
void ttt_linear_prefill_fp16_ker(
        const int NH, const int n_mini_batch, const int mini_batch_size, const int HF,
        T* __W1, T* __b1,
        const T* __ln_weight, const T* __ln_bias,
        const T* __make_last_b_matrix, const T* __make_last_eta_1_matrix,
        const T* __XV, const T* __XK, const T* __XQ, const T* __Eta,
        T* __Output
) {
    H *_W1       = reinterpret_cast<H*>(__W1) + blockIdx.x * (HF * HF);
    H *_b1       = reinterpret_cast<H*>(__b1) + blockIdx.x * (mini_batch_size * HF);

    const H *_ln_weight = reinterpret_cast<const H*>(__ln_weight) + (blockIdx.x % NH) * (mini_batch_size * HF);
    const H *_ln_bias   = reinterpret_cast<const H*>(__ln_bias) + (blockIdx.x % NH) * (mini_batch_size * HF);

    const H *_make_last_b_matrix         = reinterpret_cast<const H*>(__make_last_b_matrix);
    const H *_make_last_eta_1_matrix     = reinterpret_cast<const H*>(__make_last_eta_1_matrix);

    const H *_XV   = reinterpret_cast<const H*>(__XV) + blockIdx.x * (n_mini_batch * mini_batch_size * HF);
    const H *_XK   = reinterpret_cast<const H*>(__XK) + blockIdx.x * (n_mini_batch * mini_batch_size * HF);
    const H *_XQ   = reinterpret_cast<const H*>(__XQ) + blockIdx.x * (n_mini_batch * mini_batch_size * HF);
    const H *_Eta  = reinterpret_cast<const H*>(__Eta) + blockIdx.x * (n_mini_batch * mini_batch_size * mini_batch_size);
    H *_Output = reinterpret_cast<H*>(__Output) + blockIdx.x * (n_mini_batch * mini_batch_size * HF);

    // This is the CUDA shared memory
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    st_hf<1, 4, ducks::st_layout::swizzle> (&XK_smem)[1] = al.allocate<st_hf<1, 4, ducks::st_layout::swizzle>, 1>();
    st_hf<1, 4, ducks::st_layout::swizzle> (&XQ_smem)[1] = al.allocate<st_hf<1, 4, ducks::st_layout::swizzle>, 1>();
    st_hf<1, 4, ducks::st_layout::swizzle> (&XV_smem)[1] = al.allocate<st_hf<1, 4, ducks::st_layout::swizzle>, 1>();
    st_hf<1, 1, ducks::st_layout::swizzle> (&Eta_smem)[1] = al.allocate<st_hf<1, 1, ducks::st_layout::swizzle>, 1>();

    rt_hf<4, 4, kittens::ducks::rt_layout::col> W1_reg;
    load(W1_reg, _W1, W1_reg.cols);

    rt_hf<1, 4> b1_reg;
    load(b1_reg, _b1, b1_reg.cols);

    rt_hf<1, 4> ln_w_reg;
    rt_hf<1, 4> ln_b_reg;
    load(ln_w_reg, _ln_weight, ln_w_reg.cols);
    load(ln_b_reg, _ln_bias, ln_b_reg.cols);

    rt_hf<1, 1> make_last_b_matrix;
    rt_hf<1, 4, kittens::ducks::rt_layout::col> make_last_eta_1_matrix_col;
    // make_last_b_matrix: broadcast last row of b_bar
    load(make_last_b_matrix, _make_last_b_matrix, make_last_b_matrix.cols);
    // make_last_eta_1_matrix_col: broadcast last col of eta_transposed for multiplying X1: [bs,HF]
    load(make_last_eta_1_matrix_col, _make_last_eta_1_matrix, make_last_eta_1_matrix_col.cols);

    for (int i = 0; i < n_mini_batch; i++) {

        // Prefetch a mini-batch into shared memory
        load(XV_smem[0],  _XV  + i * X_STRIDE,  64);
        load(XK_smem[0],  _XK  + i * X_STRIDE,  64);
        load(XQ_smem[0],  _XQ  + i * X_STRIDE,  64);
        load(Eta_smem[0], _Eta + i * Eta_STRIDE, 16);

        // Z1 = XK @ W1 + b1
        rt_hf<1, 4> XK_reg;
        load(XK_reg, XK_smem[0]);

        rt_hf<1, 4> Z1_reg;
        mma_AB(Z1_reg, XK_reg, W1_reg, b1_reg);

        rt_hf<1, 4> l2_target_reg;
        load(l2_target_reg, XV_smem[0]);
        // l2_tgt = XV - XK
        sub(l2_target_reg, l2_target_reg, XK_reg);

        rt_hf<1, 4> dl_dZ1;
        ln_fused_l2_bwd_fp16(HF, Z1_reg, l2_target_reg, ln_w_reg, ln_b_reg, dl_dZ1);

        // b1_bar = b1 - (eta * Attn_b) @ dl_dZ1
        rt_hf<1, 4, ducks::rt_layout::col> &dl_dZ1_col = swap_layout_inplace(dl_dZ1);
        rt_hf<1, 4> delta_b1_reg;
        zero(delta_b1_reg);
        rt_hf<1, 1> eta_reg;
        load(eta_reg, Eta_smem[0]);
        rt_hf<1, 1> Attn1_reg;
        make_causal(eta_reg, eta_reg, base_types::constants<half>::zero());
        mma_AB(delta_b1_reg, eta_reg, dl_dZ1_col, delta_b1_reg);
        sub(b1_reg, b1_reg, delta_b1_reg);

        // Z2 = XQ @ W1 - (eta * Attn1) @ dl_dZ1 + b1_bar
        rt_hf<1, 4> XQ_reg;
        load(XQ_reg, XQ_smem[0]);

        zero(Attn1_reg);
        mma_ABt(Attn1_reg, XQ_reg, XK_reg, Attn1_reg);

        make_causal(Attn1_reg, Attn1_reg, base_types::constants<half>::zero());
        mul(Attn1_reg, eta_reg, Attn1_reg);

        rt_hf<1, 4> Z1_bar_term_1_reg;
        mma_AB(Z1_bar_term_1_reg, XQ_reg, W1_reg, b1_reg);

        rt_hf<1, 4> Z1_bar_term_2_reg;
        zero(Z1_bar_term_2_reg);
        mma_AB(Z1_bar_term_2_reg, Attn1_reg, dl_dZ1_col, Z1_bar_term_2_reg);

        sub(Z1_bar_term_1_reg, Z1_bar_term_1_reg, Z1_bar_term_2_reg);

        rt_hf<1, 4> &Z1_bar_reg = Z1_bar_term_1_reg;
        rt_hf<1, 4> LN_out_bar_reg;
        LN_fwd_fp16(HF, Z1_bar_reg, ln_w_reg, ln_b_reg, LN_out_bar_reg);

        // Output = XQ + LN(Z1_bar)
        add(LN_out_bar_reg, LN_out_bar_reg, XQ_reg);

        store(_Output + i * mini_batch_size * HF, LN_out_bar_reg, LN_out_bar_reg.cols);

        // delta_W1 of the last token in the mini-batch
        // delta_W1 = (eta_mini_batch_last * XK_mini_batch).transpose(-1, -2) @ dl_dZ1
        rt_hf<1, 4> eta_1_last_reg;
        zero(eta_1_last_reg);
        rt_hf<1, 1> &eta_transpose_reg = transpose_inplace(eta_reg);
        mma_AB(eta_1_last_reg, eta_transpose_reg, make_last_eta_1_matrix_col, eta_1_last_reg);
        mul(XK_reg, XK_reg, eta_1_last_reg);

        rt_hf<1, 4, kittens::ducks::rt_layout::col> &XK_col_reg = swap_layout_inplace(XK_reg);
        rt_hf<4, 4> delta_W1_reg;
        zero(delta_W1_reg);
        mma_AtB(delta_W1_reg, XK_col_reg, dl_dZ1_col, delta_W1_reg);

        // W1_new = W1 - delta_W1
        rt_hf<4, 4, kittens::ducks::rt_layout::col> &delta_W1_col_reg = swap_layout_inplace(delta_W1_reg);
        sub(W1_reg, W1_reg, delta_W1_col_reg);

        // delta_b1 = b1_bar[-1]
        // b1_new = b1 - delta_b1
        rt_hf<1, 4, kittens::ducks::rt_layout::col> b1_bar_col_reg;
        swap_layout(b1_bar_col_reg, b1_reg);
        zero(b1_reg);
        mma_AB(b1_reg, make_last_b_matrix, b1_bar_col_reg, b1_reg);

    }

    store(_W1, W1_reg, W1_reg.cols);
    store(_b1, b1_reg, b1_reg.cols);

}


void ttt_linear_prefill_fp16(
        torch::Tensor W1,
        torch::Tensor b1,
        torch::Tensor ln_weight,
        torch::Tensor ln_bias,
        torch::Tensor make_last_b_matrix,
        torch::Tensor make_last_eta_1_matrix,
        torch::Tensor XV,
        torch::Tensor XK,
        torch::Tensor XQ,
        torch::Tensor Eta,
        torch::Tensor Output,
        cudaStream_t stream
) {
    auto batch = XV.size(0);
    auto head = XV.size(1);
    auto n_mini_batch = XV.size(2);
    auto mini_batch_size = XV.size(3);
    auto HF = XV.size(4);

    using H = __half;
    using T = c10::Half;
    const int workers = 1;

    auto threads = workers * kittens::WARP_THREADS;

    ttt_linear_prefill_fp16_ker<H, T><<<batch * head, threads, SMEM_BLOCK, stream>>>(
            head, n_mini_batch, mini_batch_size, HF,
            W1.data_ptr<T>(), b1.data_ptr<T>(),
            ln_weight.data_ptr<T>(), ln_bias.data_ptr<T>(),
            make_last_b_matrix.data_ptr<T>(), make_last_eta_1_matrix.data_ptr<T>(),
            XV.data_ptr<T>(), XK.data_ptr<T>(), XQ.data_ptr<T>(), Eta.data_ptr<T>(),
            Output.data_ptr<T>()
    );
}
