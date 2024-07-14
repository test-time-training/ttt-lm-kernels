#include <iostream>
#include <string>
#include <math.h>
#include <assert.h>
#include <cuda_runtime_api.h>
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

# include "../../src/kittens.cuh"
# include "../../src/common/pyutils/torch_helpers.cuh"

// **** ASYn_mini_batch In_mini_batchLUDE *****
#include <cuda/pipeline>
#include <cooperative_groups.h>

using namespace kittens;

__device__ static inline void LN_fwd_fp16(
        const int HF,
        rt_hf<1, 4> &Z1_reg,
        rt_hf<1, 4> &ln_w_reg,
        rt_hf<1, 4> &ln_b_reg,
        rt_hf<1, 4> &LN_out_reg
){

    rt_hf<1, 4>::col_vec Z1_mean_reg;
    row_sum(Z1_mean_reg, Z1_reg);
    div(Z1_mean_reg, Z1_mean_reg, __float2half(float(HF)));

    rt_hf<1, 4> Z1_square_reg;
    sub_row(Z1_square_reg, Z1_reg, Z1_mean_reg);
    mul(Z1_square_reg, Z1_square_reg, Z1_square_reg);

    rt_hf<1, 4>::col_vec Z1_std_reg;
    row_sum(Z1_std_reg, Z1_square_reg);
    div(Z1_std_reg, Z1_std_reg, __float2half(float(HF)));
    add(Z1_std_reg, Z1_std_reg, __float2half(1e-6f));
    sqrt(Z1_std_reg, Z1_std_reg);

    // Z1_hat = (Z - mu) / std
    rt_hf<1, 4> Z1_hat;
    sub_row(Z1_hat, Z1_reg, Z1_mean_reg);
    div_row(Z1_hat, Z1_hat, Z1_std_reg);

    // LN_out = ln_w * Z1_hat + ln_b
    mul(LN_out_reg, Z1_hat, ln_w_reg);
    add(LN_out_reg, LN_out_reg, ln_b_reg);

}

__device__ static inline void ln_fused_l2_bwd_fp16(
        const int HF,
        rt_hf<1, 4> &Z1_reg,
        rt_hf<1, 4> &l2_target_reg,
        rt_hf<1, 4> &ln_w_reg,
        rt_hf<1, 4> &ln_b_reg,
        rt_hf<1, 4> &dl_dZ1
){
    rt_hf<1, 4>::col_vec Z1_mean_reg;
    row_sum(Z1_mean_reg, Z1_reg);
    div(Z1_mean_reg, Z1_mean_reg, __float2half(float(HF)));

    rt_hf<1, 4> Z1_square_reg;
    sub_row(Z1_square_reg, Z1_reg, Z1_mean_reg);
    mul(Z1_square_reg, Z1_square_reg, Z1_square_reg);

    rt_hf<1, 4>::col_vec Z1_std_reg;
    row_sum(Z1_std_reg, Z1_square_reg);
    div(Z1_std_reg, Z1_std_reg, __float2half(float(HF)));
    add(Z1_std_reg, Z1_std_reg, __float2half(1e-6f));
    sqrt(Z1_std_reg, Z1_std_reg);

    // Z1_hat = (Z - mu) / std
    rt_hf<1, 4> Z1_hat;
    sub_row(Z1_hat, Z1_reg, Z1_mean_reg);
    div_row(Z1_hat, Z1_hat, Z1_std_reg);

    // LN_out = ln_w * Z1_hat + ln_b
    rt_hf<1, 4> LN_out_reg;
    mul(LN_out_reg, Z1_hat, ln_w_reg);
    add(LN_out_reg, LN_out_reg, ln_b_reg);

    // dl_dLN_out = LN_out - l2_target
    // dl_dZ1_hat = dl_dLN_out * ln_weight
    rt_hf<1, 4> dl_dZ1_hat;
    sub(dl_dZ1_hat, LN_out_reg, l2_target_reg);
    mul(dl_dZ1_hat, dl_dZ1_hat, ln_w_reg);

    // LN bwd
    // dl_dZ1 = (HF * dl_dZ1_hat -
    //           dl_dZ1_hat.sum(dim=-1, keepdim=True) -
    //           Z1_hat * (dl_dZ1_hat * Z1_hat).sum(dim=-1, keepdim=True)
    //           ) / (std * HF)

    // HF * dl_dZ1_hat
    mul(dl_dZ1, dl_dZ1_hat, __float2half(float(HF)));

    // HF * dl_dZ1_hat - dl_dZ1_hat.sum(dim=-1, keepdim=True)
    rt_hf<1, 4>::col_vec dl_dZ1_vec_term;
    row_sum(dl_dZ1_vec_term, dl_dZ1_hat);
    sub_row(dl_dZ1, dl_dZ1, dl_dZ1_vec_term);

    // Z1_hat * (dl_dZ1_hat * Z1_hat).sum(dim=-1, keepdim=True)
    rt_hf<1, 4> dl_dZ1_term_3;
    mul(dl_dZ1_term_3, dl_dZ1_hat, Z1_hat);
    row_sum(dl_dZ1_vec_term, dl_dZ1_term_3);
    mul_row(dl_dZ1_term_3, Z1_hat, dl_dZ1_vec_term);

    sub(dl_dZ1, dl_dZ1, dl_dZ1_term_3);
    mul(Z1_std_reg, Z1_std_reg, __float2half(float(HF)));
    div_row(dl_dZ1, dl_dZ1, Z1_std_reg);

}
