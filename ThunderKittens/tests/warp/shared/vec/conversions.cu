#include "conversions.cuh"

#ifdef TEST_WARP_SHARED_VEC_CONVERSIONS

struct vec_copy {
    template<int S, int NW>
    using valid = std::bool_constant<NW == 1 && S<=64>; // this is warp-level
    static inline const std::string test_identifier = "shared_vec_convert";
    template<int S, int NW>
    __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int S, int NW>
    __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output) {
        __shared__ kittens::col_vec<kittens::st_bf<S, S>> vec1;
        __shared__ kittens::col_vec<kittens::st_bf<S, S>> vec2;
        kittens::load(vec1, input);
        kittens::copy(vec2, vec1);
        kittens::store(output, vec2);
    }
};

void warp::shared::vec::conversions::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/shared/vec/conversions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
                         
    sweep_size_1d_warp<vec_copy, SIZE>::run(results);
}

#endif